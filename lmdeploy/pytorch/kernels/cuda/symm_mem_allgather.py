# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Symmetric-memory ``multimem.st`` all-gather along the hidden (last) dim.

Each rank stores its ``[T, H/TP]`` shard into a multicast buffer in one NVLink
pass instead of an NCCL ring; ``create_state`` rendezvous once so launches are
CUDA-graph capturable.

Adapted from SGLang (Apache-2.0):
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

# Each thread moves _NUMEL_PER_THREAD bf16 via one 128-bit multimem op; the
# grid-strided block count is tunable in [_MIN_BLOCKS, _MAX_BLOCKS].
_BLOCK_THREADS = 1024
_NUMEL_PER_THREAD = 8
_MIN_BLOCKS = 4
_MAX_BLOCKS = 32
_TARGET_GRID_STRIDE_ITERS = 4
# A separate one-CTA barrier is cheaper once payload parallelism would make
# the per-CTA signal protocol issue many duplicate cross-rank CAS operations.
# The threshold is deliberately conservative; ``B<16`` retains the original
# single-kernel path and avoids paying two extra launches for tiny decode M.
_SINGLE_BARRIER_MIN_BLOCKS = 16


# ------------------------------------------------------------------------------
# Low-level PTX helpers
# ------------------------------------------------------------------------------


@triton.jit
def _multimem_st_128(multicast_ptrs, x, y, z, w, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        '=r,l,r,r,r,r,r',
        args=[multicast_ptrs, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _local_ld_128(in_ptr, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @!%p0 bra end;
            ld.relaxed.sys.global.v4.b32 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        '=r,=r,=r,=r,l,r',
        args=[in_ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_tid_x():
    """Return the lane's linear thread id for the x-only launch contract."""
    return tl.inline_asm_elementwise(
        'mov.u32 $0, %tid.x;',
        '=r',
        [],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sync_threads():
    tl.inline_asm_elementwise(
        'bar.sync 0;', '=r', [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _fence_proxy_alias():
    """Order multicast writes before observing the unicast buffer alias."""
    tl.inline_asm_elementwise(
        'fence.proxy.alias;', '=r', [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _send_signal(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            send_signal:
                atom.global.relaxed.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                setp.eq.u32 %p0, %tmp32_0, 0;
                @!%p0 bra send_signal;
        }
        """,
        '=r, l',
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _send_signal_release(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            send_signal:
                atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                setp.eq.u32 %p0, %tmp32_0, 0;
                @!%p0 bra send_signal;
        }
        """,
        '=r, l',
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_signal(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            wait_signal:
                atom.global.sys.relaxed.cas.b32 %tmp32_0, [$1], 1, 0;
                setp.eq.u32 %p0, %tmp32_0, 1;
                @!%p0 bra wait_signal;
        }
        """,
        '=r, l',
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_signal_acquire(addrs):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;

            wait_signal:
                atom.global.sys.acquire.cas.b32 %tmp32_0, [$1], 1, 0;
                setp.eq.u32 %p0, %tmp32_0, 1;
                @!%p0 bra wait_signal;
        }
        """,
        '=r, l',
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _blockwise_barrier(
    signal_pad_ptrs,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    sem: tl.constexpr,
    slot_offset: tl.constexpr = 0,
):
    # Every caller launches an x-only grid and Triton maps ``num_warps`` to an
    # x-only CUDA thread block.  Specializing that invariant removes the
    # generic y/z CTA and thread-index arithmetic from both barriers.
    block_id = tl.program_id(0) + slot_offset
    flat_tid = _get_tid_x()

    # Keep the cast in a distinct SSA value. Triton cannot merge the
    # pointer-typed branch value with the original int64 tensor.
    signal_pad_ptrs_u64 = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))

    if flat_tid < world_size:
        # One lane is assigned to one peer.  Keeping the peer index scalar
        # avoids materializing a rank-wide pointer vector in the Triton IR and
        # shortens the address live range (the NVIDIA backend may scalarize
        # either form, but this spelling also keeps the protocol explicit).
        # Self-send/self-wait remains intentional: it keeps the epoch complete
        # even for a single local rank in test/fake providers.
        peer = flat_tid
        remote_signal_pad_addr = tl.load(signal_pad_ptrs_u64 + peer).to(
            tl.pointer_type(tl.uint32))
        local_signal_pad_addr = tl.load(signal_pad_ptrs_u64 + rank).to(
            tl.pointer_type(tl.uint32))
        send_addr = (remote_signal_pad_addr + block_id * world_size + rank)
        wait_addr = (local_signal_pad_addr + block_id * world_size + peer)

        if sem == 'relaxed':
            _send_signal(send_addr)
            _wait_signal(wait_addr)
        else:
            _send_signal_release(send_addr)
            _wait_signal_acquire(wait_addr)


@triton.jit
def _all_gather_kernel_inner(
    input_ptr,
    multicast_ptr,
    signal_pad_ptr,
    total_tokens,
    hidden_offset,
    LOCAL_HIDDEN: tl.constexpr,
    TOTAL_HIDDEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    SKIP_ENTRY_SYNC: tl.constexpr,
    SKIP_EXIT_SYNC: tl.constexpr,
) -> None:
    if SKIP_ENTRY_SYNC == 0:
        _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE, sem='relaxed')
        _sync_threads()

    chunks_per_row: tl.constexpr = LOCAL_HIDDEN // NUMEL_PER_THREAD
    total_hidden_chunks: tl.constexpr = TOTAL_HIDDEN // NUMEL_PER_THREAD
    hidden_offset_chunks = hidden_offset // NUMEL_PER_THREAD
    total_chunks = total_tokens * chunks_per_row

    pid = tl.program_id(axis=0)
    tid = _get_tid_x()
    block_start = pid * BLOCK_SIZE

    while block_start < total_chunks:
        chunk = block_start + tid
        mask = chunk < total_chunks
        row = chunk // chunks_per_row
        col_chunk = chunk % chunks_per_row

        in_ptr = input_ptr.to(tl.pointer_type(tl.uint64)) + chunk * 2
        out_chunk = row * total_hidden_chunks + hidden_offset_chunks + col_chunk
        out_ptr = (
            multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64)) + out_chunk * 2
        )
        x, y, z, w = _local_ld_128(in_ptr, mask)
        _multimem_st_128(out_ptr, x, y, z, w, mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    # The payload writes through the multicast VA and callers consume through
    # the ordinary symmetric-buffer VA.  Hopper requires an alias-proxy fence
    # before the release/acquire completion handshake.  In split-barrier mode
    # the payload kernel is followed (on the same stream) by a one-CTA
    # completion barrier, so a grid-wide CTA barrier and per-CTA remote CAS are
    # unnecessary here; kernel completion is the grid synchronization point.
    _fence_proxy_alias()
    if SKIP_EXIT_SYNC == 0:
        _sync_threads()
        _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE, sem='acq_rel')


@triton.jit
def _one_block_barrier_kernel(
    signal_pad_ptr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    SLOT: tl.constexpr,
    RELEASE: tl.constexpr,
):
    """One-CTA cross-rank rendezvous for payload launches."""
    if RELEASE:
        _blockwise_barrier(signal_pad_ptr,
                           RANK,
                           WORLD_SIZE,
                           sem='acq_rel',
                           slot_offset=SLOT)
    else:
        _blockwise_barrier(signal_pad_ptr,
                           RANK,
                           WORLD_SIZE,
                           sem='relaxed',
                           slot_offset=SLOT)


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------


@dataclass
class MultimemAllGatherState:
    group: dist.ProcessGroup
    rank_in_group: int
    world_size: int
    device: torch.device
    max_token_num: int
    hidden_dim: int
    comm_buff: torch.Tensor
    # Rendezvous handle; stable for the buffer's lifetime, resolved once.
    symm_mem_hdl: Any


def create_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_tokens: int,
    hidden_size: int,
    device: torch.device | None = None,
    comm_buff: torch.Tensor | None = None,
) -> MultimemAllGatherState:
    """Allocate and rendezvous the symmetric-memory buffer.

    Collective: call
    once outside CUDA-graph capture with identical args on every rank.
    """
    if not isinstance(group, dist.ProcessGroup):
        raise TypeError(f'Expected ProcessGroup, got {type(group)}')
    if max_tokens <= 0:
        raise ValueError(f'max_tokens must be positive, got {max_tokens}')
    if rank_in_group < 0 or rank_in_group >= group.size():
        raise ValueError(
            f'rank_in_group={rank_in_group} is outside group size={group.size()}')
    assert hidden_size % _NUMEL_PER_THREAD == 0, (
        f"hidden_size={hidden_size} must be a multiple of {_NUMEL_PER_THREAD} "
        f"bf16 for 16-byte multimem.st row alignment"
    )
    device = torch.device(
        device or torch.device(f"cuda:{torch.cuda.current_device()}"))
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())

    if comm_buff is None:
        comm_buff = _allocate_symmetric_buffer(group, max_tokens, hidden_size,
                                                device)
    elif (comm_buff.shape != (max_tokens, hidden_size)
          or comm_buff.dtype != torch.bfloat16
          or comm_buff.device != device or not comm_buff.is_contiguous()
          or comm_buff.storage_offset() != 0):
        raise ValueError('preallocated symmetric buffer does not match state')
    hdl = symm_mem.rendezvous(comm_buff, group=group)
    # Do not raise on a rank-local handle mismatch here.  ``rendezvous`` is a
    # TP collective and the caller must be able to run one more TP-wide
    # validity reduction before deciding whether to disable the provider;
    # raising on only the mismatching rank would leave its peers in that
    # reduction (or in the next launch) indefinitely.  The provider validates
    # rank/world-size/pointers collectively after this function returns.
    return MultimemAllGatherState(
        group=group,
        rank_in_group=rank_in_group,
        world_size=group.size(),
        device=device,
        max_token_num=max_tokens,
        hidden_dim=hidden_size,
        comm_buff=comm_buff,
        symm_mem_hdl=hdl,
    )


def _allocate_symmetric_buffer(
    group: dist.ProcessGroup,
    max_tokens: int,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Perform only the local allocation half of state construction."""
    # Pad holds the per-CTA slots for the payload protocol.  Split barriers
    # use slot 0 for entry and slot 1 for a one-CTA completion rendezvous;
    # consequently the latter consumes only the next single block slot (not
    # another ``_MAX_BLOCKS`` range).  The max() never shrinks the pad.
    pad_bytes = _MAX_BLOCKS * group.size() * 4
    current_pad = symm_mem.get_signal_pad_size()
    if current_pad < pad_bytes:
        try:
            # PyTorch requires this setting before the first symmetric
            # allocation in a process.  If another symmetric-memory user has
            # already allocated with a smaller pad, do not turn the situation
            # into a rank-local exception: the caller catches this error and
            # TP-wide disables the optional provider.
            symm_mem.set_signal_pad_size(pad_bytes)
        except RuntimeError:
            if symm_mem.get_signal_pad_size() < pad_bytes:
                raise
    with torch.inference_mode(False), torch.no_grad():
        return symm_mem.empty((max_tokens, hidden_size),
                              dtype=torch.bfloat16,
                              device=device)


@lru_cache(maxsize=256)
def _launch_config(local_numel: int, *, total_tokens: int = 0, world_size: int = 0):
    """Choose the internal shape-aware launch policy, without public knobs."""
    if local_numel <= 0 or local_numel % _NUMEL_PER_THREAD:
        raise ValueError('local_numel must be a positive multiple of eight')
    chunks = local_numel // _NUMEL_PER_THREAD
    block_threads = 256 if chunks <= 2048 else 512 if chunks <= 8192 else _BLOCK_THREADS
    chunks_per_block = block_threads * _TARGET_GRID_STRIDE_ITERS
    num_blocks = (chunks + chunks_per_block - 1) // chunks_per_block
    max_blocks = _MAX_BLOCKS
    if world_size == 2 and 0 < total_tokens <= 32:
        max_blocks = _MIN_BLOCKS
    elif world_size == 2 and 32 < total_tokens <= 64:
        max_blocks = 8
    elif world_size >= 4 and 0 < total_tokens <= 8:
        max_blocks = 8
    elif world_size >= 4 and 8 < total_tokens <= 32:
        max_blocks = 16
    return min(max_blocks, max(_MIN_BLOCKS, num_blocks)), block_threads, block_threads // 32, _NUMEL_PER_THREAD


def _use_single_barrier(num_blocks: int, world_size: int) -> bool:
    """Keep small groups in one kernel; amortize TP8's larger signal
    traffic."""
    return world_size >= 8 and num_blocks >= _SINGLE_BARRIER_MIN_BLOCKS


def all_gather_inner(
    state: MultimemAllGatherState,
    hidden_states: torch.Tensor,
    tp_hidden_dim: int,
    skip_entry_sync: bool = False,
    safe: bool = True,
    *,
    _validated: bool = False,
) -> torch.Tensor:
    """Gather ``[T, H/TP]`` shards into ``[T, H]`` along the hidden dim.

    ``tp_hidden_dim`` is the gathered width ``H``. Returns a clone when ``safe``,
    else a view into the symmetric buffer (valid until the next collective).
    ``_validated`` is reserved for the admitted provider,
    whose admission check already enforces the immutable dtype/layout/width
    contract.  Public/direct callers retain all defensive assertions.
    """
    world_size = state.world_size
    if not _validated:
        assert hidden_states.dtype == torch.bfloat16, 'Only bfloat16 is supported'
        assert hidden_states.is_contiguous(), 'hidden_states must be contiguous'
        assert hidden_states.data_ptr() % 16 == 0, (
            f"hidden_states.data_ptr()={hex(hidden_states.data_ptr())} must be "
            f"16-byte aligned for 128-bit multimem.st"
        )
        assert (
            tp_hidden_dim % world_size == 0
        ), f"tp_hidden_dim={tp_hidden_dim} must be divisible by world_size={world_size}"
    local_hidden = tp_hidden_dim // world_size
    total_tokens, in_hidden = hidden_states.shape
    if not _validated:
        assert local_hidden % _NUMEL_PER_THREAD == 0, (
            f"per-rank hidden shard ({local_hidden}) must be a multiple of "
            f"{_NUMEL_PER_THREAD} bf16"
        )
        assert tp_hidden_dim <= state.hidden_dim, (
            f"comm buffer too narrow: tp_hidden_dim={tp_hidden_dim} > "
            f"state.hidden_dim={state.hidden_dim}"
        )
        assert (
            in_hidden == local_hidden
        ), f"input hidden ({in_hidden}) != this rank's shard ({local_hidden})"
        assert (
            total_tokens <= state.max_token_num
        ), f"total_tokens={total_tokens} exceeds max_token_num={state.max_token_num}"

    hidden_offset = local_hidden * state.rank_in_group
    symm_mem_hdl = state.symm_mem_hdl
    num_blocks, block_size, num_warps, numel_per_thread = _launch_config(
        total_tokens * local_hidden,
        total_tokens=total_tokens,
        world_size=world_size,
    )
    split_barrier = _use_single_barrier(num_blocks, world_size)
    if split_barrier and not skip_entry_sync:
        # The one-CTA entry barrier is launched before the payload kernel on
        # the same stream.  It retains the reuse protection of the original
        # protocol while avoiding one barrier per payload CTA.
        barrier_inner(state, slot=0, release=False)
    grid = (num_blocks, 1, 1)
    _all_gather_kernel_inner[grid](
        input_ptr=hidden_states,
        multicast_ptr=symm_mem_hdl.multicast_ptr,
        signal_pad_ptr=symm_mem_hdl.signal_pad_ptrs_dev,
        total_tokens=total_tokens,
        hidden_offset=hidden_offset,
        LOCAL_HIDDEN=local_hidden,
        TOTAL_HIDDEN=state.hidden_dim,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        SKIP_ENTRY_SYNC=1 if (skip_entry_sync or split_barrier) else 0,
        SKIP_EXIT_SYNC=1 if split_barrier else 0,
        num_warps=num_warps,
    )
    if split_barrier:
        # A kernel boundary on one CUDA stream waits for every payload CTA.
        # The completion barrier can therefore be a single CTA and still
        # publish the whole grid before the output view is returned.
        barrier_inner(state, slot=1, release=True)
    output = state.comm_buff[:total_tokens, :tp_hidden_dim]
    return output.clone() if safe else output


def barrier_inner(state: MultimemAllGatherState, *, slot: int,
                  release: bool) -> None:
    """Synchronize ranks without moving payload data.

    ``slot`` selects independent signal-pad storage. Slot zero is the entry
    epoch and slot one is the split-payload completion epoch.
    """
    if slot not in (0, 1):
        raise ValueError(f'barrier slot must be 0 or 1, got {slot}')
    handle = state.symm_mem_hdl
    _one_block_barrier_kernel[(1, )](
        signal_pad_ptr=handle.signal_pad_ptrs_dev,
        RANK=state.rank_in_group,
        WORLD_SIZE=state.world_size,
        SLOT=slot,
        RELEASE=1 if release else 0,
        num_warps=1,
    )
