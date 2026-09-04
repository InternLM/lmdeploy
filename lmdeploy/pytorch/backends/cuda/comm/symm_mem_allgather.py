# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Symmetric-memory ``multimem.st`` all-gather along the hidden (last) dim.

Each rank stores its ``[T, H/TP]`` shard into a multicast buffer in one NVLink
pass instead of an NCCL ring; ``create_state`` rendezvous once so launches are
CUDA-graph capturable.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from lmdeploy.pytorch import envs as _envs

logger = logging.getLogger(__name__)

# Each thread moves _NUMEL_PER_THREAD bf16 via one 128-bit multimem op; the
# grid-strided block count is tunable in [_MIN_BLOCKS, _MAX_BLOCKS].
_BLOCK_THREADS = 1024
_BLOCK_THREAD_CANDIDATES = (256, 512, 1024)
_NUMEL_PER_THREAD = 8
_MIN_BLOCKS = 4
_MAX_BLOCKS = 32
_TARGET_GRID_STRIDE_ITERS = 4
# A separate one-CTA barrier is cheaper once payload parallelism would make
# the per-CTA signal protocol issue many duplicate cross-rank CAS operations.
# The threshold is deliberately conservative; ``B<16`` retains the original
# single-kernel path and avoids paying two extra launches for tiny decode M.
_SINGLE_BARRIER_MIN_BLOCKS = 16
_SUPPORTED_WORLD_SIZES = {2, 4, 8}


def _is_cuda_graph_capturing() -> bool:
    """Probe capture state without touching the CUDA driver on CPU paths."""
    try:
        available = torch.cuda.is_available()
    except RuntimeError:
        available = False
    if not available:
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        # Some CPU-only test runners expose the CUDA module but no active
        # driver. Treat those calls as eager; normal device checks disable the
        # provider when no CUDA driver is available.
        return False


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
    # reduction (or in the next launch) indefinitely.  ``_build`` validates
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
def _launch_config_cached(local_numel: int, configured_threads: int,
                          configured_blocks: int, autotune: bool,
                          total_tokens: int, world_size: int):
    """Resolve a launch shape once for each static input/configuration pair.

    Decode reuses a small set of flattened token counts.  Keeping the policy
    resolution out of the hot path is useful in its own right, and including
    the environment values in the cache key keeps unit tests and long-lived
    processes that construct more than one tuning profile deterministic.
    """
    if local_numel <= 0 or local_numel % _NUMEL_PER_THREAD != 0:
        raise ValueError(
            f'local_numel must be a positive multiple of '
            f'{_NUMEL_PER_THREAD}, got {local_numel}')
    chunks = local_numel // _NUMEL_PER_THREAD

    if configured_threads and configured_threads not in _BLOCK_THREAD_CANDIDATES:
        raise ValueError(
            'LMDEPLOY_SYMM_MEM_LMHEAD_BLOCK_THREADS must be one of '
            f'{_BLOCK_THREAD_CANDIDATES}, got {configured_threads}')
    if configured_blocks and not _MIN_BLOCKS <= configured_blocks <= _MAX_BLOCKS:
        raise ValueError(
            'LMDEPLOY_SYMM_MEM_LMHEAD_BLOCKS must be in '
            f'[{_MIN_BLOCKS}, {_MAX_BLOCKS}], got {configured_blocks}')

    if configured_threads:
        block_threads = configured_threads
    elif not autotune:
        block_threads = _BLOCK_THREADS
    elif chunks <= 2048:
        block_threads = 256
    elif chunks <= 8192:
        block_threads = 512
    else:
        block_threads = 1024

    if configured_blocks:
        num_blocks = configured_blocks
    elif not autotune:
        num_blocks = _MIN_BLOCKS
    else:
        chunks_per_block = block_threads * _TARGET_GRID_STRIDE_ITERS
        num_blocks = (chunks + chunks_per_block - 1) // chunks_per_block
        max_blocks = _MAX_BLOCKS
        if world_size == 2 and 0 < total_tokens <= 32:
            # On TP2 the signal traffic is already small, so M=6..32 is
            # dominated by the two per-CTA barriers before enough payload is
            # available to profit from a wide grid.  H200 paired sweeps pick
            # four CTAs here; cap at four while retaining the full
            # 32-CTA range for larger payloads and TP4/TP8.
            max_blocks = _MIN_BLOCKS
        elif world_size == 2 and 32 < total_tokens <= 64:
            # A small middle band benefits from a little more payload
            # parallelism, but a full 32-CTA grid would still multiply the
            # entry/exit signal traffic before the copy is saturated.
            max_blocks = 8
        elif world_size >= 4 and 0 < total_tokens <= 8:
            max_blocks = 8
        elif world_size >= 4 and 8 < total_tokens <= 32:
            max_blocks = 16
        num_blocks = min(max_blocks, max(_MIN_BLOCKS, num_blocks))

    return (num_blocks, block_threads, block_threads // 32,
            _NUMEL_PER_THREAD)


def _launch_config(local_numel: int, *, total_tokens: int = 0,
                   world_size: int = 0):
    """Return the cached launch shape for the current process policy."""
    return _launch_config_cached(
        int(local_numel),
        int(_envs.symm_mem_lmhead_block_threads),
        int(_envs.symm_mem_lmhead_blocks),
        bool(_envs.symm_mem_lmhead_autotune),
        int(total_tokens),
        int(world_size),
    )


def _use_single_barrier(num_blocks: int, world_size: int) -> bool:
    """Select the barrier protocol for one payload launch.

    In ``single`` mode the entry and completion rendezvous are separate
    one-CTA kernels around a payload kernel.  Stream ordering then provides a
    grid-wide completion point, while only one CTA performs cross-rank CAS.
    This is safe for arbitrary grid sizes (unlike a per-CTA remote barrier,
    which scales its signal traffic with the number of resident CTAs).
    """
    mode = getattr(_envs, 'symm_mem_lmhead_barrier_mode', 'auto')
    if mode == 'single':
        return True
    if mode == 'per_block':
        return False
    # For TP2/TP4 the two extra launches usually cost more than the small
    # signal reduction at decode sizes.  TP8 issues 4*W signal atomics per CTA
    # across the entry/exit barriers, where the split protocol starts paying
    # back for genuinely large payload grids.
    return (world_size >= 8
            and num_blocks >= _SINGLE_BARRIER_MIN_BLOCKS)


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
    ``_validated`` is reserved for :meth:`MultimemAllGatherer.fast_call`,
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


# ------------------------------------------------------------------------------
# Guarded wrapper
# ------------------------------------------------------------------------------


class MultimemAllGatherer:
    """Guarded last-dim multimem all-gather with NCCL fallback.

    Owns one symmetric buffer built lazily on the first eager call, and uses
    the kernel only when the input fits its dtype/shape/alignment contract.
    The returned tensor owns its storage so a later collective cannot overwrite
    logits that are still being consumed on another stream.
    """

    _UNINIT = object()

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank: int,
        gathered_width: int,
        max_tokens: int,
        *,
        enabled: bool = True,
    ):
        self._group = group
        self._rank = rank
        self._gathered_width = gathered_width
        self._max_tokens = int(max_tokens)
        self._enabled = enabled
        self._logged_dispatch = False
        self._graph_ready = False
        # CUDA Graph capture is shape-specialized. A single bool is not
        # sufficient when warmup first sees M=1 and capture later replays M=8;
        # an unseen shape could trigger Triton compilation inside capture.
        self._graph_ready_shapes = set()
        # Static input admission is a setup-time collective.  It prevents a
        # rank-local alignment/device check from sending one TP rank into NCCL
        # while its peers enter the multimem signal protocol.
        self._runtime_admitted = False

        # None => always NCCL; _UNINIT => build on first eager call.
        self._state = self._UNINIT if enabled else None

    def __call__(self, x: torch.Tensor, *,
                 safe: bool = True) -> torch.Tensor | None:
        """Gather logits, optionally returning the reusable arena view.

        ``safe=True`` keeps the public owning-output contract. ``safe=False``
        is reserved for a serialized consumer that finishes using the result
        before the next collective on this instance.
        """
        state = self.get_state(x)
        if state is None or state is self._UNINIT:
            return None

        eligible = self._is_static_input_eligible(state, x)
        state_device = getattr(state, 'device', x.device)
        # Admission and CUDA-graph capture are mutually exclusive.  The
        # engine warms each decode shape eagerly before capture; an accidental
        # first call inside capture therefore stays on the portable path.
        if x.dim() >= 1:
            shape_key = (int(x.shape[0]), int(x.shape[-1]))
        else:
            shape_key = (0, 0)
        # Once a shape has been launched eagerly, its Triton specialization is
        # ready and the same shape is safe to replay under a CUDA Graph.  Do
        # not query the CUDA driver on every steady-state decode call; on
        # Hopper that host query is measurable at M=1.
        capture_probe_needed = (not self._runtime_admitted
                                or shape_key not in self._graph_ready_shapes)
        capturing = (capture_probe_needed and _is_cuda_graph_capturing())
        if not self._runtime_admitted:
            if capturing:
                return None
            if not self.agree(eligible, state_device):
                # Every rank receives the same result from ``agree``.  Drop
                # this attempted arena and leave lazy admission open: a
                # transient prefill/layout probe must not permanently disable
                # the provider needed by a later valid decode call.
                self.release()
                return None
            if not self._agree_first_shape(x, state_device):
                # The portable all-gather API also requires equal tensor
                # shapes. Converge before either rank enters the multimem
                # protocol when the first TP call violates that invariant;
                # keep the provider retryable after this rejected probe.
                self.release()
                return None
            self._runtime_admitted = True
        elif not eligible:
            raise RuntimeError(
                'multimem all-gather input contract changed after TP-wide '
                'admission')

        # Token count is intentionally the one dynamic dimension.  The model
        # scheduler presents the same flattened row count to every TP rank;
        # a capacity miss consequently follows the same NCCL fallback on all
        # ranks.  Known decode/MTP shapes are pre-reserved at construction.
        if not 0 < x.shape[0] <= state.max_token_num:
            return None
        if capturing:
            return None
        if not self._logged_dispatch:
            self._logged_dispatch = True
            if self._rank == 0:
                logger.warning(
                    'multimem all-gather direct dispatch active '
                    '(tokens=%d, local_width=%d)', x.shape[0], x.shape[-1])
        output = all_gather_inner(
            state,
            x,
            tp_hidden_dim=self._gathered_width,
            skip_entry_sync=False,
            safe=safe,
            _validated=True,
        )
        self._graph_ready = True
        self._graph_ready_shapes.add(shape_key)
        return output

    def fast_call(self, x: torch.Tensor, *, safe: bool = True
                  ) -> torch.Tensor | None:
        """Run an admitted gather with a reduced dispatch check.

        ``ParallelLMHead`` owns a stable BF16 contiguous logits tensor after
        the first eager admission.  The fast path keeps the inexpensive static
        contract guard for fail-closed behavior, while avoiding setup
        collectives and repeated state construction; arbitrary users retain
        the fully validating ``__call__`` API.
        """
        if not self._runtime_admitted:
            return self(x, safe=safe)
        state = self._state
        if state is None or state is self._UNINIT:
            return None
        if x.dim() != 2 or x.device != state.device:
            return self(x, safe=safe)
        if not self._is_static_input_eligible(state, x):
            # After TP-wide admission this is a programming/weight-contract
            # violation, not a rank-local reason to switch to NCCL. Raising
            # keeps all ranks from silently taking different collective paths.
            raise RuntimeError(
                'multimem all-gather input contract changed after TP-wide '
                'admission')
        if not 0 < x.shape[0] <= state.max_token_num:
            return None
        shape_key = (int(x.shape[0]), int(x.shape[-1]))
        if ((not self._runtime_admitted
             or shape_key not in self._graph_ready_shapes)
                and _is_cuda_graph_capturing()):
            return None
        output = all_gather_inner(
            state,
            x,
            tp_hidden_dim=self._gathered_width,
            skip_entry_sync=False,
            safe=safe,
            _validated=True,
        )
        self._graph_ready = True
        self._graph_ready_shapes.add(shape_key)
        return output

    @staticmethod
    def _is_static_input_eligible(state: MultimemAllGatherState,
                                  x: torch.Tensor) -> bool:
        """Check the properties shared by all calls of one LM-head."""
        state_device = getattr(state, 'device', x.device)
        return (x.dtype == torch.bfloat16 and x.dim() == 2
                and x.device == state_device and x.is_contiguous()
                and x.data_ptr() % 16 == 0
                and x.shape[-1] % _NUMEL_PER_THREAD == 0
                and x.shape[-1] * state.world_size == state.hidden_dim)

    def _agree_first_shape(self, x: torch.Tensor,
                           device: torch.device | str) -> bool:
        """Check the first dynamic row/width shape across the TP group."""
        if not self._uses_real_process_group():
            return True
        rows = int(x.shape[0]) if x.dim() >= 1 else -1
        width = int(x.shape[-1]) if x.dim() >= 1 else -1
        shape = torch.tensor((rows, width), dtype=torch.int64, device=device)
        lower = shape.clone()
        upper = shape.clone()
        dist.all_reduce(lower, op=dist.ReduceOp.MIN, group=self._group)
        dist.all_reduce(upper, op=dist.ReduceOp.MAX, group=self._group)
        return bool(torch.equal(lower, upper))

    def get_state(self, x: torch.Tensor):
        """Return the lazily rendezvoused symmetric-memory state."""
        state = self._state
        if state is self._UNINIT:
            # A CUDA tensor of any shape still participates in lazy build.  The
            # subsequent TP-wide static admission in ``__call__`` decides
            # whether this particular input is eligible.  This matters when a
            # malformed tensor reaches only one rank: all ranks must enter the
            # same build/admission sequence instead of one rank returning
            # before its peers reach ``rendezvous``. CPU callers cannot
            # rendezvous a CUDA arena and keep the state open for a later CUDA
            # call.
            if x.device.type != 'cuda':
                # Keep the old lazy-test/portable behavior for an explicitly
                # valid CPU-shaped probe.  Real CUDA callers take the branch
                # below; a CPU input is never sent to a symmetric-memory
                # kernel by the LM-head backend.
                if (x.dim() != 2 or x.dtype != torch.bfloat16
                        or x.shape[-1] % _NUMEL_PER_THREAD != 0):
                    return self._UNINIT
                state = self._build(x.device)
                if state is self._UNINIT or state is None:
                    # A CPU probe must not permanently disable a provider that
                    # may be moved to CUDA later.
                    return self._UNINIT
                self._state = state
                return state
            if not torch.cuda.is_available():
                return self._UNINIT
            if _is_cuda_graph_capturing():
                return self._UNINIT
            state = self._build(x.device)
            if state is not self._UNINIT:
                self._state = state
        return state

    def prepare(self, device: torch.device | str) -> bool:
        """Collectively materialize the arena before graph capture.

        Call this on every rank after the TP process group is ready.  Forward
        retains the lazy path for compatibility, but production warmup uses
        this method so allocation/rendezvous never occurs in a captured region.
        """
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if _is_cuda_graph_capturing():
            raise RuntimeError(
                'symmetric-memory prepare must run before CUDA Graph capture')
        if self._uses_real_process_group():
            # ``prepare`` is normally called with the same phase on every TP
            # rank.  Still resolve a mixed READY/UNINIT (or DISABLED) phase
            # before entering ``_build``: otherwise one rank could skip the
            # symmetric rendezvous while a peer enters it after a device
            # transition.  The two reductions are setup-only and never occur
            # on the decode hot path.
            phase = (2 if self._state is None else
                     1 if self._state is not self._UNINIT else 0)
            phase_min = torch.tensor(phase, dtype=torch.int32, device=device)
            phase_max = phase_min.clone()
            dist.all_reduce(phase_min, op=dist.ReduceOp.MIN, group=self._group)
            dist.all_reduce(phase_max, op=dist.ReduceOp.MAX, group=self._group)
            if int(phase_min.item()) != int(phase_max.item()):
                if int(phase_max.item()) == 2:
                    # A disabled rank cannot safely be rebuilt by its peers;
                    # make the provider uniformly disabled instead.
                    if self._state is not None and self._state is not self._UNINIT:
                        self.release()
                    self._state = None
                    return False
                # READY/UNINIT mismatch: all enabled ranks rebuild the same
                # arena and rendezvous in the same order.
                if self._state is not self._UNINIT:
                    self.release()
                self._state = self._UNINIT if self._enabled else None
            elif int(phase_min.item()) == 1:
                state = self._state
                local_state_ok = (
                    getattr(state, 'device', None) == device
                    and getattr(state, 'rank_in_group', -1) == self._rank
                    and getattr(state, 'group', None) is self._group)
                if not self.agree(local_state_ok, device):
                    # A stale arena (for example after an offload cycle) must
                    # not remain ready on only a subset of TP ranks.
                    self.release()
                    self._state = None
                    return False
                config = torch.tensor(
                    (state.world_size, state.max_token_num,
                     state.hidden_dim, int(self._runtime_admitted)),
                    dtype=torch.int64,
                    device=device,
                )
                config_min = config.clone()
                config_max = config.clone()
                dist.all_reduce(config_min,
                                op=dist.ReduceOp.MIN,
                                group=self._group)
                dist.all_reduce(config_max,
                                op=dist.ReduceOp.MAX,
                                group=self._group)
                if not torch.equal(config_min, config_max):
                    # A stale/mismatched ready arena cannot safely be reused;
                    # converge to the same disabled state rather than letting
                    # one rank launch with a different layout.
                    self.release()
                    self._state = None
                    return False
                return True
            elif int(phase_min.item()) == 2:
                return False

        if self._state is self._UNINIT:
            state = self._build(device)
            if state is not self._UNINIT:
                self._state = state
        return self._state is not None and self._state is not self._UNINIT

    def admit_static(self, device: torch.device | str) -> bool:
        """Mark a prepared, fixed-layout provider ready for hot dispatch.

        ``ParallelLMHead`` calls this only after its TP-wide BF16/shape
        contract and :meth:`prepare` have succeeded.  The dynamic row count
        remains checked by :meth:`fast_call`; skipping the first-call shape
        reductions removes setup collectives from the first decode request.
        Generic users should keep using :meth:`__call__`, which performs the
        defensive admission checks itself.
        """
        if _is_cuda_graph_capturing():
            raise RuntimeError(
                'symmetric-memory admission must run before CUDA Graph '
                'capture')
        state = self._state
        if state is None or state is self._UNINIT:
            return False
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        local_ready = (device == getattr(state, 'device', None)
                       and getattr(state, 'rank_in_group', -1) == self._rank
                       and getattr(state, 'world_size', -1)
                       in _SUPPORTED_WORLD_SIZES
                       and getattr(state, 'hidden_dim', -1)
                       == self._gathered_width)
        if not self.agree(local_ready, device):
            self.release()
            self._state = None
            return False
        self._runtime_admitted = True
        return True

    def _uses_real_process_group(self) -> bool:
        """Whether setup calls can safely use TP collectives."""
        return (dist.is_initialized()
                and isinstance(self._group, dist.ProcessGroup))

    def agree(self, local_ready: bool,
              device: torch.device | str) -> bool:
        """Return TP-wide readiness so ranks never split collective paths.

        This setup-only collective is intentionally forbidden during CUDA
        Graph capture.  Runtime kernels use the immutable decision recorded by
        their owner after this method returns.
        """
        # ``is_current_stream_capturing`` itself can query the CUDA driver and
        # raise on CPU-only/unit-test processes.  Consensus is only used as a
        # setup operation, so avoid that probe when CUDA is unavailable.
        if _is_cuda_graph_capturing():
            raise RuntimeError('TP readiness consensus is not graph capturable')
        # Keep the helper usable by CPU/unit-test callers that construct a
        # provider without a real process group.  Production CUDA paths always
        # pass an initialized ``ProcessGroup`` and therefore take the
        # collective branch below.
        if not self._uses_real_process_group():
            return bool(local_ready)
        device = torch.device(device)
        # A CPU/Gloo test group may call this helper without CUDA.  Production
        # symmetric-memory groups remain CUDA/NCCL and keep the fast device
        # reduction below.
        if device.type == 'cuda' and not torch.cuda.is_available():
            device = torch.device('cpu')
        ready = torch.tensor(int(local_ready), dtype=torch.int32, device=device)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=self._group)
        return bool(ready.item())

    def release(self) -> None:
        """Drop device arenas so model offload can reclaim their storage."""
        state = self._state
        state_device = getattr(state, 'device', None)
        if (state is not self._UNINIT and state is not None
                and state_device is not None
                and torch.device(state_device).type == 'cuda'
                and _is_cuda_graph_capturing()):
            # A captured graph may still hold the arena address.  Clearing the
            # Python reference here would make a later replay use freed
            # storage; release only after capture has ended and the stream is
            # quiescent.
            raise RuntimeError(
                'cannot release symmetric-memory arena during CUDA Graph '
                'capture')
        if (state is not self._UNINIT and state is not None
                and state_device is not None
                and torch.device(state_device).type == 'cuda'
                and torch.cuda.is_available()
                and not _is_cuda_graph_capturing()):
            # Device transitions are quiescent in the model-agent lifecycle,
            # but a prior launch can still be queued on this stream. Synchronize
            # before dropping the symmetric arena.
            # This synchronization is off the steady-state forward path.
            torch.cuda.current_stream(state_device).synchronize()
        self._state = self._UNINIT if self._enabled else None
        self._graph_ready = False
        self._graph_ready_shapes.clear()
        self._runtime_admitted = False

        self._logged_dispatch = False

    def _build(self, device: torch.device):
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if device.type != 'cuda':
            return None
        if _is_cuda_graph_capturing():
            # Can't allocate under capture; retry later.
            return self._UNINIT
        world_size = dist.get_world_size(self._group)
        # Validate all rank-local construction inputs before any rank enters
        # symmetric-memory rendezvous.  A malformed rank must join the same
        # reduction as its peers and cause a uniform NCCL fallback instead of
        # making the remaining ranks wait forever in ``rendezvous``.  The
        # non-ProcessGroup branch is retained for lightweight fake providers;
        # production builders always use the collective branch.
        if self._uses_real_process_group():
            group_rank = dist.get_rank(self._group)
            try:
                capability_ok = (torch.cuda.get_device_capability(device)
                                 >= (9, 0))
            except (RuntimeError, AssertionError):
                capability_ok = False
            local_contract = (
                self._rank == group_rank
                and self._max_tokens > 0
                and self._gathered_width > 0
                and self._gathered_width % _NUMEL_PER_THREAD == 0
                and world_size in _SUPPORTED_WORLD_SIZES
                and capability_ok
            )
            if not self.agree(local_contract, device):
                return None
        elif (self._max_tokens <= 0 or self._gathered_width <= 0
              or self._gathered_width % _NUMEL_PER_THREAD != 0
              or world_size not in _SUPPORTED_WORLD_SIZES):
            return None
        # tl.arange requires a power-of-two extent.  Group size is identical
        # on all ranks, so this fallback decision cannot split the protocol.

        # Allocate locally first, then make one TP-wide admission decision.
        # No rank may enter symmetric-memory rendezvous while a peer can still
        # take a local allocation fallback, otherwise the peer remains stuck
        # in this collective forever.
        comm_buff = None
        allocation_error = None
        try:
            comm_buff = _allocate_symmetric_buffer(
                self._group,
                self._max_tokens,
                self._gathered_width,
                device,
            )
        except Exception as exc:
            allocation_error = exc
        allocation_ok = comm_buff is not None
        if self._uses_real_process_group():
            allocation_ok = False
            try:
                allocation_ok = (isinstance(comm_buff, torch.Tensor)
                                 and comm_buff.shape ==
                                 (self._max_tokens, self._gathered_width)
                                 and comm_buff.dtype == torch.bfloat16
                                 and comm_buff.device == device
                                 and comm_buff.is_contiguous()
                                 and comm_buff.storage_offset() == 0
                                 and comm_buff.data_ptr() % 16 == 0)
            except Exception:
                allocation_ok = False
        if not self.agree(allocation_ok, device):
            if self._rank == 0:
                logger.warning(
                    'multimem all-gather disabled because a TP rank could not '
                    'allocate its symmetric arena%s',
                    f': {allocation_error}' if allocation_error else '')
            # Successful ranks may have allocated a private arena while a
            # peer failed.  Drop that reference before taking NCCL fallback.
            comm_buff = None
            return None
        if comm_buff is None:
            raise RuntimeError('TP admitted a missing symmetric-memory arena')

        # From this point every rank is committed to the collective.  Do not
        # catch rendezvous errors and attempt a local fallback.
        state = create_state(
            group=self._group,
            rank_in_group=self._rank,
            max_tokens=self._max_tokens,
            hidden_size=self._gathered_width,
            device=device,
            comm_buff=comm_buff,
        )

        handle = state.symm_mem_hdl
        # Handle fields are read after a collective rendezvous. Normalize all
        # of them under one guard so a malformed/downstream handle still
        # reaches the final TP-wide ``agree`` instead of throwing on one rank
        # while its peers wait there.
        multicast_ready = False
        try:
            handle_rank_raw = getattr(handle, 'rank', None)
            handle_world_raw = getattr(handle, 'world_size', None)
            handle_rank = (int(handle_rank_raw)
                           if handle_rank_raw is not None else -1)
            handle_world = (int(handle_world_raw)
                            if handle_world_raw is not None else -1)
            multicast_ptr = int(getattr(handle, 'multicast_ptr', 0) or 0)
            signal_ptrs = getattr(handle, 'signal_pad_ptrs_dev', None)
            # Torch 2.13 exposes this field as a raw device address; a few
            # downstream builds wrap it in an address-bearing Tensor.
            if hasattr(signal_ptrs, 'data_ptr'):
                signal_addr = int(signal_ptrs.data_ptr())
            else:
                signal_addr = int(signal_ptrs or 0)
            # Device addresses are unsigned in the real handle.  Requiring a
            # strictly positive aligned value also rejects malformed/mock
            # handles that happen to satisfy ``(-16) % 8 == 0``.
            signal_ready = signal_addr > 0 and signal_addr % 8 == 0
            if signal_ready and hasattr(signal_ptrs, 'numel'):
                signal_ready = int(signal_ptrs.numel()) >= world_size
            signal_pad_size = getattr(handle, 'signal_pad_size', None)
            if signal_pad_size is None:
                # ``signal_pad_size`` is not exposed on every PyTorch handle;
                # in those builds the process-wide symmetric-memory setting is
                # the authoritative bound.  Query it only during admission
                # (never from the decode path), and fail closed when an
                # explicitly available API reports a short/invalid pad.
                get_pad_size = getattr(symm_mem, 'get_signal_pad_size', None)
                if callable(get_pad_size):
                    try:
                        signal_pad_size = int(get_pad_size())
                    except Exception:
                        signal_pad_size = -1
            if signal_pad_size is not None:
                # Per-CTA slots occupy the complete configured range.  A
                # downstream Torch build may expose a smaller pad in its
                # handle (or globally) even when the pointer itself is
                # non-null; reject it before any kernel can index past the
                # allocation.
                try:
                    signal_ready = (signal_ready
                                    and int(signal_pad_size) >=
                                    _MAX_BLOCKS * world_size * 4)
                except (TypeError, ValueError, OverflowError):
                    signal_ready = False
            multicast_ready = (handle_rank == self._rank
                               and handle_world == world_size
                               and multicast_ptr > 0
                               and multicast_ptr % 16 == 0
                               and signal_ready)
        except Exception:
            multicast_ready = False
        if not self.agree(multicast_ready, device):
            if self._rank == 0:
                logger.warning(
                    'multimem all-gather disabled (invalid TP-wide symmetric '
                    'handle for world_size=%d)', state.world_size)
            state.comm_buff = None
            return None
        if self._rank == 0:
            # Ray workers normally inherit the server's WARNING log level.
            logger.warning(
                'multimem all-gather enabled (world_size=%d, '
                'gathered_width=%d, max_tokens=%d)',
                state.world_size,
                self._gathered_width,
                state.max_token_num,
            )
        return state
