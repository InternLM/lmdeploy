# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Symmetric-memory ``multimem.st`` all-gather along the hidden (last) dim.

Each rank stores its ``[T, H/TP]`` shard into a multicast buffer in one NVLink
pass instead of an NCCL ring; ``create_state`` rendezvous once so launches are
CUDA-graph capturable.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Each thread moves _NUMEL_PER_THREAD bf16 via one 128-bit multimem op; the
# grid-strided block count is tunable in [_MIN_BLOCKS, _MAX_BLOCKS].
_BLOCK_THREADS = 1024
_NUMEL_PER_THREAD = 8
_MIN_BLOCKS = 4
_MAX_BLOCKS = 32
_SUPPORTED_WORLD_SIZES = {2, 4, 8}


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
def _get_tid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        '=r,=r,=r',
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_ntid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        '=r,=r,=r',
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_flat_tid():
    tid_x, tid_y, tid_z = _get_tid()
    ntid_x, ntid_y, _ = _get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


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
):
    block_id = (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )
    flat_tid = _get_flat_tid()

    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
        tl.pointer_type(tl.uint32)
    )
    send_addrs = remote_signal_pad_addrs + block_id * world_size + rank

    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(
        tl.pointer_type(tl.uint32)
    )
    wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

    if flat_tid < world_size:
        if sem == 'relaxed':
            _send_signal(send_addrs)
            _wait_signal(wait_addrs)
        else:
            _send_signal_release(send_addrs)
            _wait_signal_acquire(wait_addrs)


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
) -> None:
    if SKIP_ENTRY_SYNC == 0:
        _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE, sem='relaxed')
        _sync_threads()

    chunks_per_row: tl.constexpr = LOCAL_HIDDEN // NUMEL_PER_THREAD
    total_hidden_chunks: tl.constexpr = TOTAL_HIDDEN // NUMEL_PER_THREAD
    hidden_offset_chunks = hidden_offset // NUMEL_PER_THREAD
    total_chunks = total_tokens * chunks_per_row

    pid = tl.program_id(axis=0)
    tid = _get_flat_tid()
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

    # The producer writes through the multicast VA and callers consume through
    # the ordinary symmetric-buffer VA.  Hopper requires an alias-proxy fence
    # before the release/acquire completion handshake.
    _fence_proxy_alias()
    _sync_threads()
    _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE, sem='acq_rel')


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
    assert type(group) is dist.ProcessGroup, f"Expected ProcessGroup, got {type(group)}"
    assert hidden_size % _NUMEL_PER_THREAD == 0, (
        f"hidden_size={hidden_size} must be a multiple of {_NUMEL_PER_THREAD} "
        f"bf16 for 16-byte multimem.st row alignment"
    )
    device = torch.device(device or torch.device(f"cuda:{torch.cuda.current_device()}"))
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())

    if comm_buff is None:
        comm_buff = _allocate_symmetric_buffer(group, max_tokens, hidden_size, device)
    elif (comm_buff.shape != (max_tokens, hidden_size) or comm_buff.dtype != torch.bfloat16
          or comm_buff.device != device or not comm_buff.is_contiguous() or comm_buff.storage_offset() != 0):
        raise ValueError('preallocated symmetric buffer does not match state')
    hdl = symm_mem.rendezvous(comm_buff, group=group)
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
    # Pad holds _MAX_BLOCKS * world_size uint32 slots; max() never shrinks it.
    pad_bytes = _MAX_BLOCKS * group.size() * 4
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))
    with torch.inference_mode(False), torch.no_grad():
        return symm_mem.empty((max_tokens, hidden_size), dtype=torch.bfloat16, device=device)


def _launch_config(local_numel: int):
    assert local_numel % _NUMEL_PER_THREAD == 0
    return _MIN_BLOCKS, _BLOCK_THREADS, _BLOCK_THREADS // 32, _NUMEL_PER_THREAD


def all_gather_inner(
    state: MultimemAllGatherState,
    hidden_states: torch.Tensor,
    tp_hidden_dim: int,
    skip_entry_sync: bool = False,
    safe: bool = True,
) -> torch.Tensor:
    """Gather ``[T, H/TP]`` shards into ``[T, H]`` along the hidden dim.

    ``tp_hidden_dim`` is the gathered width ``H``. Returns a clone when ``safe``,
    else a view into the symmetric buffer (valid until the next collective).
    """
    world_size = state.world_size
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
    assert local_hidden % _NUMEL_PER_THREAD == 0, (
        f"per-rank hidden shard ({local_hidden}) must be a multiple of "
        f"{_NUMEL_PER_THREAD} bf16"
    )
    assert tp_hidden_dim <= state.hidden_dim, (
        f"comm buffer too narrow: tp_hidden_dim={tp_hidden_dim} > "
        f"state.hidden_dim={state.hidden_dim}"
    )
    total_tokens, in_hidden = hidden_states.shape
    assert (
        in_hidden == local_hidden
    ), f"input hidden ({in_hidden}) != this rank's shard ({local_hidden})"
    assert (
        total_tokens <= state.max_token_num
    ), f"total_tokens={total_tokens} exceeds max_token_num={state.max_token_num}"

    hidden_offset = local_hidden * state.rank_in_group
    symm_mem_hdl = state.symm_mem_hdl
    num_blocks, block_size, num_warps, numel_per_thread = _launch_config(
        total_tokens * local_hidden
    )
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
        SKIP_ENTRY_SYNC=1 if skip_entry_sync else 0,
        num_warps=num_warps,
    )
    output = state.comm_buff[:total_tokens, :tp_hidden_dim]
    return output.clone() if safe else output


# ------------------------------------------------------------------------------
# Guarded wrapper
# ------------------------------------------------------------------------------


class MultimemAllGatherer:
    """Guarded last-dim multimem all-gather with NCCL fallback.

    Owns one symmetric buffer and admits the input contract collectively on the first eager call. Subsequent calls
    require the same dtype, layout and device on every rank; token count may vary but must be TP-invariant. The returned
    tensor owns its storage so a later collective cannot overwrite logits that are still being consumed on another
    stream.
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
        self._graph_ready = False
        self._runtime_admitted = False
        # None => always NCCL; _UNINIT => build on first eager call.
        self._state = self._UNINIT if enabled else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor | None:
        state = self._state
        if state is self._UNINIT:
            state = self._build(x.device)
            if state is not self._UNINIT:
                self._state = state
        if state is None or state is self._UNINIT:
            return None

        eligible = self._is_static_input_eligible(state, x)
        capturing = torch.cuda.is_current_stream_capturing()
        if not self._runtime_admitted:
            # Admission uses host-visible consensus and cannot run under graph
            # capture. In that uncommon first-call case every rank retains the
            # existing NCCL path captured by ParallelLMHead.
            if capturing:
                return None
            if not self.agree(eligible, state.device):
                self._state = None
                return None
            self._runtime_admitted = True
        elif not eligible:
            raise RuntimeError('multimem all-gather input contract changed after TP-wide admission')

        if not 0 < x.shape[0] <= state.max_token_num:
            # ParallelLMHead presents identical token counts to all TP ranks,
            # so this dynamic capacity fallback cannot split the collective.
            return None
        # State allocation may be prepared before capture while Triton has not
        # compiled this shape yet. Keep that first call on NCCL rather than
        # trying to JIT-compile inside a CUDA Graph.
        if capturing and not self._graph_ready:
            return None
        output = all_gather_inner(
            state,
            x,
            tp_hidden_dim=self._gathered_width,
            skip_entry_sync=False,
            safe=True,
        )
        self._graph_ready = True
        return output

    @staticmethod
    def _is_static_input_eligible(state: MultimemAllGatherState, x: torch.Tensor) -> bool:
        """Check properties that remain stable for one LM-head instance."""
        return (x.dtype == torch.bfloat16 and x.dim() == 2 and x.device == state.device and x.is_contiguous()
                and x.data_ptr() % 16 == 0 and x.shape[-1] % _NUMEL_PER_THREAD == 0
                and x.shape[-1] * state.world_size == state.hidden_dim)

    def prepare(self, device: torch.device | str) -> bool:
        """Collectively allocate and rendezvous the arena before graph
        capture."""
        if self._state is self._UNINIT:
            state = self._build(torch.device(device))
            if state is not self._UNINIT:
                self._state = state
        return self._state is not None and self._state is not self._UNINIT

    def agree(self, local_ready: bool, device: torch.device | str) -> bool:
        """Return a TP-wide setup decision so ranks never split paths."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('TP readiness consensus is not graph capturable')
        ready = torch.tensor(int(local_ready), dtype=torch.int32, device=device)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=self._group)
        return bool(ready.item())

    def release(self) -> None:
        """Drop the device arena so model offload can reclaim its storage."""
        self._state = self._UNINIT if self._enabled else None
        self._graph_ready = False
        self._runtime_admitted = False

    def _build(self, device: torch.device):
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # Can't allocate under capture; retry later.
            return self._UNINIT
        if self._gathered_width % _NUMEL_PER_THREAD != 0:
            return None
        world_size = dist.get_world_size(self._group)
        # tl.arange requires a power-of-two extent. Group size is identical on
        # all ranks, so this fallback decision cannot split the protocol.
        if world_size not in _SUPPORTED_WORLD_SIZES:
            return None

        # Allocate locally first, then make one TP-wide admission decision. No
        # rank may rendezvous while a peer can still take a local fallback.
        comm_buff = None
        allocation_error = None
        try:
            comm_buff = _allocate_symmetric_buffer(self._group, self._max_tokens, self._gathered_width, device)
        except Exception as exc:
            allocation_error = exc
        if not self.agree(comm_buff is not None, device):
            if self._rank == 0:
                logger.warning(
                    'multimem all-gather disabled because a TP rank could not allocate its symmetric arena%s',
                    f': {allocation_error}' if allocation_error else '',
                )
            return None
        if comm_buff is None:
            raise RuntimeError('TP admitted a missing symmetric-memory arena')

        # Every rank is committed from this point. A rendezvous error must
        # propagate; turning it into a rank-local fallback can deadlock peers.
        state = create_state(
            group=self._group,
            rank_in_group=self._rank,
            max_tokens=self._max_tokens,
            hidden_size=self._gathered_width,
            device=device,
            comm_buff=comm_buff,
        )
        multicast_ready = state.symm_mem_hdl.rank == self._rank and state.symm_mem_hdl.multicast_ptr != 0
        if not self.agree(multicast_ready, device):
            if self._rank == 0:
                logger.warning('multimem all-gather disabled (invalid TP-wide multicast handle for world_size=%d)',
                               state.world_size)
            return None
        return state
