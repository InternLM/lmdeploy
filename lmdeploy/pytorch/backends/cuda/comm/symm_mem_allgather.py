# Copyright (c) OpenMMLab. All rights reserved.
"""Symmetric-memory workspace for 2D all-gather.

The optional Triton kernels are imported only during collective preparation. Their SGLang source reference is in
kernels/cuda/symm_mem_allgather.py.
"""
import importlib

import torch
import torch.distributed as dist

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class SymmetricMemoryAllGather:
    """Own configuration, collective admission and one symmetric arena.

    Construct on every group rank after process-group setup. Only preparation performs allocation and consensus; forward
    never initializes or rebuilds. Re-prepare after coordinated, quiescent device/dtype changes. Runtime rows must match
    across ranks, as required by NCCL all-gather too. With reuse_sync=False, callers must order all ranks' consumers
    through a subsequent same-group collective before reusing the arena.
    """

    # Keep the conservative single-token NCCL policy internal to the provider.
    _MIN_TOKENS = 2
    _WORLD_SIZES = {2, 4, 8}

    def __init__(self, group: dist.ProcessGroup, rank: int, gathered_width: int,
                 *, device: torch.device, dtype: torch.dtype, capacity_bytes: int,
                 dim: int = -1, reuse_sync: bool = True):
        self.device = torch.device(device)
        self.dtype = dtype
        self._prepared = False
        self.group = group
        self._rank = rank
        self._gathered_width = gathered_width
        self._dim = dim
        self._capacity_bytes = capacity_bytes
        self._reuse_sync = reuse_sync
        self._state = None
        self._kernels = None
        self._graph_ready_shapes = set()

    def _agree(self, ready: bool, device: torch.device) -> bool:
        flag = torch.tensor(int(ready), dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.group)
        return bool(flag.item())

    def _same_config(self, values: tuple[int, ...], device: torch.device) -> bool:
        lower = torch.tensor(values, dtype=torch.int64, device=device)
        upper = lower.clone()
        dist.all_reduce(lower, op=dist.ReduceOp.MIN, group=self.group)
        dist.all_reduce(upper, op=dist.ReduceOp.MAX, group=self.group)
        return bool(torch.equal(lower, upper))

    def _disabled(self, reason: str) -> bool:
        if self._rank == 0:
            logger.warning('symmetric-memory all-gather disabled: %s', reason)
        return False

    def _valid_handle(self, state, kernels) -> bool:
        """Validate rank metadata and both pointer and signal-pad bounds."""
        try:
            handle = state.symm_mem_hdl
            multicast = int(handle.multicast_ptr or 0)
            signals = handle.signal_pad_ptrs_dev
            signal_addr = int(signals.data_ptr() if hasattr(signals, 'data_ptr') else signals or 0)
            pad_size = getattr(handle, 'signal_pad_size', None)
            if pad_size is None:
                pad_size = kernels.symm_mem.get_signal_pad_size()
            return (int(handle.rank) == self._rank
                    and int(handle.world_size) == state.world_size
                    and multicast > 0 and multicast % 16 == 0
                    and signal_addr > 0 and signal_addr % 8 == 0
                    and (not hasattr(signals, 'numel') or signals.numel() >= state.world_size)
                    and int(pad_size) >= kernels._MAX_BLOCKS * state.world_size * 4)
        except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
            return False

    def _prepare(self, device: torch.device, dtype: torch.dtype) -> bool:
        """Collectively rebuild and admit the arena, outside CUDA graphs."""
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if device.type == 'cuda' and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('symmetric-memory preparation is not graph capturable')
        self.close()
        if device.type != 'cuda' or not torch.cuda.is_available():
            return False
        if not isinstance(self.group, dist.ProcessGroup):
            return False
        # Even disabled ranks participate: mixed flags/dtypes must not leave
        # peers entering an optional rendezvous on their own.
        supported = dtype == torch.bfloat16 or (self._dim == 0 and dtype == torch.float32)
        if not self._agree(supported, device):
            return False
        world_size = dist.get_world_size(self.group)
        # The copy kernel addresses BF16 words, but never converts payloads.
        width = self._gathered_width * dtype.itemsize // torch.bfloat16.itemsize
        alignment = 8 if self._dim == 0 else world_size * 8
        try:
            capability_ok = torch.cuda.get_device_capability(device) >= (9, 0)
        except (RuntimeError, AssertionError):
            capability_ok = False
        valid = (world_size in self._WORLD_SIZES
                 and self._rank == dist.get_rank(self.group)
                 and self._dim in (0, -1)
                 and width > 0 and width % alignment == 0
                 and self._capacity_bytes >= width * torch.bfloat16.itemsize
                 and capability_ok)
        if not self._agree(valid, device):
            return self._disabled('unsupported group size, dtype, width, capacity or device')
        if not self._same_config((width, dtype.itemsize, self._dim,
                                  self._capacity_bytes, int(self._reuse_sync)), device):
            return self._disabled('inconsistent group arena configuration')

        kernels = None
        try:
            kernels = importlib.import_module('lmdeploy.pytorch.kernels.cuda.symm_mem_allgather')
        except (ImportError, RuntimeError) as exc:
            logger.debug('Optional symmetric-memory kernels unavailable: %s', exc)
        if not self._agree(kernels is not None, device):
            return self._disabled('optional symmetric-memory implementation unavailable')

        max_tokens = self._capacity_bytes // (width * torch.bfloat16.itemsize)
        buffer = None
        allocation_ok = False
        try:
            buffer = kernels._allocate_symmetric_buffer(self.group, max_tokens, width, device)
            allocation_ok = (buffer.shape == (max_tokens, width)
                             and buffer.dtype == torch.bfloat16 and buffer.device == device
                             and buffer.is_contiguous() and buffer.storage_offset() == 0
                             and buffer.data_ptr() % 16 == 0)
        except Exception as exc:
            logger.debug('Symmetric allocation failed: %s', exc)
        if not self._agree(allocation_ok, device):
            return self._disabled('a rank could not allocate its arena')

        # Once all ranks commit, rendezvous errors are fatal, not a rank-local
        # reason to fall back and strand peers inside a collective.
        state = kernels.create_state(self.group, self._rank, max_tokens, width, device, comm_buff=buffer)
        if not self._agree(self._valid_handle(state, kernels), device):
            return self._disabled('invalid symmetric-memory handle or unsupported multicast')
        self._kernels = kernels
        self._state = state
        if self._rank == 0:
            logger.warning('multimem all-gather enabled (world_size=%d, gathered_width=%d, max_tokens=%d)',
                           world_size, width, max_tokens)
        return True

    def is_available(self) -> bool:
        """Whether the symmetric-memory workspace is ready for gathering."""
        return self._state is not None

    def prepare(self):
        """Allocate once during setup, never during forward."""
        if not self._prepared:
            self._prepare(self.device, self.dtype)
            self._prepared = True

    def reset(self, device: torch.device, dtype: torch.dtype):
        """Rebuild after a coordinated, quiescent device/dtype transition."""
        device = torch.device(device)
        if (device, dtype) == (self.device, self.dtype):
            return
        self.device, self.dtype = device, dtype
        if self._prepared:
            self._prepared = False
            self.prepare()

    def all_gather(self, input: torch.Tensor, *, dim: int = -1,
                   copy_output: bool = True) -> torch.Tensor | None:
        """Return gathered shards, or None when the input is unsupported.

        With copy_output=False, the output borrows the workspace until its next use.
        """
        state = self._state
        if state is None:
            return None
        width = self._gathered_width if dim == 0 else self._gathered_width // state.world_size
        if not (dim == self._dim and input.dim() == 2 and input.dtype == self.dtype
                and input.is_contiguous() and input.shape[-1] == width):
            # Shape/dtype/layout contracts must match across ranks, as for NCCL.
            return None
        if input.device != state.device or input.data_ptr() % 16 != 0:
            raise RuntimeError('multimem all-gather device or alignment changed after group-wide admission')
        rows = input.shape[0]
        min_rows = 1 if dim == 0 else self._MIN_TOKENS
        max_rows = state.max_token_num // state.world_size if dim == 0 else state.max_token_num
        if not min_rows <= rows <= max_rows:
            return None
        if rows not in self._graph_ready_shapes and torch.cuda.is_current_stream_capturing():
            return None
        output = self._kernels.all_gather_inner(
            state, input.view(torch.bfloat16), tp_hidden_dim=state.hidden_dim, dim=dim,
            skip_entry_sync=not self._reuse_sync, copy_output=copy_output, _validated=True)
        self._graph_ready_shapes.add(rows)
        return output.view(self.dtype)

    def close(self) -> None:
        """Release only after all consumers/graphs have been retired."""
        state = self._state
        if state is not None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('cannot release symmetric-memory arena during capture')
            # A consumer may run on a different stream. This is lifecycle-only,
            # never part of normal forward or CUDA-graph replay.
            torch.cuda.synchronize(state.device)
        self._state = None
        self._kernels = None
        self._graph_ready_shapes.clear()
        self._prepared = False
