# Copyright (c) OpenMMLab. All rights reserved.
"""Setup-owned LM-head all-gather with a portable NCCL fallback.

The optional Triton kernels are imported only during collective preparation. Their SGLang source reference is in
kernels/cuda/symm_mem_allgather.py.
"""
import importlib
import logging

import torch
import torch.distributed as dist

from lmdeploy.pytorch import envs as _envs

logger = logging.getLogger(__name__)


class MultimemAllGatherer:
    """Own configuration, TP admission and the symmetric arena for one LM-head.

    Construct on every TP rank after process-group setup. Only preparation performs allocation and consensus; forward
    never initializes or rebuilds. Call reset_for_weight after coordinated, quiescent device/dtype changes. Runtime rows
    must match across ranks, as required by NCCL all-gather too.
    """

    # Keep the conservative single-token NCCL policy internal to the provider.
    _MIN_TOKENS = 2
    _WORLD_SIZES = {2, 4, 8}

    def __init__(self, group: dist.ProcessGroup, rank: int, gathered_width: int,
                 device: torch.device, dtype: torch.dtype):
        self._group = group
        self._rank = rank
        self._gathered_width = gathered_width
        self._enabled = _envs.enable_symm_mem_lmhead
        self._capacity_bytes = _envs.symm_mem_lmhead_max_mb * 1024 * 1024
        self._state = None
        self._kernels = None
        self._graph_ready_shapes = set()
        self._weight_contract = None
        self.prepare(device, dtype)

    def _agree(self, ready: bool, device: torch.device) -> bool:
        flag = torch.tensor(int(ready), dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self._group)
        return bool(flag.item())

    def _same_config(self, values: tuple[int, ...], device: torch.device) -> bool:
        lower = torch.tensor(values, dtype=torch.int64, device=device)
        upper = lower.clone()
        dist.all_reduce(lower, op=dist.ReduceOp.MIN, group=self._group)
        dist.all_reduce(upper, op=dist.ReduceOp.MAX, group=self._group)
        return bool(torch.equal(lower, upper))

    def _disabled(self, reason: str) -> bool:
        if self._rank == 0:
            logger.warning('symmetric-memory LM-head disabled: %s', reason)
        return False

    def prepare(self, device: torch.device, dtype: torch.dtype) -> bool:
        """Collectively rebuild and admit the arena, outside CUDA graphs."""
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if device.type == 'cuda' and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('symmetric-memory preparation is not graph capturable')
        self.release()
        self._weight_contract = (device, dtype)
        if device.type != 'cuda' or not torch.cuda.is_available():
            return False
        if not isinstance(self._group, dist.ProcessGroup):
            return False
        # Even disabled ranks participate: mixed flags/dtypes must not leave
        # peers entering an optional rendezvous on their own.
        if not self._agree(self._enabled and dtype == torch.bfloat16, device):
            return False
        world_size = dist.get_world_size(self._group)
        width = self._gathered_width
        try:
            capability_ok = torch.cuda.get_device_capability(device) >= (9, 0)
        except (RuntimeError, AssertionError):
            capability_ok = False
        valid = (world_size in self._WORLD_SIZES
                 and self._rank == dist.get_rank(self._group)
                 and width > 0 and width % (world_size * 8) == 0
                 and self._capacity_bytes >= width * torch.bfloat16.itemsize
                 and capability_ok)
        if not self._agree(valid, device):
            return self._disabled('unsupported TP, dtype, width, capacity or device')
        if not self._same_config((width, self._capacity_bytes), device):
            return self._disabled('inconsistent TP arena configuration')

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
            buffer = kernels._allocate_symmetric_buffer(self._group, max_tokens, width, device)
            allocation_ok = (buffer.shape == (max_tokens, width)
                             and buffer.dtype == torch.bfloat16 and buffer.device == device
                             and buffer.is_contiguous() and buffer.storage_offset() == 0
                             and buffer.data_ptr() % 16 == 0)
        except Exception as exc:
            logger.debug('Symmetric allocation failed: %s', exc)
        if not self._agree(allocation_ok, device):
            return self._disabled('a TP rank could not allocate its arena')

        # Once all ranks commit, rendezvous errors are fatal, not a rank-local
        # reason to fall back and strand peers inside a collective.
        state = kernels.create_state(self._group, self._rank, max_tokens, width, device, comm_buff=buffer)
        if not self._agree(self._valid_handle(state, kernels), device):
            return self._disabled('invalid symmetric-memory handle or unsupported multicast')
        self._kernels = kernels
        self._state = state
        if self._rank == 0:
            logger.warning('multimem all-gather enabled (world_size=%d, gathered_width=%d, max_tokens=%d)',
                           world_size, width, max_tokens)
        return True

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

    def reset_for_weight(self, weight: torch.Tensor) -> None:
        """Notify the provider after a quiescent, TP-wide model transition."""
        if (weight.device, weight.dtype) != self._weight_contract:
            self.prepare(weight.device, weight.dtype)

    def release(self) -> None:
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

    def __call__(self, x: torch.Tensor) -> torch.Tensor | None:
        """Gather admitted logits into an owning output; None means NCCL."""
        state = self._state
        if state is None:
            return None
        if not (x.dim() == 2 and x.dtype == torch.bfloat16 and x.device == state.device
                and x.is_contiguous() and x.data_ptr() % 16 == 0
                and x.shape[-1] * state.world_size == state.hidden_dim):
            # After static TP admission this is a programming error. A local
            # NCCL fallback could disagree with peers launching multimem.
            raise RuntimeError('multimem all-gather input contract changed after TP-wide admission')
        rows = x.shape[0]
        if not self._MIN_TOKENS <= rows <= state.max_token_num:
            return None
        if rows not in self._graph_ready_shapes and torch.cuda.is_current_stream_capturing():
            return None
        output = self._kernels.all_gather_inner(
            state, x, tp_hidden_dim=self._gathered_width, safe=True, _validated=True)
        self._graph_ready_shapes.add(rows)
        return output
