# Copyright (c) OpenMMLab. All rights reserved.
# Modified from: https://github.com/vllm-project/vllm
import torch
from torch import distributed as dist

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

_MiB = 1024 * 1024

_MAX_SIZES = {
    90: {
        2: 64 * _MiB,
        4: 32 * _MiB,
        6: 64 * _MiB,
        8: 64 * _MiB,
    },
    100: {
        2: 8 * _MiB,
        4: 32 * _MiB,
        6: 128 * _MiB,
        8: 128 * _MiB,
    },
    103: {
        2: 4 * _MiB,
        4: 32 * _MiB,
        6: 32 * _MiB,
        8: 64 * _MiB,
    },
}

_MULTIMEM_WORLD_SIZES = {
    90: {4, 6, 8},
    100: {6, 8},
    103: {6, 8},
}


class SymmetricMemoryAllReduce:
    """Hopper multimem all-reduce backed by PyTorch symmetric memory."""

    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self._buffer = None
        self._enabled = False
        self._max_size = 0

        world_size = dist.get_world_size(group)
        major, minor = torch.cuda.get_device_capability()
        device_capability = major * 10 + minor
        self._max_size = _MAX_SIZES.get(device_capability, {}).get(world_size, 0)
        if self._max_size == 0:
            return
        self._use_multimem = world_size in _MULTIMEM_WORLD_SIZES[device_capability]

        try:
            import torch.distributed._symmetric_memory as symm_mem
            op_name = 'multimem_all_reduce_' if self._use_multimem else 'two_shot_all_reduce_'
            if not (hasattr(symm_mem, 'empty') and hasattr(symm_mem, 'rendezvous')
                    and hasattr(torch.ops.symm_mem, op_name)):
                logger.warning('PyTorch symmetric-memory all-reduce is unavailable in this PyTorch build.')
                return
            self._buffer = symm_mem.empty(
                self._max_size // torch.bfloat16.itemsize,
                dtype=torch.bfloat16,
                device=torch.device('cuda', torch.cuda.current_device()),
            )
            handle = symm_mem.rendezvous(self._buffer, group.group_name)
        except (ImportError, RuntimeError) as e:
            logger.warning(f'PyTorch symmetric-memory all-reduce is unavailable: {e}')
            self._buffer = None
            return

        if getattr(handle, 'multicast_ptr', 0) == 0:
            logger.warning('PyTorch symmetric-memory all-reduce is unavailable: multicast is not supported.')
            self._buffer = None
            return
        self._enabled = True
        if dist.get_rank(group) == 0:
            logger.info(f'Using PyTorch symmetric-memory all-reduce for TP{world_size}.')

    def supports(self, dtype: torch.dtype) -> bool:
        """Whether this group can all-reduce the given dtype."""
        return self._enabled and dtype == torch.bfloat16

    def is_available(self) -> bool:
        """Whether the optimized collective implementation is available."""
        return self._enabled

    def all_reduce_(self, input: torch.Tensor) -> bool:
        """All-reduce ``input`` in place, returning whether it was handled."""
        if (not self.supports(input.dtype) or not input.is_contiguous()
                or input.nbytes > self._max_size or input.nbytes % 4 != 0):
            return False

        buffer = self._buffer[:input.numel()]
        buffer.copy_(input.view(-1))
        if self._use_multimem:
            torch.ops.symm_mem.multimem_all_reduce_(buffer, 'sum', self.group.group_name)
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(buffer, 'sum', self.group.group_name)
        input.copy_(buffer.view_as(input))
        return True

    def close(self):
        """Release the symmetric buffer."""
        self._buffer = None
        self._enabled = False
