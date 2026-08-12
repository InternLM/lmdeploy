# Copyright (c) OpenMMLab. All rights reserved.
# Modified from: https://github.com/vllm-project/vllm
import torch
from torch import distributed as dist

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

_MAX_SIZE = 64 * 1024 * 1024


class SymmetricMemoryAllReduce:
    """Hopper multimem all-reduce backed by PyTorch symmetric memory."""

    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self._buffer = None
        self._enabled = False

        world_size = dist.get_world_size(group)
        # Copy-in/copy-out costs more than NCCL at TP4 on H200. Keep this path
        # for TP8, where serving traces show that multimem closes the AR gap.
        if torch.cuda.get_device_capability() != (9, 0) or world_size != 8:
            return

        try:
            import torch.distributed._symmetric_memory as symm_mem
            self._buffer = symm_mem.empty(
                _MAX_SIZE // torch.bfloat16.itemsize,
                dtype=torch.bfloat16,
                device=torch.device('cuda', torch.cuda.current_device()),
            )
            handle = symm_mem.rendezvous(self._buffer, group.group_name)
        except (ImportError, RuntimeError) as e:
            logger.warning(f'PyTorch symmetric-memory all-reduce is unavailable: {e}')
            self._buffer = None
            return

        if handle.multicast_ptr == 0:
            logger.warning('PyTorch symmetric-memory all-reduce is unavailable: multicast is not supported.')
            self._buffer = None
            return
        self._enabled = True

    def supports(self, dtype: torch.dtype) -> bool:
        """Whether this group can all-reduce the given dtype."""
        return self._enabled and dtype == torch.bfloat16

    def all_reduce_(self, input: torch.Tensor) -> bool:
        """All-reduce ``input`` in place, returning whether it was handled."""
        if (not self.supports(input.dtype) or not input.is_contiguous()
                or input.nbytes > _MAX_SIZE or input.nbytes % 4 != 0):
            return False

        buffer = self._buffer[:input.numel()]
        buffer.copy_(input.view(-1))
        torch.ops.symm_mem.multimem_all_reduce_(buffer, 'sum', self.group.group_name)
        input.copy_(buffer.view_as(input))
        return True

    def close(self):
        """Release the symmetric buffer."""
        self._buffer = None
        self._enabled = False
