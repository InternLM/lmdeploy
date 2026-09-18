# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import distributed as dist


class DeviceCommunicator:
    """Device-neutral collective operations for one process group."""

    def __init__(self, device_group: dist.ProcessGroup):
        self.device_group = device_group

    def supports_optimized_all_reduce(self) -> bool:
        """Whether an optimized all-reduce implementation is available."""
        return False

    def supports_fused_all_reduce_residual_rms_norm(self) -> bool:
        """Whether fused all-reduce, residual and RMSNorm is available."""
        return False

    def try_fused_all_reduce_residual_rms_norm(self,
                                               input: torch.Tensor,
                                               residual: torch.Tensor,
                                               weight: torch.Tensor,
                                               eps: float):
        """Run fused all-reduce, residual and RMSNorm when eligible."""
        return None

    def all_reduce_(self, input: torch.Tensor):
        """All-reduce ``input`` in place."""
        dist.all_reduce(input, group=self.device_group)

    def prepare_query_gather(self, num_heads: int, head_size: int):
        """Prepare an optional head-axis query gather before graph capture."""
        pass

    def gather_query(self, query: torch.Tensor) -> torch.Tensor:
        """Gather query heads across ranks.

        Consume the result on the same stream before the next query gather; CUDA implementations may borrow an arena.
        """
        world_size = dist.get_world_size(self.device_group)
        if world_size == 1:
            return query
        if not query.is_contiguous():
            transposed = query.transpose(0, 1).contiguous()
            gathered = transposed.new_empty(world_size * transposed.size(0), *transposed.shape[1:])
            dist.all_gather_into_tensor(gathered, transposed, group=self.device_group)
            return gathered.transpose(0, 1).contiguous()

        # Contiguous token-major inputs need only the final head-axis packing.
        gathered = query.new_empty(world_size * query.size(0), *query.shape[1:])
        dist.all_gather_into_tensor(gathered, query, group=self.device_group)
        gathered = gathered.view(world_size, *query.shape)
        return gathered.transpose(0, 1).reshape(query.size(0), -1, query.size(2)).contiguous()

    def close(self):
        """Release communicator-owned resources."""
        pass


def build_communicator(cpu_group: dist.ProcessGroup, device_group: dist.ProcessGroup,
                       dist_config, *, group_roles: tuple[str, ...] = ('tp', )):
    """Build the default process-group communicator."""
    return DeviceCommunicator(device_group=device_group)
