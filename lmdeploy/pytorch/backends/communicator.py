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

    def close(self):
        """Release communicator-owned resources."""
        pass


def build_communicator(cpu_group: dist.ProcessGroup, device_group: dist.ProcessGroup,
                       dist_config):
    """Build the default process-group communicator."""
    return DeviceCommunicator(device_group=device_group)
