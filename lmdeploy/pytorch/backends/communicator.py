# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

import torch
from torch import distributed as dist


class DeviceCommunicator:
    """Device-neutral collective operations for one process group.

    Workspace creation returns a prepared, caller-owned workspace when available. Collective methods execute with native
    fallback when an optimized implementation cannot handle the input.
    """

    def __init__(self, device_group: dist.ProcessGroup):
        self.device_group = device_group

    def all_reduce_(self, input: torch.Tensor):
        """All-reduce ``input`` in place."""
        dist.all_reduce(input, group=self.device_group)

    def all_gather(self, input: torch.Tensor, *, dim: int = -1,
                   workspace=None, copy_output: bool = True) -> torch.Tensor:
        """Gather 2D inputs along dimension 0 or -1 on every group rank.

        With copy_output=False, optimized output may borrow the workspace until its next use. Consume it on the same
        stream. Native gathering always returns an owned output for multi-rank groups.
        """
        if input.dim() != 2 or dim not in (0, -1):
            raise ValueError('all_gather requires a 2D input and dim=0 or dim=-1')
        world_size = dist.get_world_size(self.device_group)
        if world_size == 1:
            return input
        input = input.contiguous()
        gathered = input.new_empty(world_size * input.size(0), input.size(1))
        dist.all_gather_into_tensor(gathered, input, group=self.device_group)
        if dim == 0:
            return gathered
        return gathered.view(world_size, *input.shape).transpose(0, 1).reshape(input.size(0), -1)

    def create_all_gather_workspace(self, gathered_width: int, device: torch.device, dtype: torch.dtype,
                                    *, dim: int = -1, reuse_sync: bool = True):
        """Create a caller-owned workspace.

        Disable reuse_sync only when a subsequent same-group collective orders all consumers before the next gather.
        """
        return None

    def close(self):
        """Release group-owned resources; gather workspaces belong to
        callers."""


def build_communicator(cpu_group: dist.ProcessGroup, device_group: dist.ProcessGroup,
                       dist_config, *, group_name: Literal['tp', 'dcp'] = 'tp'):
    """Build the default process-group communicator."""
    return DeviceCommunicator(device_group=device_group)
