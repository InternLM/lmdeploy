# Copyright (c) OpenMMLab. All rights reserved.
import socket
from typing import Literal

import torch
from torch import distributed as dist

from lmdeploy.utils import get_logger

from ...communicator import DeviceCommunicator
from .symm_mem_allreduce import SymmetricMemoryAllReduce

logger = get_logger('lmdeploy')


class CudaCommunicator(DeviceCommunicator):
    """Prepare group-owned providers and dispatch by operation and input
    shape."""

    def __init__(self, cpu_group: dist.ProcessGroup, device_group: dist.ProcessGroup, *,
                 group_name: Literal['tp', 'dcp']):
        super().__init__(device_group=device_group)
        self._cpu_group = cpu_group
        self._all_reduce_provider = SymmetricMemoryAllReduce(cpu_group) if group_name == 'tp' else None
        if self._all_reduce_provider is not None:
            self._check_all_reduce_provider()
        if self._all_reduce_provider is not None:
            self._all_reduce_provider.prepare()
            self._check_all_reduce_provider()
        if dist.get_rank(device_group) == 0:
            logger.info('Communication backend: auto group=%s symmetric_all_reduce=%s device=%s',
                        group_name, self._all_reduce_provider is not None, torch.cuda.get_device_name())

    def _check_all_reduce_provider(self):
        """Keep or retire symmetric all-reduce consistently on every rank."""
        provider = self._all_reduce_provider
        flags = [None] * dist.get_world_size(self._cpu_group)
        dist.all_gather_object(flags, provider.is_available(), group=self._cpu_group)
        if not all(flags):
            provider.close()
            self._all_reduce_provider = None
            logger.info('Symmetric-memory all-reduce unavailable; using process-group fallback')

    def all_reduce_(self, input: torch.Tensor):
        if self._all_reduce_provider is not None and self._all_reduce_provider.all_reduce_(input):
            return
        super().all_reduce_(input)

    def create_all_gather_workspace(self, gathered_width: int, device: torch.device, dtype: torch.dtype,
                                    *, dim: int = -1):
        from .symm_mem_allgather import SymmetricMemoryAllGather

        workspace = SymmetricMemoryAllGather(
            self.device_group, dist.get_rank(self.device_group), gathered_width,
            device=device, dtype=dtype, capacity_bytes=64 * 1024 * 1024, dim=dim)
        workspace.prepare()
        return workspace

    def all_gather(self, input: torch.Tensor, *, dim: int = -1,
                   workspace=None, copy_output: bool = True) -> torch.Tensor:
        if workspace is not None:
            output = workspace.all_gather(input, dim=dim, copy_output=copy_output)
            if output is not None:
                return output
        return super().all_gather(input, dim=dim)

    def close(self):
        if self._all_reduce_provider is not None:
            self._all_reduce_provider.close()
            self._all_reduce_provider = None


def build_cuda_communicator(cpu_group: dist.ProcessGroup, device_group: dist.ProcessGroup,
                            dist_config, *, group_name: Literal['tp', 'dcp'] = 'tp'):
    """Agree on configuration and topology before optional collective setup."""
    backend = dist_config.communication_backend
    compatible = dist_config.dp == 1 and dist_config.ep == 1 and not dist_config.enable_microbatch
    if not compatible:
        return None

    device = torch.cuda.current_device()
    local = (backend, group_name, socket.gethostname(), device,
             torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device))
    world_size = dist.get_world_size(device_group)
    peers = [None] * world_size
    dist.all_gather_object(peers, local, group=cpu_group)
    if any(peer[:2] != local[:2] for peer in peers):
        raise ValueError('communication_backend and group_name must agree across ranks')
    if backend == 'nccl':
        return None

    same_node = all(peer[2] == local[2] for peer in peers)
    same_device = all(peer[4:] == local[4:] for peer in peers)
    peer_access = (same_node and same_device and len({peer[3] for peer in peers}) == world_size
                   and all(peer[3] == device or torch.cuda.can_device_access_peer(device, peer[3]) for peer in peers))
    flags = [None] * world_size
    dist.all_gather_object(flags, peer_access, group=cpu_group)
    if not all(flags):
        if dist.get_rank(device_group) == 0:
            logger.info('Communication policy: process-group fallback (peer topology unavailable)')
        return None

    return CudaCommunicator(cpu_group=cpu_group, device_group=device_group,
                            group_name=group_name)
