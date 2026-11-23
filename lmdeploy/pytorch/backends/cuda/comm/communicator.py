# Copyright (c) OpenMMLab. All rights reserved.
import socket
from typing import Literal

import torch
from torch import distributed as dist

from lmdeploy.pytorch import envs as _envs
from lmdeploy.utils import get_logger

from ...communicator import DeviceCommunicator
from .flashinfer_allreduce import FlashInferAllReduce
from .symm_mem_allreduce import SymmetricMemoryAllReduce

logger = get_logger('lmdeploy')


class CudaCommunicator(DeviceCommunicator):
    """Prepare group-owned providers and dispatch by operation and input
    shape."""

    def __init__(self, cpu_group: dist.ProcessGroup, device_group: dist.ProcessGroup, *,
                 group_name: Literal['tp', 'dcp'],
                 all_reduce_backend: Literal['symm_mem', 'flashinfer'] | None = 'symm_mem',
                 enable_all_gather: bool = True):
        super().__init__(device_group=device_group)
        self._cpu_group = cpu_group
        self._enable_all_gather = enable_all_gather
        self._flashinfer = (FlashInferAllReduce(cpu_group)
                           if group_name == 'tp' and all_reduce_backend == 'flashinfer' else None)
        if self._flashinfer is not None:
            flags = [None] * dist.get_world_size(cpu_group)
            dist.all_gather_object(flags, self._flashinfer.is_available(), group=cpu_group)
            if not all(flags):
                self._flashinfer.close()
                self._flashinfer = None
        self._all_reduce_provider = (SymmetricMemoryAllReduce(cpu_group)
                                    if group_name == 'tp' and all_reduce_backend == 'symm_mem' else None)
        if self._all_reduce_provider is not None:
            self._check_all_reduce_provider()
        if self._all_reduce_provider is not None:
            self._all_reduce_provider.prepare()
            self._check_all_reduce_provider()
        if dist.get_rank(device_group) == 0:
            logger.info('CUDA communication group=%s symmetric_all_reduce=%s flashinfer=%s device=%s',
                        group_name, self._all_reduce_provider is not None,
                        self._flashinfer is not None, torch.cuda.get_device_name())

    def supports_optimized_all_reduce(self) -> bool:
        """Whether TP reductions can be deferred to the following norm."""
        return self._flashinfer is not None or self._all_reduce_provider is not None

    def supports_fused_all_reduce_residual_rms_norm(self) -> bool:
        """Whether FlashInfer can consume the pending TP reduction."""
        return self._flashinfer is not None

    def try_fused_all_reduce_residual_rms_norm(self, input: torch.Tensor, residual: torch.Tensor,
                                              weight: torch.Tensor, eps: float):
        """Try fusion; RMSNorm performs reduction and norm separately on
        fallback."""
        if self._flashinfer is None:
            return None
        return self._flashinfer.fused_all_reduce_residual_rms_norm(
            input=input, residual=residual, weight=weight, eps=eps)

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
        if self._flashinfer is not None and self._flashinfer.all_reduce_(input):
            return
        if self._all_reduce_provider is not None and self._all_reduce_provider.all_reduce_(input):
            return
        super().all_reduce_(input)

    def create_all_gather_workspace(self, gathered_width: int, device: torch.device, dtype: torch.dtype,
                                    *, dim: int = -1, reuse_sync: bool = True, capacity_bytes: int = 64 * 1024 * 1024):
        from .symm_mem_allgather import SymmetricMemoryAllGather

        if not self._enable_all_gather:
            return None
        workspace = SymmetricMemoryAllGather(
            self.device_group, dist.get_rank(self.device_group), gathered_width,
            device=device, dtype=dtype, capacity_bytes=capacity_bytes, dim=dim, reuse_sync=reuse_sync)
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
        if self._flashinfer is not None:
            self._flashinfer.close()
            self._flashinfer = None
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

    # Legacy opt-ins keep their operation-specific meaning, including with the
    # default nccl policy. auto selects symmetric TP reduction unless FlashInfer
    # is explicitly requested; only auto enables DCP gather optimization.
    flashinfer = group_name == 'tp' and _envs.enable_flashinfer_allreduce
    symm_reduce = group_name == 'tp' and _envs.enable_symm_mem_allreduce
    gather = backend == 'auto' or (group_name == 'tp' and _envs.enable_symm_mem_lmhead)
    policy = (backend, group_name, flashinfer, symm_reduce, gather)
    device = torch.cuda.current_device()
    local = (policy, socket.gethostname(), device,
             torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device))
    world_size = dist.get_world_size(device_group)
    peers = [None] * world_size
    dist.all_gather_object(peers, local, group=cpu_group)
    if any(peer[0] != policy for peer in peers):
        raise ValueError('communication_backend, group_name and communication environment must agree across ranks')
    if flashinfer and symm_reduce:
        raise ValueError('FlashInfer and symmetric-memory all-reduce cannot be enabled together.')
    if backend == 'nccl' and not (flashinfer or symm_reduce or gather):
        return None

    same_node = all(peer[1] == local[1] for peer in peers)
    same_device = all(peer[3:] == local[3:] for peer in peers)
    peer_access = (same_node and same_device and len({peer[2] for peer in peers}) == world_size
                   and all(peer[2] == device or torch.cuda.can_device_access_peer(device, peer[2]) for peer in peers))
    flags = [None] * world_size
    dist.all_gather_object(flags, peer_access, group=cpu_group)
    if not all(flags):
        if dist.get_rank(device_group) == 0:
            logger.info('Communication policy: process-group fallback (peer topology unavailable)')
        return None

    reduce_backend = ('flashinfer' if flashinfer else
                      'symm_mem' if backend == 'auto' or symm_reduce else None)
    return CudaCommunicator(cpu_group=cpu_group, device_group=device_group,
                            group_name=group_name, all_reduce_backend=reduce_backend, enable_all_gather=gather)
