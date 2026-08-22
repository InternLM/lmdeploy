# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import distributed as dist

from lmdeploy.pytorch import envs as _envs

from ...communicator import DeviceCommunicator
from .flashinfer_allreduce import FlashInferAllReduce
from .symm_mem_allreduce import SymmetricMemoryAllReduce


class CudaCommunicator(DeviceCommunicator):
    """Dispatch optional CUDA collectives with a process-group fallback."""

    def __init__(self, cpu_group: dist.ProcessGroup, device_group: dist.ProcessGroup):
        super().__init__(device_group=device_group)
        if _envs.enable_flashinfer_allreduce and _envs.enable_symm_mem_allreduce:
            raise ValueError('FlashInfer and symmetric-memory all-reduce cannot be enabled together.')

        self._flashinfer = (FlashInferAllReduce(cpu_group)
                            if _envs.enable_flashinfer_allreduce else None)
        self._symm_mem = (SymmetricMemoryAllReduce(cpu_group)
                          if _envs.enable_symm_mem_allreduce else None)

    def supports_optimized_all_reduce(self) -> bool:
        """Whether an optimized all-reduce implementation is available."""
        return ((self._flashinfer is not None and self._flashinfer.is_available())
                or (self._symm_mem is not None and self._symm_mem.is_available()))

    def supports_fused_all_reduce_residual_rms_norm(self) -> bool:
        """Whether fused all-reduce, residual and RMSNorm is available."""
        return self._flashinfer is not None and self._flashinfer.is_available()

    def try_fused_all_reduce_residual_rms_norm(self,
                                               input: torch.Tensor,
                                               residual: torch.Tensor,
                                               weight: torch.Tensor,
                                               eps: float):
        """Run fused all-reduce, residual and RMSNorm when eligible."""
        if self._flashinfer is None:
            return None
        return self._flashinfer.fused_all_reduce_residual_rms_norm(
            input=input,
            residual=residual,
            weight=weight,
            eps=eps,
        )

    def all_reduce_(self, input: torch.Tensor):
        """Dispatch all-reduce through optimized CUDA backends."""
        if self._flashinfer is not None and self._flashinfer.all_reduce_(input):
            return
        if self._symm_mem is not None and self._symm_mem.all_reduce_(input):
            return
        super().all_reduce_(input)

    def close(self):
        """Release communicator-owned workspaces."""
        if self._flashinfer is not None:
            self._flashinfer.close()
        if self._symm_mem is not None:
            self._symm_mem.close()


def should_try_symm_mem(dist_config) -> bool:
    """Whether this configuration is a symmetric-memory candidate."""
    return (_envs.enable_symm_mem_allreduce and dist_config.dp == 1
            and dist_config.ep == 1 and dist_config.attn_tp > 1
            and not dist_config.enable_microbatch)


def build_cuda_communicator(cpu_group: dist.ProcessGroup, device_group: dist.ProcessGroup,
                            dist_config):
    """Build the optional CUDA communicator for a TP group."""
    compatible = dist_config.dp == 1 and dist_config.ep == 1 and not dist_config.enable_microbatch
    enabled = _envs.enable_flashinfer_allreduce or should_try_symm_mem(dist_config)
    if not compatible or not enabled:
        return None
    return CudaCommunicator(cpu_group=cpu_group, device_group=device_group)
