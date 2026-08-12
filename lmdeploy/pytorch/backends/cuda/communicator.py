# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import distributed as dist

from lmdeploy.pytorch import envs as _envs

from .flashinfer_allreduce import FlashInferAllReduce
from .symm_mem_allreduce import SymmetricMemoryAllReduce


class CudaCommunicator:
    """Dispatch optional CUDA collectives with an NCCL fallback."""

    def __init__(self, cpu_group: dist.ProcessGroup, gpu_group: dist.ProcessGroup):
        self.gpu_group = gpu_group
        self._flashinfer = (FlashInferAllReduce(cpu_group)
                            if _envs.allreduce_use_flashinfer else None)
        self._symm_mem = (SymmetricMemoryAllReduce(cpu_group)
                          if _envs.allreduce_use_symm_mem else None)

    def supports_deferred_allreduce(self, dtype: torch.dtype) -> bool:
        """Whether all-reduce can be deferred to the following RMSNorm."""
        return ((self._flashinfer is not None and self._flashinfer.supports(dtype))
                or (self._symm_mem is not None and self._symm_mem.supports(dtype)))

    def try_fused_allreduce_rmsnorm(self,
                                    input: torch.Tensor,
                                    residual: torch.Tensor,
                                    weight: torch.Tensor,
                                    eps: float):
        """Run FlashInfer all-reduce and RMSNorm fusion when eligible."""
        if self._flashinfer is None:
            return None
        return self._flashinfer.fused_allreduce_rmsnorm(
            input=input,
            residual=residual,
            weight=weight,
            eps=eps,
        )

    def all_reduce_(self, input: torch.Tensor):
        """Select symmetric-memory all-reduce or fall back to NCCL."""
        if self._symm_mem is not None and self._symm_mem.all_reduce_(input):
            return
        dist.all_reduce(input, group=self.gpu_group)

    def close(self):
        """Release communicator-owned workspaces."""
        if self._flashinfer is not None:
            self._flashinfer.close()
        if self._symm_mem is not None:
            self._symm_mem.close()


def build_cuda_communicator(cpu_group: dist.ProcessGroup, gpu_group: dist.ProcessGroup, dist_config):
    """Build the optional CUDA communicator for a TP group."""
    enabled = _envs.allreduce_use_flashinfer or _envs.allreduce_use_symm_mem
    if not enabled or dist_config.dp != 1 or dist_config.ep != 1 or dist_config.enable_microbatch:
        return None
    return CudaCommunicator(cpu_group=cpu_group, gpu_group=gpu_group)
