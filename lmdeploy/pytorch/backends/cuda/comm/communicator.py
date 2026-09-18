# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import distributed as dist

from lmdeploy.pytorch import envs as _envs

from ...communicator import DeviceCommunicator
from .flashinfer_allreduce import FlashInferAllReduce
from .symm_mem_allreduce import SymmetricMemoryAllReduce


class CudaCommunicator(DeviceCommunicator):
    """Dispatch optional CUDA collectives with a process-group fallback."""

    def __init__(self, cpu_group: dist.ProcessGroup, device_group: dist.ProcessGroup, *,
                 all_reduce_backend: str | None = None, symm_mem_query_gather: bool = False):
        super().__init__(device_group=device_group)
        self._all_reduce_backend = all_reduce_backend
        self._all_reduce = None
        if all_reduce_backend == 'flashinfer':
            self._all_reduce = FlashInferAllReduce(cpu_group)
        elif all_reduce_backend == 'symm_mem':
            self._all_reduce = SymmetricMemoryAllReduce(cpu_group)
        self._symm_mem_query_gather = symm_mem_query_gather
        self._query_gatherer = None
        self._query_shape = None

    def supports_optimized_all_reduce(self) -> bool:
        """Whether an optimized all-reduce implementation is available."""
        return self._all_reduce is not None and self._all_reduce.is_available()

    def supports_fused_all_reduce_residual_rms_norm(self) -> bool:
        """Whether fused all-reduce, residual and RMSNorm is available."""
        return self._all_reduce_backend == 'flashinfer' and self._all_reduce.is_available()

    def try_fused_all_reduce_residual_rms_norm(self,
                                               input: torch.Tensor,
                                               residual: torch.Tensor,
                                               weight: torch.Tensor,
                                               eps: float):
        """Run fused all-reduce, residual and RMSNorm when eligible."""
        if self._all_reduce_backend != 'flashinfer':
            return None
        return self._all_reduce.fused_all_reduce_residual_rms_norm(
            input=input,
            residual=residual,
            weight=weight,
            eps=eps,
        )

    def all_reduce_(self, input: torch.Tensor):
        """Dispatch all-reduce through optimized CUDA backends."""
        if self._all_reduce is not None and self._all_reduce.all_reduce_(input):
            return
        super().all_reduce_(input)

    def close(self):
        """Release communicator-owned workspaces."""
        if self._all_reduce is not None:
            self._all_reduce.close()
        if self._query_gatherer is not None:
            self._query_gatherer.release()
            self._query_gatherer = None

    def prepare_query_gather(self, num_heads: int, head_size: int):
        """Share one bounded query arena across attention layers and MTP."""
        if self._query_gatherer is not None:
            return
        from .symm_mem_allgather import MultimemAllGatherer

        self._query_shape = (num_heads, head_size)
        self._query_gatherer = MultimemAllGatherer(
            self.device_group, dist.get_rank(self.device_group),
            num_heads * head_size * dist.get_world_size(self.device_group),
            torch.device('cuda', torch.cuda.current_device()), torch.bfloat16,
            enabled=self._symm_mem_query_gather, capacity_bytes=64 * 1024 * 1024)

    def gather_query(self, query: torch.Tensor) -> torch.Tensor:
        """Gather query heads, borrowing the symmetric arena when eligible."""
        if (self._query_gatherer is not None and query.dtype == torch.bfloat16
                and tuple(query.shape[1:]) == self._query_shape and query.is_contiguous()):
            # Same-stream attention finishes before the next gather's entry
            # barrier permits any rank to overwrite the borrowed arena.
            output = self._query_gatherer(query.flatten(1), safe=False)
            if output is not None:
                return output.view(query.size(0), -1, query.size(2))
        return super().gather_query(query)


def should_try_symm_mem(dist_config) -> bool:
    """Whether this configuration is a symmetric-memory candidate."""
    return (_envs.enable_symm_mem_allreduce and dist_config.dp == 1
            and dist_config.ep == 1 and dist_config.attn_tp > 1
            and not dist_config.enable_microbatch)


def build_cuda_communicator(cpu_group: dist.ProcessGroup, device_group: dist.ProcessGroup,
                            dist_config, *, group_roles: tuple[str, ...] = ('tp', )):
    """Build the optional CUDA communicator for a TP or DCP group."""
    compatible = dist_config.dp == 1 and dist_config.ep == 1 and not dist_config.enable_microbatch
    if not compatible:
        return None
    all_reduce_backend = None
    if 'tp' in group_roles:
        if _envs.enable_flashinfer_allreduce and _envs.enable_symm_mem_allreduce:
            raise ValueError('FlashInfer and symmetric-memory all-reduce cannot be enabled together.')
        if _envs.enable_flashinfer_allreduce:
            all_reduce_backend = 'flashinfer'
        elif should_try_symm_mem(dist_config):
            all_reduce_backend = 'symm_mem'
    query_gather = 'dcp' in group_roles and dist_config.dcp > 1
    if all_reduce_backend is None and not query_gather:
        return None
    # DCP ranks prepare together even when disabled, so admission can agree
    # on NCCL fallback if the opt-in flag differs between ranks.
    return CudaCommunicator(cpu_group=cpu_group, device_group=device_group,
                            all_reduce_backend=all_reduce_backend,
                            symm_mem_query_gather=query_gather and _envs.enable_symm_mem_dcp)
