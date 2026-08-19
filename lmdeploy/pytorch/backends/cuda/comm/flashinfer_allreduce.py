# Copyright (c) OpenMMLab. All rights reserved.
# Modified from: https://github.com/vllm-project/vllm
import torch
from torch import distributed as dist

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

_MiB = 1024 * 1024
_MAX_TOKEN_NUM = 2048

# Empirical FlashInfer fusion limits used by vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/allreduce_rms_fusion.py
# The separate token cap bounds each process-group workspace.
_FUSION_MAX_SIZE = {
    90: {
        2: 64 * _MiB,
        4: 2 * _MiB,
        8: _MiB // 2,
    },
    100: {
        2: 64 * _MiB,
        4: 32 * _MiB,
        8: _MiB,
    },
    103: {
        2: 64 * _MiB,
        4: 64 * _MiB,
        8: 2 * _MiB,
    },
}
_ONE_SHOT_MAX_SIZE = {
    90: {
        2: 32 * _MiB,
        4: 2 * _MiB,
        8: _MiB // 2,
    },
    100: {
        2: 32 * _MiB,
        4: 4 * _MiB,
        8: _MiB,
    },
    103: {
        2: 32 * _MiB,
        4: 4 * _MiB,
        8: 2 * _MiB,
    },
}


class FlashInferAllReduce:
    """FlashInfer all-reduce operations for one distributed group."""

    def __init__(self, group: dist.ProcessGroup):
        """Initialize lazy FlashInfer state for ``group``."""
        self.group = group
        self._comm = None
        self._workspace = None
        self._hidden_dim = None
        self._dtype = None
        self._disabled = False
        self._world_size = dist.get_world_size(group)
        major, minor = torch.cuda.get_device_capability()
        self._device_capability = major * 10 + minor
        self._max_size = _FUSION_MAX_SIZE.get(self._device_capability, {}).get(self._world_size, 0)
        self._one_shot_max_size = _ONE_SHOT_MAX_SIZE.get(self._device_capability, {}).get(self._world_size, 0)

    def _disable(self, reason):
        """Disable this backend after a process-group-wide setup failure."""
        self._disabled = True
        logger.warning(
            f'Disabling FlashInfer all-reduce for this process group: {reason}')

    def _initialize(self) -> bool:
        """Load and validate the optional FlashInfer communication API."""
        if self._disabled:
            return False
        if self._comm is not None:
            return True
        message = 'FlashInfer all-reduce fusion requires flashinfer-python with the unified comm API.'
        try:
            import flashinfer.comm as comm
            from flashinfer.comm.cuda_ipc import cudart
            if not (hasattr(comm, 'allreduce_fusion') and hasattr(
                    comm, 'create_allreduce_fusion_workspace')):
                raise ImportError(message)

            # Resolve FlashInfer's CUDA runtime before TileLang loads its
            # libcudart shim, which the lazy loader could otherwise select.
            cudart.cudaSetDevice(torch.cuda.current_device())
        except Exception as e:
            self._disable(e)
            return False
        self._comm = comm
        return True

    def is_available(self) -> bool:
        """Whether the fused collective implementation is available."""
        return self._max_size > 0 and self._initialize()

    def supports(self, dtype: torch.dtype) -> bool:
        """Whether this group can handle the given dtype."""
        return dtype in (torch.float16, torch.bfloat16) and self.is_available()

    def _supports_input(self, input: torch.Tensor) -> bool:
        """Whether a flattened input satisfies the FlashInfer launch limits."""
        return (input.nbytes <= self._max_size
                and input.size(0) <= _MAX_TOKEN_NUM
                and input.is_contiguous()
                and self.supports(input.dtype))

    def _get_workspace(self, input: torch.Tensor):
        """Return the workspace for the input shape and dtype."""
        hidden_dim = input.size(-1)
        if self._workspace is not None:
            assert self._hidden_dim == hidden_dim and self._dtype == input.dtype
            return self._workspace

        rank = dist.get_rank(self.group)
        max_token_num = min(_MAX_TOKEN_NUM, self._max_size // input[0].nbytes)
        try:
            workspace = self._comm.create_allreduce_fusion_workspace(
                backend='trtllm',
                world_size=self._world_size,
                rank=rank,
                max_token_num=max_token_num,
                hidden_dim=hidden_dim,
                dtype=input.dtype,
                group=self.group,
            )
            if workspace is None:
                raise RuntimeError('workspace creation returned None')
        except Exception as e:
            self._disable(f'workspace initialization failed: {e}')
            return None

        self._workspace = workspace
        self._hidden_dim = hidden_dim
        self._dtype = input.dtype
        logger.info(
            f'FlashInfer all-reduce workspace initialized: rank={rank}, world_size={self._world_size}, '
            f'max_token_num={max_token_num}, hidden_dim={hidden_dim}')
        return self._workspace

    def all_reduce_(self, input: torch.Tensor) -> bool:
        """All-reduce ``input`` in place, returning whether it was handled."""
        if input.dim() < 2 or not input.is_contiguous():
            return False
        input_2d = input.flatten(0, -2)
        if not self._supports_input(input_2d):
            return False

        workspace = self._get_workspace(input_2d)
        if workspace is None:
            return False
        output = self._comm.allreduce_fusion(
            input=input_2d,
            workspace=workspace,
            pattern=self._comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            trigger_completion_at_end=True,
            use_oneshot=input_2d.nbytes <= self._one_shot_max_size,
        )
        input_2d.copy_(output)
        return True

    def fused_all_reduce_residual_rms_norm(self,
                                           input: torch.Tensor,
                                           residual: torch.Tensor,
                                           weight: torch.Tensor,
                                           eps: float):
        """Fuse all-reduce, residual addition and RMSNorm when supported."""
        if weight.dtype != input.dtype:
            return None
        input_2d = input.flatten(0, -2)
        residual_2d = residual.flatten(0, -2)
        if (not self._supports_input(input_2d)
                or not residual_2d.is_contiguous()
                or not weight.is_contiguous()):
            return None

        norm_out = torch.empty_like(input_2d)
        residual_out = torch.empty_like(residual_2d)
        workspace = self._get_workspace(input_2d)
        if workspace is None:
            return None
        self._comm.allreduce_fusion(
            input=input_2d,
            workspace=workspace,
            pattern=self._comm.AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=True,
            # Keep completion conservative; early signaling requires separate
            # validation for the one-shot and two-shot paths.
            trigger_completion_at_end=True,
            use_oneshot=input_2d.nbytes <= self._one_shot_max_size,
            residual_in=residual_2d,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=weight,
            rms_eps=eps,
        )
        return norm_out.view_as(input), residual_out.view_as(residual)

    def close(self):
        """Release the group-bound FlashInfer workspace."""
        if self._workspace is not None:
            self._workspace.destroy()
            self._workspace = None
