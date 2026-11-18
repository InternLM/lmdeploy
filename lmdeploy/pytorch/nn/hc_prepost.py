# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.hc_prepost import HCPrePostBuildSpec
from lmdeploy.pytorch.models.patch import get_build_model_context


@torch.compile(dynamic=True)
def _cast_fp32(x: torch.Tensor) -> torch.Tensor:
    """Vectorize the large HC input conversion without changing reductions."""
    return x.float()


class HcPrePost(nn.Module):
    """DeepSeek-V4 hyper-connection pre/post reduction wrapper."""

    def __init__(self, hc_mult: int, sinkhorn_iters: int = 20, eps: float = 1e-6, *, avoid_gemv: bool = False):
        super().__init__()
        self.avoid_gemv = avoid_gemv
        self.impl = get_backend().build_op(
            HCPrePostBuildSpec(hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

    def prepare_projection_weight(self, weight: torch.Tensor) -> torch.Tensor | None:
        return self.impl.prepare_projection_weight(weight)

    def pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_eps: float,
        norm_weight: torch.Tensor | None = None,
        x_fp32: torch.Tensor | None = None,
        projection_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states, dtype = x, x.dtype
        if hasattr(self.impl, "pre_native") and self.impl._native_available():
            return self.impl.pre_native(
                hidden_states, hc_fn, hc_scale, hc_base, dtype,
                norm_weight=norm_weight, norm_eps=norm_eps)
        if x_fp32 is not None:
            assert x_fp32.shape == x.shape and x_fp32.dtype == torch.float32
        x = x.flatten(2)
        # Long prefills amortize the additional compiled-call overhead.  NPU
        # stays eager: Dynamo probes CUDA capability while tracing (same trap
        # as rms_scale below, and the lazy first compile may fire inside
        # NPUGraph capture warmup).  ``x.is_cuda`` cannot gate this --
        # transfer_to_npu aliases it to ``is_npu``, so check the device type.
        if x_fp32 is not None:
            x = x_fp32.flatten(2)
        else:
            x = (_cast_fp32(x) if x.device.type == 'cuda' and x.is_contiguous()
                 and x.size(0) * x.size(1) >= 8192 else x.float())
        single_token = x.size(0) == 1 and x.size(1) == 1
        mixes = None
        if projection_weight is not None and (not single_token or self.avoid_gemv):
            mixes = self.impl.try_project_mixes(x, projection_weight)
        if mixes is None:
            if self.avoid_gemv and single_token:
                # Single-token decode otherwise selects GEMV, whose reduction
                # order can differ from multi-token speculative verification.
                mixes = F.linear(F.pad(x, (0, 0, 0, 1)), hc_fn)[:, :1].contiguous()
            else:
                mixes = F.linear(x, hc_fn)
        scaled_mixes = self.impl.try_scale_mixes(mixes, x, norm_eps)
        if scaled_mixes is not None:
            mixes = scaled_mixes
        elif x.device.type == 'cuda':
            mixes = rms_scale(mixes, x, eps=norm_eps)
        else:
            # Keep this reduction in eager Torch on Ascend: the shared helper
            # is torch.compile-decorated and probes CUDA while tracing.
            mixes = self._mix_scale(mixes, x, norm_eps)
        return self.impl.pre(hidden_states, mixes, hc_scale, hc_base, dtype,
                             norm_weight=norm_weight, norm_eps=norm_eps)

    def _mix_scale(self, mixes: torch.Tensor, x: torch.Tensor,
                   norm_eps: float) -> torch.Tensor:
        """mixes * rsqrt(mean(x^2) + eps): the mHC weightless RMS scale.

        On NPU one fused npu_rms_norm call replaces the eager
        square/mean/rsqrt chain (2 per layer in decode); only the per-row
        rstd (second return) is consumed.  The kernel accumulates in fp32,
        so it matches the eager fp32 formula on the same input values.
        """
        if x.device.type == 'npu':
            try:
                import torch_npu
                ones = self.__dict__.get('_mix_scale_ones')
                if (ones is None or ones.device != x.device
                        or ones.shape[-1] != x.shape[-1]
                        or ones.dtype != x.dtype):
                    ones = torch.ones(x.shape[-1], dtype=x.dtype,
                                      device=x.device)
                    self.__dict__['_mix_scale_ones'] = ones
                rstd = torch_npu.npu_rms_norm(x, ones, epsilon=norm_eps)[1]
                return mixes * rstd
            except Exception:
                pass
        return mixes.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + norm_eps)

    def pre_reduce(self, x: torch.Tensor, pre: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        return self.impl.pre_reduce(x, pre, out_dtype)

    def post_expand(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                    comb: torch.Tensor) -> torch.Tensor:
        return self.impl.post_expand(x, residual, post, comb)

    def post_expand_with_fp32(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                              comb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.impl.post_expand_with_fp32(x, residual, post, comb)
