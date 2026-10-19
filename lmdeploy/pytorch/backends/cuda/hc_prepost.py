# Copyright (c) OpenMMLab. All rights reserved.
import torch
from packaging.version import InvalidVersion, Version

from lmdeploy.pytorch.backends.hc_prepost import HCPrePostImpl
from lmdeploy.pytorch.kernels.cuda.dsv4.hc_prepost import hc_post_expand, hc_pre_reduce, hc_pre_reduce_norm


class TritonHCPrePostImpl(HCPrePostImpl):

    def __init__(self, hc_mult: int, sinkhorn_iters: int, eps: float):
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        import triton
        try:
            self._gluon = Version('3.6.0') <= Version(triton.__version__) < Version('3.8.0')
        except InvalidVersion:
            self._gluon = False

    def prepare_projection_weight(self, weight: torch.Tensor) -> torch.Tensor | None:
        if (self.hc_mult == 4 and weight.is_cuda
                and weight.shape == (24, 16384) and weight.dtype == torch.float32):
            return weight.t().contiguous()
        return None

    def try_project_mixes(self, x_fp32: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
        if (self.hc_mult == 4 and x_fp32.is_cuda and x_fp32.ndim == 3
                and x_fp32.shape[0] == 1 and 0 < x_fp32.shape[1] <= 32
                and x_fp32.shape[2] == 16384 and weight.shape == (16384, 24)
                and x_fp32.dtype == weight.dtype == torch.float32
                and x_fp32.is_contiguous() and weight.is_contiguous()
                and x_fp32.device == weight.device
                and not torch.backends.cuda.matmul.allow_tf32):
            from lmdeploy.pytorch.kernels.cuda.dsv4.hc_pre_fused import hc_project_mixes

            return hc_project_mixes(x_fp32, weight)
        return None

    def try_scale_mixes(self, mixes: torch.Tensor, x_fp32: torch.Tensor, eps: float) -> torch.Tensor | None:
        if (self._gluon and x_fp32.is_cuda and x_fp32.ndim == 3 and x_fp32.shape[0] == 1
                and 0 < x_fp32.shape[1] <= 64 and x_fp32.shape[2] == 16384
                and mixes.shape == (*x_fp32.shape[:-1], 24)
                and mixes.dtype == x_fp32.dtype == torch.float32
                and mixes.is_contiguous() and x_fp32.is_contiguous()):
            from lmdeploy.pytorch.kernels.cuda.dsv4.hc_pre_fused import hc_scale_mixes

            return hc_scale_mixes(mixes, x_fp32, eps)
        return None

    def pre(
        self,
        x: torch.Tensor,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        out_dtype: torch.dtype,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (self.hc_mult == 4 and x.is_cuda and x.shape[-2:] == (4, 4096)
                and 0 < x.numel() // (4 * 4096) <= 64
                and x.dtype == out_dtype == torch.bfloat16 and x.is_contiguous()
                and mixes.dtype == torch.float32 and mixes.is_contiguous()
                and norm_weight is not None and norm_weight.dtype == torch.bfloat16
                and norm_weight.shape == (4096,) and norm_weight.is_contiguous()):
            from lmdeploy.pytorch.kernels.cuda.dsv4.hc_pre_fused import hc_pre_fused

            return hc_pre_fused(x, mixes, hc_scale, hc_base, norm_weight,
                                self.sinkhorn_iters, self.eps, norm_eps)
        from lmdeploy.pytorch.kernels.cuda.dsv4.hc_split_sinkhorn import hc_split_sinkhorn
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.sinkhorn_iters, self.eps)
        if norm_weight is None:
            y = self.pre_reduce(x, pre, out_dtype)
        else:
            y = hc_pre_reduce_norm(x, pre, self.hc_mult, norm_weight, norm_eps, out_dtype)
        return y, post, comb

    def pre_reduce(self, x: torch.Tensor, pre: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        return hc_pre_reduce(x, pre, self.hc_mult, out_dtype=out_dtype)

    def post_expand(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                    comb: torch.Tensor) -> torch.Tensor:
        return hc_post_expand(x, residual, post, comb, self.hc_mult)

    def post_expand_with_fp32(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                              comb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fp32 = torch.empty((*x.shape[:-1], self.hc_mult, x.size(-1)), device=x.device, dtype=torch.float32)
        out = hc_post_expand(x, residual, post, comb, self.hc_mult, out_fp32=fp32)
        return out, fp32
