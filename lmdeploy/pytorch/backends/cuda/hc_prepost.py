# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.backends.hc_prepost import HCPrePostImpl
from lmdeploy.pytorch.kernels.cuda.dsv4.hc_prepost import hc_post_expand, hc_pre_reduce, hc_pre_reduce_norm


class TritonHCPrePostImpl(HCPrePostImpl):

    def __init__(self, hc_mult: int, sinkhorn_iters: int, eps: float):
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

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
