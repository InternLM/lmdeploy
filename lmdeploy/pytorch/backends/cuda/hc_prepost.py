# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.backends.hc_prepost import BaseHcPrePost, BaseHcPrePostBuilder
from lmdeploy.pytorch.kernels.cuda.dsv4.hc_prepost import hc_post_expand, hc_pre_reduce


class TritonHcPrePostImpl(BaseHcPrePost):

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from lmdeploy.pytorch.kernels.cuda.dsv4.hc_split_sinkhorn import hc_split_sinkhorn
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.sinkhorn_iters, self.eps)
        y = self.pre_reduce(x, pre, out_dtype)
        return y, post, comb

    def pre_reduce(self, x: torch.Tensor, pre: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        return hc_pre_reduce(x, pre, self.hc_mult, out_dtype=out_dtype)

    def post_expand(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                    comb: torch.Tensor) -> torch.Tensor:
        return hc_post_expand(x, residual, post, comb, self.hc_mult)


class TritonHcPrePostBuilder(BaseHcPrePostBuilder):

    @staticmethod
    def build(hc_mult: int, sinkhorn_iters: int, eps: float) -> BaseHcPrePost:
        return TritonHcPrePostImpl(hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps)
