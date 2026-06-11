# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.backends.hc_split_sinkhorn import BaseHcSplitSinkhorn, BaseHcSplitSinkhornBuilder


class TritonHcSplitSinkhornImpl(BaseHcSplitSinkhorn):

    def __init__(self, hc_mult: int, sinkhorn_iters: int, eps: float):
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

    def forward(
        self,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from lmdeploy.pytorch.kernels.cuda.dsv4.hc_split_sinkhorn import hc_split_sinkhorn
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult,
                                 self.sinkhorn_iters, self.eps)


class TritonHcSplitSinkhornBuilder(BaseHcSplitSinkhornBuilder):

    @staticmethod
    def build(hc_mult: int, sinkhorn_iters: int, eps: float) -> BaseHcSplitSinkhorn:
        return TritonHcSplitSinkhornImpl(hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps)
