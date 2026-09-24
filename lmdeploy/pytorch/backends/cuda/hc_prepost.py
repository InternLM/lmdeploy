# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.backends.hc_prepost import HCPrePostImpl
from lmdeploy.pytorch.kernels.cuda.dsv4.hc_prepost import hc_post_expand, hc_pre_reduce
from lmdeploy.pytorch.kernels.cuda.linear_bf16xfp32 import linear_bf16xfp32


class TritonHCPrePostImpl(HCPrePostImpl):

    def __init__(self, hc_mult: int, sinkhorn_iters: int, eps: float):
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

    def pre(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_eps: float,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Local imports: nn.norm imports the backend package, and the sinkhorn
        # kernel is a tilelang JIT built per configuration.
        from lmdeploy.pytorch.kernels.cuda.dsv4.hc_split_sinkhorn import hc_split_sinkhorn
        from lmdeploy.pytorch.nn.norm import rms_scale

        dim = x.size(-1)
        # `x` stays in its low precision dtype: bf16 -> fp32 is lossless, so the
        # RMS factor and the reduction below are unchanged.
        x2d = x.reshape(-1, self.hc_mult * dim)
        mixes = linear_bf16xfp32(x2d, weight)
        # `out_dtype` is required: `b` is bf16 now, so rms_scale would otherwise
        # return bf16 and drag pre/post/comb down with it.
        mixes = rms_scale(mixes, x2d, dim=-1, eps=norm_eps, out_dtype=torch.float32)
        pre, post, comb = hc_split_sinkhorn(
            mixes.view(*x.shape[:-2], weight.size(0)), hc_scale, hc_base, self.hc_mult, self.sinkhorn_iters, self.eps)
        return self.pre_reduce(x, pre, out_dtype), post, comb

    def pre_reduce(self, x: torch.Tensor, pre: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        return hc_pre_reduce(x, pre, self.hc_mult, out_dtype=out_dtype)

    def post_expand(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                    comb: torch.Tensor) -> torch.Tensor:
        return hc_post_expand(x, residual, post, comb, self.hc_mult)
