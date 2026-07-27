# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend


class HcPrePost(nn.Module):
    """DeepSeek-V4 hyper-connection pre/post reduction wrapper."""

    def __init__(self, hc_mult: int, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        backend = get_backend()
        impl_builder = backend.get_layer_impl_builder(OpType.HcPrePost)
        self.impl = impl_builder.build(hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps)

    def pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from lmdeploy.pytorch.nn.norm import rms_scale
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        mixes = rms_scale(F.linear(x, hc_fn), x, eps=norm_eps)
        return self.impl.pre(x.view(shape), mixes, hc_scale, hc_base, dtype)

    def pre_reduce(self, x: torch.Tensor, pre: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        return self.impl.pre_reduce(x, pre, out_dtype)

    def post_expand(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                    comb: torch.Tensor) -> torch.Tensor:
        return self.impl.post_expand(x, residual, post, comb)
