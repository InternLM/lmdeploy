# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend


class HcSplitSinkhorn(nn.Module):
    """DeepSeek V4 HC split sinkhorn wrapper."""

    def __init__(self, hc_mult: int, sinkhorn_iters: int, eps: float):
        super().__init__()
        backend = get_backend()
        impl_builder = backend.get_layer_impl_builder(OpType.HcSplitSinkhorn)
        self.impl = impl_builder.build(
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps)

    def forward(
        self,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.impl.forward(mixes, hc_scale, hc_base)
