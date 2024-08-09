# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch import nn

from ..backends import LayerType, get_backend


class SoftmaxTopK(nn.Module):
    """softmax topk."""

    def __init__(self, top_k: int, dim: int = -1):
        super().__init__()
        self.top_k = top_k
        impl_builder = get_backend().get_layer_impl_builder(
            LayerType.SoftmaxTopK)
        self.impl = impl_builder.build(top_k, dim)

    def forward(self, x: torch.Tensor):
        """forward."""
        return self.impl.forward(x)


def build_moe_from_mlp(
    gates: List[nn.Linear],
    ups: List[nn.Linear],
    downs: List[nn.Linear],
    top_k: int,
    renormalize: bool = False,
):
    """build moe from mlp."""
    impl_builder = get_backend().get_layer_impl_builder(LayerType.FusedMoE)
    return impl_builder.build_from_mlp(gates,
                                       ups,
                                       downs,
                                       top_k=top_k,
                                       renormalize=renormalize)
