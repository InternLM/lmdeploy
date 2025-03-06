# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn.functional as F

import lmdeploy.pytorch.distributed as dist

from ..linear import LinearBuilder, LinearImpl


def _reduce_scatter_input(out: torch.Tensor, rank: int, tp_sizes: List[int]):
    """reduce scatter."""
    outs = out.split(tp_sizes, -2)
    out = outs[rank]
    outs = list(outs)
    dist.reduce_scatter(out, outs)
    return out


class DefaultLinearImpl(LinearImpl):
    """Linear implementation api."""

    def forward(self,
                x,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                scatter: bool = False,
                rank: int = 0,
                scatter_size: List[int] = None):
        """forward."""
        out = F.linear(x, weight, bias)
        if all_reduce:
            if scatter:
                out = _reduce_scatter_input(out, rank, scatter_size)
            else:
                dist.all_reduce(out)
        return out


class DefaultLinearBuilder(LinearBuilder):
    """linear implementation builder."""

    @staticmethod
    def build(in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = None):
        """build."""
        return DefaultLinearImpl()
