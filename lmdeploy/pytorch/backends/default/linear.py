# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn.functional as F

import lmdeploy.pytorch.distributed as dist

from ..linear import LinearBuilder, LinearImpl


def _reduce_scatter_input(out: torch.Tensor, rank: int, tp_sizes: List[int]):
    """Reduce scatter."""
    out = out.transpose(0, -2)
    if not out.is_contiguous():
        out = out.contiguous()
    outs = out.split(tp_sizes, 0)
    out = outs[rank]
    outs = list(outs)
    dist.reduce_scatter(out, outs)
    out = out.transpose(0, -2)
    return out


class DefaultLinearImpl(LinearImpl):
    """Linear implementation api."""

    def forward(self,
                x,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                rank: int = 0,
                scatter_size: List[int] = None):
        """forward."""
        out = F.linear(x, weight, bias)
        if all_reduce:
            if scatter_size is not None:
                out = _reduce_scatter_input(out, rank, scatter_size)
            else:
                dist.all_reduce(out)
        return out


class DefaultLinearBuilder(LinearBuilder):
    """Linear implementation builder."""

    @staticmethod
    def build(in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = None):
        """build."""
        return DefaultLinearImpl()
