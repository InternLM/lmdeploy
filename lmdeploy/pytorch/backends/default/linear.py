# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..linear import LinearBuilder, LinearImpl


class DefaultLinearImpl(LinearImpl):
    """Linear implementation api."""

    def forward(self,
                x,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                group: dist.ProcessGroup = None,
                rank: int = 0,
                scatter_size: List[int] = None):
        """forward."""
        out = F.linear(x, weight, bias)
        if all_reduce:
            if scatter_size is not None:
                from lmdeploy.pytorch.distributed import reduce_scatter_by_tp_sizes
                out = reduce_scatter_by_tp_sizes(out, rank, scatter_size, group=group)
            else:
                dist.all_reduce(out, group=group)
        return out


class DefaultLinearBuilder(LinearBuilder):
    """Linear implementation builder."""

    @staticmethod
    def build(in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = None):
        """build."""
        return DefaultLinearImpl()
