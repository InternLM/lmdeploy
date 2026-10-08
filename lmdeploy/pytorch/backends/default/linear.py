# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..linear import LinearImpl


class DefaultLinearImpl(LinearImpl):
    """Linear implementation api."""

    supports_batched = True

    def forward_batched(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Batch equal-sized local projections without changing
        accumulation."""
        return torch.bmm(x, weight.transpose(1, 2))

    def forward(self,
                x,
                weight: torch.Tensor,
                bias: torch.Tensor | None = None,
                all_reduce: bool = False,
                group: dist.ProcessGroup = None,
                rank: int = 0,
                scatter_size: list[int] = None):
        """forward."""
        out = F.linear(x, weight, bias)
        if all_reduce:
            if scatter_size is not None:
                from lmdeploy.pytorch.distributed import reduce_scatter_by_tp_sizes
                out = reduce_scatter_by_tp_sizes(out, rank, scatter_size, group=group)
            else:
                dist.all_reduce(out, group=group)
        return out
