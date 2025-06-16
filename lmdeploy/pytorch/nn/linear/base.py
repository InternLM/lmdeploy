# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn

from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

from .utils import update_tp_args


def _gather_input(x: torch.Tensor, tp_sizes: List[int]):
    """Gather input."""
    shape0 = x.shape[:-2]
    shape1 = x.shape[-1:]
    shapes = [shape0 + (size, ) + shape1 for size in tp_sizes]
    new_x = [x.new_empty(shape) for shape in shapes]
    dist.all_gather(new_x, x)
    x = torch.cat(new_x, dim=-2)
    return x


class LinearBase(nn.Module):
    """Base class for linear layers."""

    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        colwise: bool = True,
        is_tp: bool = False,
        all_reduce: bool = True,
        tp_align_size: int = 1,
        dp_gather: bool = False,
        dp_scatter: bool = False,
    ):
        super().__init__()
        is_tp, all_reduce = update_tp_args(is_tp, all_reduce, colwise)
        self.colwise = colwise
        self.is_tp = is_tp
        self.all_reduce = all_reduce
        self.tp_align_size = tp_align_size
        self.dp_gather = dp_gather
        self.dp_scatter = dp_scatter
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16
        self.device = device
        self.dtype = dtype

        self.lora_adapters = nn.ModuleDict()

    def _forward_default(self, x, all_reduce: bool, tp_sizes: List[int]):
        """Default forward implement."""
        raise NotImplementedError('This method should be implemented in subclasses.')

    def _forward_lora(self, x, tp_sizes: List[int]):
        """Forward with LoRA."""
        from .lora import forward_adapters
        out = self._forward_default(x, False, tp_sizes)
        out = forward_adapters(x,
                               out,
                               self.lora_adapters.values(),
                               all_reduce=self.all_reduce,
                               dp_scatter=self.dp_scatter,
                               tp_sizes=tp_sizes)
        return out

    def forward(self, x):
        """Forward of linear layer."""
        tp_sizes = None
        if self.dp_gather or self.dp_scatter:
            step_ctx = get_step_ctx_manager().current_context()
            dp_meta = step_ctx.dp_meta
            tp_sizes = dp_meta.tp_sizes

        if self.dp_gather:
            x = _gather_input(x, tp_sizes)

        if len(self.lora_adapters) == 0:
            return self._forward_default(x, self.all_reduce, tp_sizes)
        else:
            return self._forward_lora(x, tp_sizes)
