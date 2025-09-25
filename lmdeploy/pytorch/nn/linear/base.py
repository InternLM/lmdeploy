# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn

from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.distributed import (gather_by_tp_sizes, get_dist_group, get_dist_manager, get_tp_world_rank,
                                          reduce_scatter_by_tp_sizes)
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

from .utils import update_tp_args


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
        layer_type: str = 'attn',
    ):
        super().__init__()
        self.init_tp_args(is_tp, all_reduce, colwise, layer_type)
        self.colwise = colwise
        self.tp_align_size = tp_align_size
        self.dp_gather = dp_gather
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16
        self.device = device
        self.dtype = dtype
        self.layer_type = layer_type

        self.lora_adapters = nn.ModuleDict()

    def init_tp_args(self, is_tp: bool, all_reduce: bool, colwise: bool, layer_type: str):
        if getattr(self, '_tp_args_initialized', False):
            return
        is_tp, all_reduce = update_tp_args(is_tp, all_reduce, colwise, layer_type=layer_type)
        self.is_tp = is_tp
        self.all_reduce = all_reduce
        if is_tp:
            dist_cfg = get_dist_manager().current_config()
            _, rank = get_tp_world_rank(layer_type)
            tp, tp_mode = dist_cfg.get_tp_by_layer(layer_type)
            self.tp_rank = rank
            self.tp = tp
            self.tp_mode = tp_mode
            dist_group = get_dist_group(layer_type=layer_type)
            self.tp_group = dist_group.gpu_group
            self.gather_group = dist_group.gpu_gather_group
        else:
            self.tp_rank = 0
            self.tp = 1
            self.tp_mode = TPMode.DEFAULT
            self.tp_group = None
            self.gather_group = None

        self._tp_args_initialized = True

    def get_tp_world_rank(self):
        """Get tp world rank."""
        assert hasattr(self, 'tp') and hasattr(self, 'tp_rank'), 'Please run init_tp_args first.'
        return self.tp, self.tp_rank

    def update_weights(self):
        """Update weights."""
        raise NotImplementedError('This method should be implemented in subclasses.')

    def _forward_default(self, x, all_reduce: bool, tp_sizes: List[int]):
        """Default forward implement."""
        raise NotImplementedError('This method should be implemented in subclasses.')

    def _forward_lora(self, x, tp_sizes: List[int]):
        """Forward with LoRA."""
        out = self._forward_default(x, False, tp_sizes)

        for lora_adapter in self.lora_adapters.values():
            out = lora_adapter(x, out)
        if self.all_reduce:
            if self.tp_mode == TPMode.DP_TP:
                out = reduce_scatter_by_tp_sizes(out, self.tp_rank, tp_sizes, group=self.tp_group)
            else:
                dist.all_reduce(out, group=self.tp_group)
        return out

    def forward(self, x):
        """Forward of linear layer."""
        tp_sizes = None
        if self.dp_gather or (self.all_reduce and self.tp_mode == TPMode.DP_TP):
            step_ctx = get_step_ctx_manager().current_context()
            dp_meta = step_ctx.dp_meta
            tp_sizes = dp_meta.tp_sizes

        if self.dp_gather:
            x = gather_by_tp_sizes(x, tp_sizes, group=self.gather_group)

        if len(self.lora_adapters) == 0:
            return self._forward_default(x, self.all_reduce, tp_sizes)
        else:
            return self._forward_lora(x, tp_sizes)
