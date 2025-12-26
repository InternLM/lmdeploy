# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional

import torch
import torch.distributed as dist
from torch import nn

from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.distributed import (gather_by_tp_sizes, get_dist_group, get_dist_manager, get_tp_world_rank,
                                          reduce_scatter_by_tp_sizes)
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

from .utils import update_tp_args


class LinearForwardDPTP:

    def __init__(self, gemm_func: Callable, max_tokens_per_round: int = 8192):
        """Linear forward dp tp."""
        self.gemm_func = gemm_func
        self.dist_ctx = get_dist_manager().current_context()
        self.dist_config = self.dist_ctx.dist_config
        self.tp = self.dist_config.mlp_tp
        self.attn_tp = self.dist_config.attn_tp

        tp_group = self.dist_ctx.mlp_tp_group
        self.rank = tp_group.rank
        self.gather_rank = self.rank // self.attn_tp
        self.gather_group = tp_group.gpu_gather_group
        self.tp_group = tp_group.gpu_group
        self.max_tokens_per_round = max_tokens_per_round * self.attn_tp // self.tp // 2

    def all_gather(self, hidden_states: torch.Tensor, tp_sizes: List[int]):
        """All gather."""
        hidden_states, handle = dist.gather_by_tp_sizes(hidden_states, tp_sizes, group=self.gather_group, async_op=True)
        return hidden_states, handle

    def reduce_scatter(self, hidden_states: torch.Tensor, out_states: torch.Tensor, tp_sizes: List[int]):
        """Reduce scatter."""
        hidden_states_list = list(hidden_states.split(tp_sizes, -2))
        cur_out_states = hidden_states_list[self.gather_rank]
        out_states.copy_(cur_out_states)
        hidden_states_list = [item for item in hidden_states_list for _ in range(self.attn_tp)]
        hidden_states_list[self.rank] = out_states
        handle = dist.reduce_scatter(out_states, hidden_states_list, group=self.tp_group, async_op=True)
        return out_states, handle

    def _gemm_and_reduce_scatter(self, hidden_states: torch.Tensor, output_states: torch.Tensor, tp_sizes: List[int],
                                 handle: dist.Work):
        """Gemm and reduce scatter."""
        handle.wait()
        cur_out = self.gemm_func(hidden_states)
        return self.reduce_scatter(cur_out, output_states, tp_sizes)

    def forward(self, hidden_states: torch.Tensor):
        """forward."""

        def __slice_tensor(tensor: torch.Tensor, slice_size: int):
            """Slice tensor."""
            cur_tensor = tensor[:slice_size]
            tensor = tensor[slice_size:]
            return cur_tensor, tensor

        def __slice_and_gather():
            """Slice and gather."""
            nonlocal hidden_states, tp_sizes, output_states
            cur_tp_sizes = tp_sizes.minimum(max_tokens_per_round)
            tp_sizes -= cur_tp_sizes
            cur_tp_sizes = cur_tp_sizes.tolist()

            slice_size = cur_tp_sizes[self.gather_rank]
            cur_hidden_states, hidden_states = __slice_tensor(hidden_states, slice_size)
            cur_output, output_states = __slice_tensor(output_states, slice_size)

            # all gather
            cur_hidden_states, handle = self.all_gather(cur_hidden_states, cur_tp_sizes)
            return dict(hidden_states=cur_hidden_states, output_states=cur_output, handle=handle, tp_sizes=cur_tp_sizes)

        step_ctx = get_step_ctx_manager().current_context()
        tp_sizes = step_ctx.dp_meta.moe_tp_sizes
        tp_sizes = torch.tensor(tp_sizes)
        max_tokens_per_round = tp_sizes.new_tensor(self.max_tokens_per_round)

        output_states = torch.empty_like(hidden_states)
        return_states = output_states

        # pre
        cur_inputs = __slice_and_gather()
        handles = []

        # main loop
        while tp_sizes.sum() > 0:
            next_inputs = __slice_and_gather()
            _, handle = self._gemm_and_reduce_scatter(**cur_inputs)
            handles.append(handle)
            cur_inputs = next_inputs

        # post
        _, handle = self._gemm_and_reduce_scatter(**cur_inputs)
        handles.append(handle)
        for handle in handles:
            handle.wait()
        return return_states


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

        if self.tp > 1 and self.tp_mode == TPMode.DP_TP:

            def _gemm_func(self, x):
                out = self._forward_default(x, False, None)

                for lora_adapter in self.lora_adapters.values():
                    out = lora_adapter(x, out)
                return out

            self.linear_dptp_forward = LinearForwardDPTP(_gemm_func)

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

    def _forward_lora(self, x, tp_sizes: List[int] = None):
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

    def _forward_dp_tp(self, x):
        """Forward dp_tp."""
        if self.dp_gather and self.all_reduce:
            return self.linear_dptp_forward.forward(x)

        step_ctx = get_step_ctx_manager().current_context()
        dp_meta = step_ctx.dp_meta
        tp_sizes = dp_meta.tp_sizes

        if self.dp_gather:
            x = gather_by_tp_sizes(x, tp_sizes, group=self.gather_group)

        if len(self.lora_adapters) == 0:
            return self._forward_default(x, self.all_reduce, tp_sizes)
        else:
            return self._forward_lora(x, tp_sizes)

    def forward(self, x):
        """Forward of linear layer."""
        if self.tp > 1 and self.tp_mode == TPMode.DP_TP:
            return self._forward_dp_tp(x)

        if len(self.lora_adapters) == 0:
            return self._forward_default(x, self.all_reduce, None)
        else:
            return self._forward_lora(x)
