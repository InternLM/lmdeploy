# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.distributed import get_dist_manager, get_tp_world_rank
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager


class MoeType(Enum):
    """Batch ecex type."""
    Default = auto()
    DSAsyncDecode = auto()
    DSAsyncPrefill = auto()


class SoftmaxTopK(nn.Module):
    """Softmax topk."""

    def __init__(self, top_k: int, dim: int = -1, n_groups: int = -1):
        super().__init__()
        self.top_k = top_k
        impl_builder = get_backend().get_layer_impl_builder(OpType.SoftmaxTopK)
        self.impl = impl_builder.build(top_k, dim, n_groups=n_groups)

    def forward(self, x: torch.Tensor):
        """forward."""
        return self.impl.forward(x)


def update_dims(hidden_dim: int, ffn_dim: int):
    """Update dims."""
    world_size, _ = get_tp_world_rank('moe')
    assert ffn_dim % world_size == 0
    ffn_dim = ffn_dim // world_size
    return hidden_dim, ffn_dim


def split_size(size: int, world_size: int, align: int):
    size = size // align
    base = size // world_size
    remain = size % world_size
    split_size = [base + 1] * remain + [base] * (world_size - remain)
    split_size = [s * align for s in split_size]
    return split_size


def moe_gather_inputs(hidden_states, topk_weights, topk_ids, group: Optional[dist.ProcessGroup] = None):
    dist_config = get_dist_manager().current_config()
    tp = dist_config.moe_tp
    if tp == 1:
        return hidden_states, topk_weights, topk_ids

    tp_mode = dist_config.moe_tp_mode
    if tp_mode == TPMode.DEFAULT:
        return hidden_states, topk_weights, topk_ids
    elif tp_mode == TPMode.DP_TP:
        step_ctx = get_step_ctx_manager().current_context()
        dp_meta = step_ctx.dp_meta
        tp_sizes = dp_meta.moe_tp_sizes
        hidden_states = dist.gather_by_tp_sizes(hidden_states, tp_sizes, group=group)
        topk_weights = dist.gather_by_tp_sizes(topk_weights, tp_sizes, group=group)
        topk_ids = dist.gather_by_tp_sizes(topk_ids, tp_sizes, group=group)
    else:
        raise RuntimeError('Not supported.')

    return hidden_states, topk_weights, topk_ids


def moe_reduce(ret, rank: int, tp_mode: TPMode, group: Optional[dist.ProcessGroup] = None):
    dist_config = get_dist_manager().current_config()
    if dist_config.moe_tp == 1:
        return ret

    if tp_mode == TPMode.DEFAULT:
        dist.all_reduce(ret, group=group)
        return ret
    elif tp_mode == TPMode.DP_TP:
        step_ctx = get_step_ctx_manager().current_context()
        dp_meta = step_ctx.dp_meta
        tp_size = dp_meta.moe_tp_sizes
        ret = dist.reduce_scatter_by_tp_sizes(ret, rank, tp_size, group=group)
        return ret
    else:
        raise RuntimeError('Not supported.')


class MoEForwardDPTP:

    def __init__(self, gemm_func: Callable, max_tokens_per_round: int = 8192):
        """MoE forward dp tp."""
        self.gemm_func = gemm_func
        self.dist_ctx = get_dist_manager().current_context()
        self.dist_config = self.dist_ctx.dist_config
        self.tp = self.dist_config.moe_tp
        self.attn_tp = self.dist_config.attn_tp

        tp_group = self.dist_ctx.moe_tp_group
        self.rank = tp_group.rank
        self.gather_rank = self.rank // self.attn_tp
        self.gather_group = tp_group.gpu_gather_group
        self.tp_group = tp_group.gpu_group
        self.max_tokens_per_round = max_tokens_per_round * self.attn_tp // self.tp

    def all_gather(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                   tp_sizes: List[int]):
        """All gather."""
        hidden_states, h0 = dist.gather_by_tp_sizes(hidden_states, tp_sizes, group=self.gather_group, async_op=True)
        topk_weights, h1 = dist.gather_by_tp_sizes(topk_weights, tp_sizes, group=self.gather_group, async_op=True)
        topk_ids, h2 = dist.gather_by_tp_sizes(topk_ids, tp_sizes, group=self.gather_group, async_op=True)
        return hidden_states, topk_weights, topk_ids, (h0, h1, h2)

    def reduce_scatter(self, hidden_states: torch.Tensor, out_states: torch.Tensor, tp_sizes: List[int]):
        """Reduce scatter."""
        hidden_states_list = list(hidden_states.split(tp_sizes, -2))
        cur_out_states = hidden_states_list[self.gather_rank]
        out_states.copy_(cur_out_states)
        hidden_states_list = [item for item in hidden_states_list for _ in range(self.attn_tp)]
        hidden_states_list[self.rank] = out_states
        handle = dist.reduce_scatter(out_states, hidden_states_list, group=self.tp_group, async_op=True)
        return out_states, handle

    def _gemm_and_reduce_scatter(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                                 output_states: torch.Tensor, tp_sizes: List[int], handles: List[dist.Work]):
        """Gemm and reduce scatter."""
        for handle in handles:
            handle.wait()
        cur_out = self.gemm_func(hidden_states, topk_weights, topk_ids)
        return self.reduce_scatter(cur_out, output_states, tp_sizes)

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        """forward."""

        def __slice_tensor(tensor: torch.Tensor, slice_size: int):
            """Slice tensor."""
            cur_tensor = tensor[:slice_size]
            tensor = tensor[slice_size:]
            return cur_tensor, tensor

        def __slice_and_gather():
            """Slice and gather."""
            nonlocal hidden_states, topk_weights, topk_ids, tp_sizes, output_states
            cur_tp_sizes = tp_sizes.minimum(max_tokens_per_round)
            tp_sizes -= cur_tp_sizes
            cur_tp_sizes = cur_tp_sizes.tolist()

            slice_size = cur_tp_sizes[self.gather_rank]
            cur_hidden_states, hidden_states = __slice_tensor(hidden_states, slice_size)
            cur_topk_weights, topk_weights = __slice_tensor(topk_weights, slice_size)
            cur_topk_ids, topk_ids = __slice_tensor(topk_ids, slice_size)
            cur_output, output_states = __slice_tensor(output_states, slice_size)

            # all gather
            cur_hidden_states, cur_topk_weights, cur_topk_ids, handles = self.all_gather(
                cur_hidden_states, cur_topk_weights, cur_topk_ids, cur_tp_sizes)
            return dict(hidden_states=cur_hidden_states,
                        topk_weights=cur_topk_weights,
                        topk_ids=cur_topk_ids,
                        output_states=cur_output,
                        handles=handles,
                        tp_sizes=cur_tp_sizes)

        step_ctx = get_step_ctx_manager().current_context()
        tp_sizes = step_ctx.dp_meta.moe_tp_sizes
        tp_sizes = torch.tensor(tp_sizes)
        max_tokens_per_round = tp_sizes.new_tensor(self.max_tokens_per_round)

        output_states = torch.empty_like(hidden_states)
        return_states = output_states

        # pre
        cur_inputs = __slice_and_gather()

        out_handles = []
        # main loop
        while tp_sizes.sum() > 0:
            next_inputs = __slice_and_gather()
            _, handle = self._gemm_and_reduce_scatter(**cur_inputs)
            out_handles.append(handle)
            cur_inputs = next_inputs

        # post
        _, handle = self._gemm_and_reduce_scatter(**cur_inputs)
        out_handles.append(handle)
        for handle in out_handles:
            handle.wait()
        return return_states


def _renormalize(topk_weights: torch.Tensor, renormalize: bool):
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()
    return topk_weights


@dataclass
class DispatchInputs:
    """Dispatch inputs."""
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_idx: torch.LongTensor
    moe_type: MoeType = MoeType.Default

    @classmethod
    def from_dict(cls, input: Dict):
        """From dict."""
        assert ['hidden_states', 'topk_weights', 'topk_idx'] in input
        moe_type = input.get('moe_type', MoeType.Default)
        return cls(
            hidden_states=input['hidden_states'],
            topk_weights=input['topk_weights'],
            topk_idx=input['topk_idx'],
            moe_type=moe_type,
        )

    def to_dict(self) -> Dict:
        """To dict."""
        return {
            'hidden_states': self.hidden_states,
            'topk_weights': self.topk_weights,
            'topk_idx': self.topk_idx,
            'moe_type': self.moe_type,
        }


class FusedMoEBase(nn.Module):
    """Fused MoE base."""

    def __init__(self, tp: int, tp_mode: TPMode, do_renormalize: bool):
        super().__init__()
        self.tp = tp
        self.tp_mode = tp_mode
        self.do_renormalize = do_renormalize

    def init_dist_args(self, all_reduce: bool):
        """Init tp args."""
        dist_ctx = get_dist_manager().current_context()
        dist_cfg = dist_ctx.dist_config
        _, tp_mode = dist_cfg.get_tp_by_layer('moe')
        tp, tp_rank = get_tp_world_rank('moe')
        all_reduce = all_reduce if tp > 1 else False

        self.ep = dist_cfg.ep
        self.tp = tp
        self.tp_rank = tp_rank
        self.tp_mode = tp_mode
        self.all_reduce = all_reduce
        self.tp_group = dist_ctx.moe_tp_group.gpu_group
        self.gather_group = dist_ctx.moe_tp_group.gpu_gather_group

        if self.tp > 1 and self.tp_mode == TPMode.DP_TP:

            def __gemm_func(hidden_states, topk_weights, topk_ids):
                return self.gemm(
                    dict(
                        hidden_states=hidden_states,
                        topk_weights=topk_weights,
                        topk_idx=topk_ids,
                        moe_type=MoeType.Default,
                    ))['hidden_states']

            self._forward_dptp = MoEForwardDPTP(__gemm_func)
        else:
            self._forward_dptp = None

    def before_dispatch(self, state: DispatchInputs):
        """Before dispatch."""
        raise NotImplementedError

    def dispatch(self, state: Dict):
        """dispatch."""
        raise NotImplementedError

    def gemm(self, state: Dict):
        """gemm."""
        raise NotImplementedError

    def combine(self, state: Dict):
        """combine."""
        raise NotImplementedError

    def wait(self, state: Dict):
        """wait."""
        raise NotImplementedError

    @property
    def forward_dptp(self) -> MoEForwardDPTP:
        """Forward dptp."""
        return self._forward_dptp

    def forward_default(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_idx: torch.LongTensor):
        """Default forward."""
        state = {
            'hidden_states': hidden_states,
            'topk_idx': topk_idx,
            'topk_weights': topk_weights,
            'moe_type': MoeType.Default,
        }
        recv_state = self.dispatch(state)
        gemm_state = self.gemm(recv_state)
        out_state = self.combine(gemm_state)
        return out_state['hidden_states']

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_idx: torch.LongTensor):
        """forward."""
        if self.tp > 1 and self.tp_mode == TPMode.DP_TP:
            return self.forward_dptp.forward(hidden_states, topk_weights, topk_idx)
        else:
            return self.forward_default(hidden_states, topk_weights, topk_idx)

    def renormalize(self, topk_weights):
        """renormalize."""
        return _renormalize(topk_weights, self.do_renormalize)
