# Copyright (c) 2024, DeepLink. All rights reserved.
import functools
import weakref
from collections.abc import Callable

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

hidden_states_gather_buffer = None
topk_weights_gather_buffer = None
topk_ids_gather_buffer = None


@functools.lru_cache
def _eager_mode():
    from lmdeploy.pytorch.backends.ascend import AscendOpsBackend

    return not AscendOpsBackend.enable_graph


@functools.lru_cache
def _get_max_batch_size():
    from lmdeploy.pytorch.backends.ascend import AscendOpsBackend

    return AscendOpsBackend.max_batches


def _clear_moe_comm_buffers():
    """Clear module-level MoE comm buffers."""
    global hidden_states_gather_buffer
    global topk_weights_gather_buffer
    global topk_ids_gather_buffer
    hidden_states_gather_buffer = None
    topk_weights_gather_buffer = None
    topk_ids_gather_buffer = None


class AscendMoEForwardDPTP:
    def __init__(self, gemm_func: Callable, max_tokens_per_round: int = 3000):
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

        # self.max_tokens_per_round = max_tokens_per_round * self.attn_tp // self.tp // 2
        self.max_tokens_per_round = max_tokens_per_round
        self.use_comm_buffer = None

        # When instance is GC'ed, clear module-level global buffers
        self._finalizer = weakref.finalize(self, _clear_moe_comm_buffers)

    def _init_comm_buffer(
        self,
        step_ctx,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        self.max_batch_size = _get_max_batch_size()
        self.hidden_size = hidden_states.size(-1)
        self.topk = topk_ids.size(-1)
        self.dp_size = len(step_ctx.dp_meta.moe_tp_sizes)

        # MTP decode sends max_batch_size * (num_spec_tokens + 1) tokens per step.
        # Pre-allocate for the worst case to avoid runtime reallocation.
        num_spec_tokens = get_step_ctx_manager().build_ctx.num_spec_tokens
        max_tokens = self.max_batch_size * (num_spec_tokens + 1)

        global hidden_states_gather_buffer
        global topk_weights_gather_buffer
        global topk_ids_gather_buffer

        hidden_states_gather_buffer = hidden_states.new_empty(max_tokens * self.dp_size * self.hidden_size)
        topk_weights_gather_buffer = topk_weights.new_empty(max_tokens * self.dp_size * self.topk)
        topk_ids_gather_buffer = topk_ids.new_empty(max_tokens * self.dp_size * self.topk)

    def all_gather(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        tp_sizes: list[int],
    ):
        """All gather."""
        hidden_states = dist.gather_by_tp_sizes(hidden_states, tp_sizes, group=self.gather_group, async_op=False)
        topk_weights = dist.gather_by_tp_sizes(topk_weights, tp_sizes, group=self.gather_group, async_op=False)
        topk_ids = dist.gather_by_tp_sizes(topk_ids, tp_sizes, group=self.gather_group, async_op=False)
        return hidden_states, topk_weights, topk_ids

    def reduce_scatter(self, hidden_states: torch.Tensor, out_states: torch.Tensor, tp_sizes: list[int]):
        """Reduce scatter."""
        hidden_states_list = list(hidden_states.split(tp_sizes, -2))
        cur_out_states = hidden_states_list[self.gather_rank]
        out_states.copy_(cur_out_states)
        hidden_states_list = [item for item in hidden_states_list for _ in range(self.attn_tp)]
        hidden_states_list[self.rank] = out_states
        dist.reduce_scatter(out_states, hidden_states_list, group=self.tp_group, async_op=False)
        return out_states

    def _gemm_and_reduce_scatter(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        output_states: torch.Tensor,
        tp_sizes: list[int],
    ):
        """Gemm and reduce scatter."""
        cur_out = self.gemm_func(hidden_states, topk_weights, topk_ids)
        return self.reduce_scatter(cur_out, output_states, tp_sizes)

    def forward_decode(
        self,
        step_ctx,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        """forward."""
        tp_sizes = step_ctx.dp_meta.moe_tp_sizes

        if self.use_comm_buffer:
            num_tokens = hidden_states.size(0)
            needed_hs = num_tokens * self.dp_size * self.hidden_size
            needed_tw = num_tokens * self.dp_size * self.topk

            cur_hidden_states = hidden_states_gather_buffer[:needed_hs].view(
                num_tokens * self.dp_size, self.hidden_size
            )
            cur_topk_weights = topk_weights_gather_buffer[:needed_tw].view(num_tokens * self.dp_size, self.topk)
            cur_topk_ids = topk_ids_gather_buffer[:needed_tw].view(num_tokens * self.dp_size, self.topk)

            torch.distributed.all_gather_into_tensor(
                cur_hidden_states,
                hidden_states,
                group=self.gather_group,
                async_op=False,
            )
            torch.distributed.all_gather_into_tensor(
                cur_topk_weights, topk_weights, group=self.gather_group, async_op=False
            )
            torch.distributed.all_gather_into_tensor(cur_topk_ids, topk_ids, group=self.gather_group, async_op=False)
        else:
            cur_hidden_states, cur_topk_weights, cur_topk_ids = self.all_gather(
                hidden_states, topk_weights, topk_ids, tp_sizes
            )

        # MoE gemm
        cur_out = self.gemm_func(cur_hidden_states, cur_topk_weights, cur_topk_ids)
        output_states = dist.reduce_scatter_by_tp_sizes(cur_out, self.rank, tp_sizes, group=self.tp_group)
        return output_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        """forward."""
        step_ctx = get_step_ctx_manager().current_context()
        if step_ctx.is_decoding:
            return self.forward_decode(step_ctx, hidden_states, topk_weights, topk_ids)

        # lazy init comm buffer
        if self.use_comm_buffer is None:
            self.eager_mode = _eager_mode()
            self.use_comm_buffer = not self.eager_mode

            if self.use_comm_buffer:
                self._init_comm_buffer(step_ctx, hidden_states, topk_weights, topk_ids)

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

            total_tokens = sum(cur_tp_sizes)
            sh = (
                *cur_hidden_states.shape[:-2],
                total_tokens,
                *cur_hidden_states.shape[-1:],
            )
            sw = (
                *cur_topk_weights.shape[:-2],
                total_tokens,
                *cur_topk_weights.shape[-1:],
            )
            si = (*cur_topk_ids.shape[:-2], total_tokens, *cur_topk_ids.shape[-1:])

            b_hidden = cur_hidden_states.new_empty(sh)
            b_weights = cur_topk_weights.new_empty(sw)
            b_ids = cur_topk_ids.new_empty(si)

            # map gather-group local index -> global rank for broadcast src
            from lmdeploy.pytorch.distributed import get_world_rank

            world_size, global_rank = get_world_rank()
            tp = self.tp
            attn_tp = self.attn_tp
            tp_group_id = global_rank // tp
            group_start = tp_group_id * tp
            gather_group_id = (global_rank - group_start) % attn_tp
            g_start = group_start + gather_group_id
            g_ranks = list(range(world_size))[g_start : (g_start + tp) : attn_tp]

            offset = 0
            for src_idx, sz in enumerate(cur_tp_sizes):
                h_slice = b_hidden.narrow(-2, offset, sz)
                w_slice = b_weights.narrow(-2, offset, sz)
                i_slice = b_ids.narrow(-2, offset, sz)

                if src_idx == self.gather_rank:
                    h_slice.copy_(cur_hidden_states)
                    w_slice.copy_(cur_topk_weights)
                    i_slice.copy_(cur_topk_ids)

                src_global = g_ranks[src_idx]

                dist.broadcast(h_slice, src_global, group=self.gather_group, async_op=False)
                dist.broadcast(w_slice, src_global, group=self.gather_group, async_op=False)
                dist.broadcast(i_slice, src_global, group=self.gather_group, async_op=False)

                offset += sz
            cur_hidden_states, cur_topk_weights, cur_topk_ids = (
                b_hidden,
                b_weights,
                b_ids,
            )

            return dict(
                hidden_states=cur_hidden_states,
                topk_weights=cur_topk_weights,
                topk_ids=cur_topk_ids,
                output_states=cur_output,
                tp_sizes=cur_tp_sizes,
            )

        tp_sizes = step_ctx.dp_meta.moe_tp_sizes
        tp_sizes = torch.tensor(tp_sizes)
        max_tokens_per_round = tp_sizes.new_tensor(self.max_tokens_per_round)

        output_states = torch.empty_like(hidden_states)
        return_states = output_states

        # pre
        cur_inputs = __slice_and_gather()

        # main loop
        while tp_sizes.sum() > 0:
            next_inputs = __slice_and_gather()
            self._gemm_and_reduce_scatter(**cur_inputs)
            cur_inputs = next_inputs

        # post
        self._gemm_and_reduce_scatter(**cur_inputs)
        return return_states
