# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DeviceMesh
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..dist_utils import (colwise_parallelize_linear_fn,
                          rowwise_parallelize_linear_fn)
from ..kernels import apply_rotary_pos_emb, fill_kv_cache, paged_attention_fwd


class PatchedMixtralAttention(nn.Module):
    """Rewrite module of MixtralAttention."""

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['q_proj', 'k_proj', 'v_proj']:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['o_proj']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """default rewrite."""

        context = self.context.context
        kv_seq_length = context.kv_seq_length
        q_seq_length = context.q_seq_length
        q_start_loc = context.q_start_loc
        block_offsets = context.block_offsets
        max_q_seq_length = context.max_q_seq_length
        max_kv_seq_length = context.max_kv_seq_length
        position_ids_1d = context.position_ids_1d

        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_key_value_heads // world_size
        hidden_size = num_heads * self.head_dim

        def __qkv_proj(hidden_states):
            """qkv proj."""
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            if hasattr(self, 'rotary_emb'):
                if not hasattr(context, '_cos'):
                    cos, sin = self.rotary_emb(value_states,
                                               seq_len=max_kv_seq_length)
                    context._cos = cos
                    context._sin = sin
                else:
                    cos = context._cos
                    sin = context._sin
                query_states, key_states = apply_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    position_ids,
                    position_ids_1d,
                    q_embed=query_states,
                    k_embed=key_states)
            return query_states, key_states, value_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)

        query_states = query_states.view(-1, num_heads, self.head_dim)
        key_states = key_states.view(-1, num_kv_heads, self.head_dim)
        value_states = value_states.view(-1, num_kv_heads, self.head_dim)

        query_states, key_states, value_states = __rotary_emb_fn(
            query_states, key_states, value_states)
        # fill kv cache
        fill_kv_cache(
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            q_start_loc,
            q_seq_length,
            kv_seq_length=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            block_offsets=block_offsets,
        )
        # page attention
        attn_output = query_states
        window_size = self.config.sliding_window or -1
        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_q_seq_length,
            window_size=window_size,
        )

        attn_output = attn_output.reshape(*hidden_states.shape[:-1],
                                          hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of MistralAttention.forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids,
            past_key_value,
            output_attentions,
            attention_mask=attention_mask,
            world_size=world_size,
        )


def _div_up(a, b):
    """div up."""
    return (a + b - 1) // b


class PatchedMixtralSparseMoeBlock(nn.Module):

    def _distribute_partition_fn(self, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""

        if mod_name == 'experts':
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            num_experts: int = self.num_experts
            assert num_experts > world_size, (
                f'world_size: {world_size} should not greater than '
                f'num_experts: {num_experts}')
            num_experts_per_rank = _div_up(num_experts, world_size)

            first_experts_id = rank * num_experts_per_rank
            last_experts_id = min(num_experts,
                                  first_experts_id + num_experts_per_rank)
            for i in range(num_experts):
                if i >= first_experts_id and i < last_experts_id:
                    continue
                mod[i] = nn.Identity()

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def forward_naive(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """miaxtral forward."""
        import torch.nn.functional as F

        def __get_expert_index(selected_experts, num_experts, first_experts_id,
                               last_experts_id):
            """get expert index."""
            idxs, top_xs = [None] * num_experts, [None] * num_experts
            # Loop over all available experts in the model
            for expert_idx in range(first_experts_id, last_experts_id):
                pos = torch.nonzero(selected_experts == expert_idx)
                if pos.size(0) > 0:
                    top_x, idx = pos.t()
                    idxs[expert_idx] = idx
                    top_xs[expert_idx] = top_x
            return idxs, top_xs

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device)

        rank = 0
        world_size = 1
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        num_experts = self.num_experts
        num_experts_per_rank = _div_up(num_experts, world_size)
        first_experts_id = rank * num_experts_per_rank
        last_experts_id = min(num_experts,
                              first_experts_id + num_experts_per_rank)

        idxs, top_xs = __get_expert_index(selected_experts, num_experts,
                                          first_experts_id, last_experts_id)

        for expert_idx in range(first_experts_id, last_experts_id):
            idx, top_x = idxs[expert_idx], top_xs[expert_idx]
            if idx is None:
                continue
            expert_layer = self.experts[expert_idx]

            current_state = hidden_states.index_select(dim=0, index=top_x)
            current_hidden_states = expert_layer(
                current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        final_hidden_states = final_hidden_states.unflatten(
            0, (-1, sequence_length))
        return final_hidden_states, router_logits

    def forward_all(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """forward all."""
        import torch.nn.functional as F
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device)

        rank = 0
        world_size = 1
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        num_experts = self.num_experts
        num_experts_per_rank = _div_up(num_experts, world_size)
        first_experts_id = rank * num_experts_per_rank
        last_experts_id = min(num_experts,
                              first_experts_id + num_experts_per_rank)

        for expert_idx in range(first_experts_id, last_experts_id):
            expert_layer = self.experts[expert_idx]
            valid_mask = (selected_experts == expert_idx)
            weights = routing_weights * valid_mask
            weights = weights.sum(1, keepdim=True)
            current_hidden_states = expert_layer(hidden_states) * weights
            final_hidden_states.add_(current_hidden_states)
        final_hidden_states = final_hidden_states.unflatten(
            0, (-1, sequence_length))
        return final_hidden_states, router_logits

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """forward."""
        context = self.context.context

        # naive implementation is faster but require synchronize.
        if context.enable_naive_moe:
            return self.forward_naive(hidden_states)
        else:
            # synchronize has negative effect on engine pipeline
            # compute all tokens with all experts is better
            # than stream sync
            return self.forward_all(hidden_states)


class PatchedMixtralModel(nn.Module):

    def _continuous_batching_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of LlamaModel.forward."""

        from transformers.modeling_outputs import MoeModelOutputWithPast
        context = self.context.context
        output_attentions = False
        use_cache = True

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Attention mask is not necessary in continuous batching
        attention_mask = None

        hidden_states = inputs_embeds

        moe_threshold = len(self.layers) // 2
        for idx, decoder_layer in enumerate(self.layers):
            context.enable_naive_moe = (not context.is_decoding
                                        or idx < moe_threshold)
            past_key_value = past_key_values[idx]
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(last_hidden_state=hidden_states,
                                      past_key_values=past_key_values,
                                      hidden_states=None,
                                      attentions=None,
                                      router_logits='')

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite of LlamaModel.forward."""
        return self._continuous_batching_forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
        )
