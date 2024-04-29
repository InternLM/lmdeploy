# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from torch import distributed as dist
from torch import nn
from torch.distributed._tensor import DeviceMesh


def _div_up(a, b):
    """div up."""
    return (a + b - 1) // b


class PatchedQwen2MoeSparseMoeBlock(nn.Module):

    @classmethod
    def _get_expert_range(cls, num_experts: int):
        rank = 0
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        num_experts_per_rank = _div_up(num_experts, world_size)
        first_experts_id = rank * num_experts_per_rank
        last_experts_id = min(num_experts,
                              first_experts_id + num_experts_per_rank)
        return first_experts_id, last_experts_id

    def _distribute_partition_fn(self, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name == 'experts':
            world_size = dist.get_world_size()
            num_experts: int = len(self.experts)
            assert num_experts > world_size, (
                f'world_size: {world_size} should not greater than '
                f'num_experts: {num_experts}')
            first_experts_id, last_experts_id = self._get_expert_range(
                num_experts)
            for i in range(num_experts):
                if i >= first_experts_id and i < last_experts_id:
                    continue
                mod[i] = nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """moe forward."""
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
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device)

        num_experts = self.num_experts
        first_experts_id, last_experts_id = self._get_expert_range(num_experts)

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

        if dist.is_initialized():
            dist.all_reduce(final_hidden_states)

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(
            self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.unflatten(
            0, (-1, sequence_length))

        return final_hidden_states, router_logits


class PatchedQwen2MoeModel(nn.Module):

    def _continuous_batching_forward(
            self,
            input_ids: torch.LongTensor = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None):
        """Rewrite implementation of Qwen2MoeModel.forward."""
        from transformers.modeling_outputs import MoeModelOutputWithPast

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):

            past_key_value = past_key_values[idx]
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(last_hidden_state=hidden_states,
                                      past_key_values=past_key_values,
                                      hidden_states=None,
                                      attentions=None,
                                      router_logits=None)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """Rewrite of Qwen2MoeModel.forward."""
        return self._continuous_batching_forward(
            input_ids,
            position_ids,
            past_key_values,
            inputs_embeds,
        )
