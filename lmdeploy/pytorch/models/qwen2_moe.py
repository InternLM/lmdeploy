# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch import nn

from lmdeploy.pytorch.kernels.fused_moe import fused_moe
from lmdeploy.pytorch.kernels.moe_gating_topk_softmax import \
    moe_gating_topk_softmax


class PatchedQwen2MoeSparseMoeBlock(nn.Module):

    def _update_model_fn(self):
        """update model."""
        num_experts = len(self.experts)

        def __get_meta():
            exp = self.experts[0]
            ffn_dim = exp.gate_proj.weight.size(0)
            hidden_dim = exp.down_proj.weight.size(0)
            dtype = exp.gate_proj.weight.dtype
            device = exp.gate_proj.weight.device
            return ffn_dim, hidden_dim, dtype, device

        def __copy_assign_param(param, weight):
            """copy assign."""
            weight.copy_(param.data)
            param.data = weight

        ffn_dim, hidden_dim, dtype, device = __get_meta()

        gate_up_weights = torch.empty(num_experts,
                                      ffn_dim * 2,
                                      hidden_dim,
                                      device=device,
                                      dtype=dtype)
        down_weights = torch.empty(num_experts,
                                   hidden_dim,
                                   ffn_dim,
                                   device=device,
                                   dtype=dtype)

        for exp_id, exp in enumerate(self.experts):
            __copy_assign_param(exp.gate_proj.weight,
                                gate_up_weights[exp_id, :ffn_dim])
            __copy_assign_param(exp.up_proj.weight, gate_up_weights[exp_id,
                                                                    ffn_dim:])
            __copy_assign_param(exp.down_proj.weight, down_weights[exp_id])

        torch.cuda.empty_cache()

        self.register_buffer('gate_up_weights', gate_up_weights)
        self.register_buffer('down_weights', down_weights)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """moe forward."""

        _, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1,
                                                       sorted=False)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        out_states = fused_moe(hidden_states,
                               self.gate_up_weights,
                               self.down_weights,
                               routing_weights,
                               selected_experts,
                               topk=self.top_k,
                               renormalize=False)

        # all reduce of shared_expert is not necessary
        shared_expert_output = self.shared_expert.forward(hidden_states)
        shared_expert_output = F.sigmoid(
            self.shared_expert_gate(hidden_states)) * shared_expert_output

        out_states = out_states + shared_expert_output
        out_states = out_states.unflatten(0, (-1, sequence_length))

        if dist.is_initialized():
            dist.all_reduce(out_states)

        return out_states, router_logits


class PatchedQwen2MoeSparseMoeBlockAscend(nn.Module):

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """"""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights, selected_experts = moe_gating_topk_softmax(
            router_logits, self.top_k)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        out_states = torch.zeros((batch_size * sequence_length, hidden_dim),
                                 dtype=hidden_states.dtype,
                                 device=hidden_states.device)

        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(
                current_state) * routing_weights[top_x, idx, None]

            out_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(
            self.shared_expert_gate(hidden_states)) * shared_expert_output

        out_states = out_states + shared_expert_output
        out_states = out_states.unflatten(0, (-1, sequence_length))

        return out_states, router_logits


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
