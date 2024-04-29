# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeModelOutputWithPast

from ..dist_utils import rowwise_parallelize_linear_fn, try_to_local
from ..kernels import fill_kv_cache, fused_rotary_emb, paged_attention_fwd


def _colwise_split_parallelize_linear(mod: nn.Module, sections: List[int],
                                      device_mesh: DeviceMesh):
    """split and colwise parallelize."""
    for name, param in mod.named_parameters():
        splited_param = param.split(sections, dim=0)
        updated_param = []
        for p in splited_param:
            dist_tensor = distribute_tensor(p, device_mesh, [Shard(0)])
            dist_tensor = try_to_local(dist_tensor)
            updated_param.append(dist_tensor)
        param = torch.cat(updated_param)
        dist_param = torch.nn.Parameter(param)
        mod.register_parameter(name, dist_param)


class PatchedDbrxAttention(nn.Module):

    def _distribute_qkv_linear(self, mod: nn.Module, device_mesh: DeviceMesh):
        """distribute qkv linear."""
        sections = [
            self.num_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
        ]
        return _colwise_split_parallelize_linear(mod, sections, device_mesh)

    def _distribute_partition_fn(self, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['Wqkv']:
            self._distribute_qkv_linear(mod, device_mesh)
        elif mod_name in ['out_proj']:
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
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Implement of attention forward."""
        context = self.context.context
        q_start_loc = context.q_start_loc
        q_seq_length = context.q_seq_length
        kv_seq_length = context.kv_seq_length
        block_offsets = context.block_offsets
        max_q_seq_length = context.max_q_seq_length

        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_key_value_heads // world_size
        head_dim = self.head_dim

        def __qkv_proj(hidden_states):
            """qkv_proj."""
            qkv_states = self.Wqkv(hidden_states)
            if self.clip_qkv is not None:
                qkv_states = qkv_states.clamp(min=-self.clip_qkv,
                                              max=self.clip_qkv)

            query_states, key_states, value_states = qkv_states.split(
                [
                    num_heads * head_dim,
                    num_kv_heads * head_dim,
                    num_kv_heads * head_dim,
                ],
                dim=-1,
            )

            query_states = query_states.view(-1, num_heads, head_dim)
            key_states = key_states.view(-1, num_kv_heads, head_dim)
            value_states = value_states.view(-1, num_kv_heads, head_dim)
            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            scaling_factor = 1.0
            rotary_emb = self.rotary_emb
            if rotary_emb.inv_freq is None:
                rotary_emb.inv_freq = 1.0 / (rotary_emb.base**(torch.arange(
                    0,
                    rotary_emb.dim,
                    2,
                    dtype=torch.int64,
                    device=query_states.device).float() / rotary_emb.dim))
            inv_freq = rotary_emb.inv_freq
            query_states, key_states = fused_rotary_emb(
                query_states[None],
                key_states[None],
                context.position_ids_1d[None],
                inv_freq=inv_freq,
                scaling_factor=scaling_factor,
                out_q=query_states[None],
                out_k=key_states[None])
            return query_states[0], key_states[0], value_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)

        query_states, key_states, value_states = __rotary_emb_fn(
            query_states, key_states, value_states)

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

        attn_output = query_states
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
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            past_key_value,
            world_size=world_size,
        )


class PatchedDbrxExpertGLU(nn.Module):

    def _distribute_partition_fn(self, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""

        def __colwise_partition(name, param):
            dist_tensor = distribute_tensor(param, device_mesh, [Shard(0)])
            dist_tensor = try_to_local(dist_tensor)
            dist_param = torch.nn.Parameter(dist_tensor)
            mod.register_parameter(name, dist_param)

        if mod_name == '':
            __colwise_partition('w1', mod.w1)
            __colwise_partition('v1', mod.v1)
            __colwise_partition('w2', mod.w2)

    def forward(self, x: torch.Tensor, expert_w1: torch.Tensor,
                expert_v1: torch.Tensor,
                expert_w2: torch.Tensor) -> torch.Tensor:
        """remote code and transformers has different implementation."""
        gate_proj = x.matmul(expert_w1.t())
        up_proj = x.matmul(expert_v1.t())
        gate_proj = self.activation_fn(gate_proj)
        intermediate_states = gate_proj * up_proj
        down_proj = intermediate_states.matmul(expert_w2)
        return down_proj


def _div_up(a, b):
    """div up."""
    return (a + b - 1) // b


class PatchedDbrxExperts(nn.Module):

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

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs

    def forward(self, x: torch.Tensor, weights: torch.Tensor,
                top_weights: torch.Tensor,
                top_experts: torch.LongTensor) -> torch.Tensor:
        """moe forward."""

        def __get_expert_index(num_experts, first_experts_id, last_experts_id):
            """get expert index."""
            idxs, top_xs = [None] * num_experts, [None] * num_experts
            # Loop over all available experts in the model
            for expert_idx in range(first_experts_id, last_experts_id):
                pos = torch.nonzero(top_experts == expert_idx)
                if pos.size(0) > 0:
                    top_x, idx = pos.t()
                    idxs[expert_idx] = idx
                    top_xs[expert_idx] = top_x
            return idxs, top_xs

        bsz, q_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)

        num_experts = self.mlp.moe_num_experts

        first_experts_id, last_experts_id = self._get_expert_range(num_experts)

        idxs, top_xs = __get_expert_index(num_experts, first_experts_id,
                                          last_experts_id)

        ffn_hidden_size = self.mlp.ffn_hidden_size

        w1_chunked = self.mlp.w1.unflatten(0, (-1, ffn_hidden_size))
        v1_chunked = self.mlp.v1.unflatten(0, (-1, ffn_hidden_size))
        w2_chunked = self.mlp.w2.unflatten(0, (-1, ffn_hidden_size))

        for expert_idx in range(first_experts_id, last_experts_id):
            idx, top_x = idxs[expert_idx], top_xs[expert_idx]
            if idx is None:
                continue

            mlp_idx = expert_idx - first_experts_id
            current_state = x.index_select(dim=0, index=top_x)
            expert_out = self.mlp(
                current_state,
                w1_chunked[mlp_idx],
                v1_chunked[mlp_idx],
                w2_chunked[mlp_idx],
            )
            expert_out *= top_weights[top_x, idx, None]

            out.index_add_(0, top_x, expert_out)

        out = out.reshape(bsz, q_len, hidden_size)
        return out


class PatchedDbrxModel(nn.Module):

    def _continuous_batching_forward(
        self,
        input_ids: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        """forward impl."""
        output_attentions = False
        use_cache = True
        output_router_logits = False

        inputs_embeds = self.wte(input_ids)

        # Attention mask is not necessary in continuous batching
        attention_mask = None
        cache_position = None

        hidden_states = inputs_embeds

        for idx, block in enumerate(self.blocks):
            past_key_value = past_key_values[idx]
            block_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = block_outputs[0]

        hidden_states = self.norm_f(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        """Rewrite of LlamaModel.forward."""
        return self._continuous_batching_forward(
            input_ids,
            position_ids,
            past_key_values,
        )
