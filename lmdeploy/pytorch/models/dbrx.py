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
        position_ids: Optional[torch.LongTensor] = None,
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
            inv_freq = self.rotary_emb.inv_freq
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
            position_ids,
            past_key_value,
            world_size=world_size,
        )


class PatchedDbrxExpertGLU(nn.Module):

    def _distribute_partition_fn(self, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""

        world_size = dist.get_world_size()

        def __partiton_moe(weight: nn.Parameter, name: str):
            weight = weight.view(self.moe_num_experts, self.ffn_hidden_size,
                                 self.hidden_size)
            weight = distribute_tensor(weight, device_mesh, [Shard(1)])
            weight = try_to_local(weight)
            weight = weight.flatten(0, 1)
            self.register_parameter(name, nn.Parameter(weight))

        if getattr(self, '__finish_partition', False):
            return

        __partiton_moe(self.w1, 'w1')
        __partiton_moe(self.v1, 'v1')
        __partiton_moe(self.w2, 'w2')

        self.ffn_hidden_size = self.ffn_hidden_size // world_size
        self.__finish_partition = True

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs


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
