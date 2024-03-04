# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..dist_utils import (colwise_parallelize_linear_fn,
                          rowwise_parallelize_linear_fn, try_to_local)
from ..kernels import apply_rotary_pos_emb
from ..kernels.alibi_pagedattention import alibi_paged_attention_fwd
from ..kernels.fill_kv_cache import fill_kv_cache
from ..kernels.pagedattention import paged_attention_fwd


class PatchedRMSNorm(nn.Module):
    """Rewrite RMSNorm."""

    def forward(self, hidden_states):
        """forward."""
        from ..kernels import rms_norm
        epsilon = getattr(self, 'epsilon', None)
        if epsilon is None:
            epsilon = getattr(self, 'variance_epsilon', 1e-10)
        ret = rms_norm(hidden_states, self.weight, epsilon)

        return ret


def _attention_partition_fn(mod_name: str, mod: nn.Module,
                            device_mesh: DeviceMesh):
    """A function for attention partition."""

    def __w_pack_linear_fn(mod: nn.Module):
        """fn for w pack linear."""
        for name, param in mod.named_parameters():
            param = param.unflatten(0, (3, -1))
            dist_tensor = distribute_tensor(param, device_mesh, [Shard(1)])
            dist_tensor = try_to_local(dist_tensor)
            dist_tensor = dist_tensor.flatten(0, 1)
            dist_param = torch.nn.Parameter(dist_tensor)
            mod.register_parameter(name, dist_param)

    def __w_pack_lora_linear_fn(mod: nn.Module):
        """fn for w pack lora linear."""
        mod._tp_mode = 'colwise'
        base_layer = mod.base_layer
        __w_pack_linear_fn(base_layer)

        for lora_a_mod in mod.lora_A.values():
            colwise_parallelize_linear_fn(lora_a_mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

        for lora_b_mod in mod.lora_B.values():
            __w_pack_linear_fn(lora_b_mod)

    if mod_name in ['W_pack']:
        from peft.tuners.lora import Linear as LoraLinear
        if isinstance(mod, LoraLinear):
            __w_pack_lora_linear_fn(mod)
        else:
            __w_pack_linear_fn(mod)
    elif mod_name in ['o_proj']:
        rowwise_parallelize_linear_fn(mod,
                                      device_mesh=device_mesh,
                                      to_local=True)


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        return _attention_partition_fn(mod_name, mod, device_mesh)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

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
        """Rewrite of Attention.forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward(
            hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            world_size=world_size,
        )

    def _contiguous_batching_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of Attention.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        assert not output_attentions
        context = self.context.context
        history_lengths = context.history_lengths
        kv_seq_length = context.kv_seq_length
        q_seq_length = context.q_seq_length
        q_start_loc = context.q_start_loc
        block_offsets = context.block_offsets
        max_seq_length = context.max_seq_length

        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_heads // world_size
        head_dim = self.head_dim

        def _qkv_proj(hidden_states):
            """qkv proj."""
            proj = self.W_pack(hidden_states)
            return proj.chunk(3, -1)

        def _rotary_emb_fn(query_states, key_states, value_states):
            if hasattr(self, 'rotary_emb'):
                kv_seq_len = max_seq_length + max(history_lengths)
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids,
                    context.position_ids_1d)
            return query_states, key_states, value_states

        query_states, key_states, value_states = _qkv_proj(hidden_states)

        query_states = query_states.view(-1, num_heads, head_dim)
        key_states = key_states.view(-1, num_kv_heads, head_dim)
        value_states = value_states.view(-1, num_kv_heads, head_dim)

        query_states, key_states, value_states = _rotary_emb_fn(
            query_states, key_states, value_states)

        fill_kv_cache(key_states,
                      value_states,
                      past_key_value[0],
                      past_key_value[1],
                      q_start_loc,
                      q_seq_length,
                      block_offsets=block_offsets,
                      history_lengths=history_lengths,
                      context=context)

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
            max_seqlen=max_seq_length,
        )

        hidden_size = num_heads * head_dim
        attn_output = attn_output.reshape(*hidden_states.shape[:-1],
                                          hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class BaichuanAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        return _attention_partition_fn(mod_name, mod, device_mesh)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of BaichuanAttention.forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward(
            hidden_states,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            world_size=world_size,
        )

    def _contiguous_batching_forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of BaichuanAttention.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        assert not output_attentions
        context = self.context.context
        history_lengths = context.history_lengths
        kv_seq_length = context.kv_seq_length
        q_seq_length = context.q_seq_length
        q_start_loc = context.q_start_loc
        block_offsets = context.block_offsets
        max_seq_length = context.max_seq_length

        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_heads // world_size
        head_dim = self.head_dim

        def _qkv_proj(hidden_states):
            proj = self.W_pack(hidden_states)
            return proj.chunk(3, -1)

        query_states, key_states, value_states = _qkv_proj(hidden_states)
        query_states = query_states.view(-1, num_heads, head_dim)
        key_states = key_states.view(-1, num_kv_heads, head_dim)
        value_states = value_states.view(-1, num_kv_heads, head_dim)

        fill_kv_cache(key_states,
                      value_states,
                      past_key_value[0],
                      past_key_value[1],
                      q_start_loc,
                      q_seq_length,
                      block_offsets=block_offsets,
                      history_lengths=history_lengths,
                      context=context)

        attn_output = query_states

        num_heads_full = num_heads
        head_offset = 0
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            num_heads_full = num_heads * world_size
            head_offset = num_heads * rank
        alibi_paged_attention_fwd(query_states,
                                  past_key_value[0],
                                  past_key_value[1],
                                  attn_output,
                                  block_offsets,
                                  b_start_loc=q_start_loc,
                                  b_seq_len=q_seq_length,
                                  b_kv_seq_len=kv_seq_length,
                                  max_input_len=max_seq_length,
                                  head_offset=head_offset,
                                  num_heads=num_heads_full)

        hidden_size = num_heads * head_dim
        attn_output = attn_output.reshape(*hidden_states.shape[:-1],
                                          hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class BaichuanModel(nn.Module):

    def _continuous_batching_forward_7b(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of 7b BaichuanModel.forward."""
        output_attentions = False
        use_cache = True

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Attention mask is not necessary in continuous batching
        attention_mask = None

        hidden_states = inputs_embeds

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
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

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def _continuous_batching_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of BaichuanModel.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        use_cache = False
        output_attentions = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # Attention mask is not necessary in continuous batching
        attention_mask = None
        hidden_states = inputs_embeds

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        """Rewrite of BaichuanModel.forward."""
        if position_ids is not None:
            return self._continuous_batching_forward_7b(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
            )
        else:
            return self._continuous_batching_forward(
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
            )
