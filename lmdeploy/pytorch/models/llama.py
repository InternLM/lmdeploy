# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import transformers
from packaging import version
from torch import nn
from torch.distributed._tensor import DeviceMesh
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..kernels import apply_rotary_pos_emb as apply_rotary_pos_emb_old
from ..kernels import (fill_kv_cache, fused_rotary_emb, paged_attention_fwd,
                       rms_norm)
from ..weight_loader.dist_utils import (colwise_parallelize_linear,
                                        rowwise_parallelize_linear)

TRANSFORMERS_VERSION = version.parse(transformers.__version__)


class LlamaRMSNorm(nn.Module):
    """Rewrite RMSNorm."""

    def forward(self, hidden_states):
        """forward."""
        # torch.nn.functional.normalize based implementation might leads
        # to wrong output
        ret = rms_norm(hidden_states, self.weight, self.variance_epsilon)

        return ret


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Rewrite module of LlamaAttention."""

    def _load_weights(self, loader, rank: int = 0, world_size: int = 1):
        """load weights."""
        for mod_name in ['q_proj', 'k_proj', 'v_proj']:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        rowwise_parallelize_linear(self.o_proj,
                                   loader,
                                   rank=rank,
                                   world_size=world_size,
                                   prefix='o_proj')

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_default_impl(
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

        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_key_value_heads // world_size
        head_dim = self.head_dim
        hidden_size = num_heads * head_dim

        def __qkv_proj(hidden_states):
            """qkv proj."""
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            return query_states, key_states, value_states

        def __rotary_emb_fn_old(query_states, key_states, value_states):
            """rotary embedding old."""
            if max_kv_seq_length >= self.rotary_emb.max_seq_len_cached:
                # create larger cache
                cos, sin = self.rotary_emb(value_states,
                                           seq_len=max_kv_seq_length + 128)
            cos = self.rotary_emb.cos_cached
            sin = self.rotary_emb.sin_cached
            query_states, key_states = apply_rotary_pos_emb_old(
                query_states,
                key_states,
                cos,
                sin,
                position_ids,
                context.position_ids_1d,
                q_embed=query_states,
                k_embed=key_states)
            return query_states, key_states, value_states

        def __rotary_emb_fn_438_naive(query_states, key_states, value_states):
            """rotary embedding transformers>4.38."""
            cos, sin = self.rotary_emb(value_states,
                                       context.position_ids_1d[None])
            cos = cos[0]
            sin = sin[0]
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin)
            return query_states, key_states, value_states

        def __rotary_emb_fn_438_fused(query_states, key_states, value_states):
            scaling_factor = getattr(self.rotary_emb, 'scaling_factor', 1.0)
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

        def __rotary_emb_fn_438(query_states, key_states, value_states):
            rotary_name = type(self.rotary_emb).__name__
            if rotary_name in [
                    'LlamaRotaryEmbedding', 'LlamaLinearScalingRotaryEmbedding'
            ]:
                return __rotary_emb_fn_438_fused(query_states, key_states,
                                                 value_states)
            else:
                return __rotary_emb_fn_438_naive(query_states, key_states,
                                                 value_states)

        def __rotary_emb_fn(query_states, key_states, value_states):
            """rotary embedding."""
            if TRANSFORMERS_VERSION >= version.parse('4.38.0'):
                return __rotary_emb_fn_438(query_states, key_states,
                                           value_states)
            else:
                return __rotary_emb_fn_old(query_states, key_states,
                                           value_states)

        query_states, key_states, value_states = __qkv_proj(hidden_states)

        query_states = query_states.view(-1, num_heads, head_dim)
        key_states = key_states.view(-1, num_kv_heads, head_dim)
        value_states = value_states.view(-1, num_kv_heads, head_dim)

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
        attn_output = attn_output.reshape(*hidden_states.shape[:-1],
                                          hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

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
        """Rewrite implementation of LlamaAttention.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        assert not output_attentions
        return self._contiguous_batching_forward_default_impl(
            hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            attention_mask=attention_mask,
            world_size=world_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of LlamaAttention.forward."""
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


class LlamaMLP(nn.Module):

    def _load_weights(self, loader, rank: int = 0, world_size: int = 1):
        """load weights."""
        for mod_name in ['gate_proj', 'up_proj']:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        rowwise_parallelize_linear(self.down_proj,
                                   loader,
                                   rank=rank,
                                   world_size=world_size,
                                   prefix='down_proj')

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs


class LlamaModel(nn.Module):

    def _continuous_batching_forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of LlamaModel.forward."""
        output_attentions = False
        use_cache = True

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Attention mask is not necessary in continuous batching
        attention_mask = None

        hidden_states = inputs_embeds

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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite of LlamaModel.forward."""
        return self._continuous_batching_forward(
            input_ids,
            position_ids,
            past_key_values,
            inputs_embeds,
        )
