# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from lmdeploy.pytorch.layers import (ApplyRotaryEmb, Attention, EmbeddingType,
                                     RMSNorm, SiluAndMul, build_merged_linear,
                                     build_rotary_embedding)

from ..weight_loader.dist_utils import (colwise_parallelize_linear,
                                        rowwise_parallelize_linear)


class LlamaAttention(nn.Module):
    """Rewrite module of LlamaAttention."""

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
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

    def _update_model_fn(self):
        """update model."""

        # qkv
        self.qkv_proj = build_merged_linear(self.q_proj, self.k_proj,
                                            self.v_proj)
        del self.q_proj, self.k_proj, self.v_proj

        # rotary embedding
        old_emb = self.rotary_emb
        rotary_name = type(old_emb).__name__
        if rotary_name in [
                'LlamaRotaryEmbedding', 'LlamaLinearScalingRotaryEmbedding'
        ]:
            emb_type = EmbeddingType.LinearScaling
        elif rotary_name == 'LlamaDynamicNTKScalingRotaryEmbedding':
            emb_type = EmbeddingType.DynamicNTKScaling
        self.rotary_emb = build_rotary_embedding(
            old_emb.dim,
            old_emb.max_position_embeddings,
            old_emb.base,
            old_emb.scaling_factor,
            emb_type,
        )

        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_key_value_heads // world_size
        head_size = self.head_dim
        self.attn_fwd = Attention(
            num_heads,
            head_size,
            num_kv_heads=num_kv_heads,
            v_head_size=head_size,
        )

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
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

        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_key_value_heads // world_size
        head_dim = self.head_dim
        hidden_size = num_heads * head_dim

        def __qkv_proj(hidden_states):
            """qkv proj."""
            qkv_states = self.qkv_proj(hidden_states)
            # (-1, heads, head_dim)
            qkv_states = qkv_states.flatten(0,
                                            -2).unflatten(-1, (-1, head_dim))
            query_states, key_states, value_states = qkv_states.split(
                (num_heads, num_kv_heads, num_kv_heads), dim=1)

            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states):
            """rotary embedding."""
            if not hasattr(context, '_cos'):
                cos, sin = self.rotary_emb(query_states,
                                           context.position_ids_1d[None])
                cos = cos[0]
                sin = sin[0]
                context._cos = cos
                context._sin = sin
            else:
                cos = context._cos
                sin = context._sin

            query_states, key_states = self.apply_rotary_pos_emb(query_states,
                                                                 key_states,
                                                                 cos,
                                                                 sin,
                                                                 inplace=True)
            return query_states, key_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)
        query_states, key_states = __rotary_emb_fn(query_states, key_states)
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            context.attn_meta,
            inplace=True,
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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of LlamaAttention.forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_default_impl(
            hidden_states,
            position_ids,
            past_key_value,
            output_attentions,
            attention_mask=attention_mask,
            world_size=world_size,
        )


class LlamaMLP(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
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
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs

    def _update_model_fn(self):
        """update model."""

        # gate up
        self.gate_up_proj = build_merged_linear(self.gate_proj, self.up_proj)
        del self.gate_proj, self.up_proj

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class PatchedLlamaDecoderLayer(nn.Module):

    def _update_model_fn(self):
        """update model."""
        input_layernorm = self.input_layernorm
        self.input_layernorm = RMSNorm(input_layernorm.weight,
                                       input_layernorm.variance_epsilon)
        post_attention_layernorm = self.post_attention_layernorm
        self.post_attention_layernorm = RMSNorm(
            post_attention_layernorm.weight,
            post_attention_layernorm.variance_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[List[torch.FloatTensor]] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class LlamaModel(nn.Module):

    def _update_model_fn(self):
        """update model."""
        norm = self.norm
        self.norm = RMSNorm(norm.weight, norm.variance_epsilon)

    def _continuous_batching_forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of LlamaModel.forward."""
        context = self.context.context
        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)
        # Attention mask is not necessary in continuous batching
        attention_mask = None

        hidden_states = inputs_embeds
        residual = None
        for idx, decoder_layer in enumerate(self.layers):

            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                residual=residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

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
