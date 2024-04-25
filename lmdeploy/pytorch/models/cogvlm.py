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

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

# flake8: noqa: F821


def get_expert_mask(
    token_type_ids: 'torch.LongTensor(B, L)'
) -> '[torch.BoolTensor(B, L), torch.BoolTensor(B, L)]':
    vision_token_mask = torch.zeros_like(token_type_ids, dtype=torch.bool)
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1]
                                 == VISION_TOKEN_TYPE) & (token_type_ids[:, 1:]
                                                          == VISION_TOKEN_TYPE)
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


class PatchedVisionExpertAttention(nn.Module):

    def _distribute_partition_fn(self, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in [
                'vision_expert_query_key_value',
                'language_expert_query_key_value'
        ]:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['vision_expert_dense', 'language_expert_dense']:
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
        token_type_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of Attention.forward.

        Add continuous batching support. Add paged attention support.
        """
        context = self.context.context
        q_start_loc = context.q_start_loc
        q_seq_length = context.q_seq_length
        kv_seq_length = context.kv_seq_length
        block_offsets = context.block_offsets
        max_q_seq_length = context.max_q_seq_length
        max_kv_seq_length = context.max_kv_seq_length
        position_ids_1d = context.position_ids_1d
        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_heads // world_size
        head_dim = self.head_dim
        hidden_size = num_heads * head_dim
        vision_token_mask, language_token_mask = get_expert_mask(
            token_type_ids)

        def __qkv_proj(hidden_states):
            """qkv_proj."""
            shape = list(hidden_states.shape)
            shape[-1] = shape[-1] * 3 // world_size
            mixed_raw_layer = torch.empty(shape,
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

            mixed_raw_layer[
                vision_token_mask] = self.vision_expert_query_key_value(
                    hidden_states[vision_token_mask])
            mixed_raw_layer[
                language_token_mask] = self.language_expert_query_key_value(
                    hidden_states[language_token_mask])
            query_states, key_states, value_states = torch.split(
                mixed_raw_layer, hidden_size, dim=-1)
            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            """rotary embedding func."""
            cos, sin = self.rotary_emb(value_states, seq_len=max_kv_seq_length)
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos.squeeze(1),
                sin.squeeze(1),
                position_ids,
                position_ids_1d=position_ids_1d)
            return query_states, key_states, value_states

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

        context_layer = query_states
        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[1],
            context_layer,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_q_seq_length,
        )
        context_layer = context_layer.reshape(*hidden_states.shape[:-1], -1)
        ctx_shape = list(context_layer.shape)
        ctx_shape[-1] *= world_size

        attn_output = torch.empty(ctx_shape,
                                  dtype=hidden_states.dtype,
                                  device=hidden_states.device)

        attn_output[vision_token_mask] = self.vision_expert_dense(
            context_layer[vision_token_mask])
        attn_output[language_token_mask] = self.language_expert_dense(
            context_layer[language_token_mask])
        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            past_key_value=past_key_value,
            world_size=world_size,
        )


class PatchedCogVLMModel(nn.Module):

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # not allow for inputs_embeds, because we want to process image feature
        assert input_ids is not None
        context = self.context.context
        # get inputs from context
        vision_embeddings = context.inputs.input_embeddings
        vision_embedding_ranges = context.inputs.input_embedding_ranges
        token_type_ids = context.inputs.token_type_ids
        inputs_embeds = self.embed_tokens(input_ids)

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            # multi-modality
            assert token_type_ids is not None, 'multi-modality requires `token_type_ids`!'
            assert len(vision_embeddings) == len(vision_embedding_ranges)
            for emb, ranges in zip(vision_embeddings, vision_embedding_ranges):
                inputs_embeds[0, ranges[0]:ranges[1]] = emb.to(inputs_embeds)
        else:
            if token_type_ids is None:
                token_type_ids = torch.ones_like(
                    input_ids, dtype=torch.long,
                    device=input_ids.device) * LANGUAGE_TOKEN_TYPE
        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            layer_outputs = decoder_layer(
                hidden_states,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_value,
            hidden_states=None,
            attentions=None,
        )
