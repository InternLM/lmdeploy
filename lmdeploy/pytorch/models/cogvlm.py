# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DeviceMesh
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..dist_utils import (colwise_split_parallelize_linear_fn,
                          rowwise_parallelize_linear_fn)
from ..kernels import apply_rotary_pos_emb, fill_kv_cache, paged_attention_fwd

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

# flake8: noqa: F821


def get_vision_expert_mask(
    token_type_ids: 'torch.LongTensor(B, L)'
) -> '[torch.BoolTensor(B, L), torch.BoolTensor(B, L)]':
    vision_token_mask = torch.zeros_like(token_type_ids, dtype=torch.bool)
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1]
                                 == VISION_TOKEN_TYPE) & (token_type_ids[:, 1:]
                                                          == VISION_TOKEN_TYPE)
    return vision_token_mask


class PatchedVisionExpertAttention(nn.Module):

    def _distribute_qkv_linear(self, mod: nn.Module, device_mesh: DeviceMesh):
        """distribute qkv linear."""
        sections = [self.num_heads * self.head_dim] * 3
        colwise_split_parallelize_linear_fn(mod, sections, device_mesh)

    def _distribute_partition_fn(self, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in [
                'vision_expert_query_key_value',
                'language_expert_query_key_value'
        ]:
            self._distribute_qkv_linear(mod, device_mesh=device_mesh)
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
        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_heads // world_size
        head_dim = self.head_dim
        hidden_size = num_heads * head_dim
        # for embedding splitting
        if hasattr(context, 'vision_token_mask'):
            vision_token_mask = context.vision_token_mask
        else:
            vision_token_mask = get_vision_expert_mask(token_type_ids)

        language_token_mask = ~vision_token_mask

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
                position_ids_1d=position_ids)
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
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        inputs_embeds = self.embed_tokens(input_ids)
        token_type_ids = None
        position_ids = _get_cogvlm_position_ids(context)
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            # multi-modality
            token_type_ids = vision_embedding_indexing.int()
            inputs_embeds[vision_embedding_indexing] = vision_embeddings.to(
                inputs_embeds)
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


def build_position_ids(
    x: 'torch.BoolTensor(B, L)',
    attention_mask: Optional['torch.BoolTensor(B, L)'] = None
) -> 'torch.LongTensor(B, L)':
    if attention_mask is not None:
        tmp = x.clone()
        tmp[~(attention_mask.bool())] = -1
    else:
        tmp = x.clone()
    # image boi eoi token as LANGUAGE_TOKEN_TYPE
    is_boi_eoi = torch.zeros_like(x, dtype=torch.bool)
    is_boi_eoi[:, 1:] |= (tmp[:, 1:] == VISION_TOKEN_TYPE) & (
        tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, 0] |= (tmp[:, 0] == VISION_TOKEN_TYPE)
    is_boi_eoi[:, :-1] |= (tmp[:, :-1] == VISION_TOKEN_TYPE) & (
        tmp[:, 1:] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, -1] |= (tmp[:, -1] == VISION_TOKEN_TYPE)
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE
    # final position ids
    y = torch.zeros_like(x, dtype=torch.long)
    y[:, 1:] = (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE) | (
        (tmp[:, 1:] == VISION_TOKEN_TYPE) &
        (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE))
    y = y.cumsum(dim=-1)
    return y


def _get_cogvlm_position_ids(context):
    """get cogvlm position_ids."""
    inputs = context.inputs

    q_seq_length = inputs.seq_length
    vision_input_info = inputs.vision_inputs
    history_position_lengths = vision_input_info.history_lengths - vision_input_info.history_image_token_lengths + vision_input_info.history_image_nums * 3
    device = inputs.history_lengths.device
    if inputs.is_decoding:
        position_ids = history_position_lengths
    else:
        if context.input_embeddings is not None and len(
                context.input_embeddings) > 0:
            token_type_ids = torch.tensor(
                [[e is not None for e in li]
                 for li in vision_input_info.input_embeddings],
                dtype=torch.bool,
                device=device)
            position_ids_all = vision_input_info.history_lengths[:,
                                                                 None] + build_position_ids(
                                                                     token_type_ids
                                                                 )
            starts = inputs.history_lengths - vision_input_info.history_lengths
            ends = starts + q_seq_length
            position_ids = torch.cat([
                pids[s:e]
                for (pids, s, e) in zip(position_ids_all, starts, ends)
            ]).unsqueeze(0)
            vision_token_mask_all = get_vision_expert_mask(token_type_ids)
            vision_token_mask = torch.cat([
                masks[s:e]
                for (masks, s, e) in zip(vision_token_mask_all, starts, ends)
            ]).unsqueeze(0)
            context.vision_token_mask = vision_token_mask
        else:
            position_ids = (context.attention_mask.long().cumsum(-1) -
                            1).squeeze(0)
            position_ids += history_position_lengths + inputs.history_lengths - vision_input_info.history_lengths
    return position_ids
