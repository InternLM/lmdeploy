# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DeviceMesh
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..dist_utils import (colwise_split_parallelize_linear_fn,
                          rowwise_parallelize_linear_fn)
from ..kernels import fill_kv_cache, fused_rotary_emb, paged_attention_fwd

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
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


class PatchedVisionExpertMLP(nn.Module):

    def forward(self, hidden_states: 'torch.Tensor(B, L, D)',
                token_type_ids: 'torch.LongTensor(B, L)'):
        context = self.context.context
        only_has_language = context.is_decoding
        if not context.is_decoding:
            # for embedding splitting
            if hasattr(context, 'vision_token_mask') and hasattr(
                    context, 'language_token_mask'):
                vision_token_mask = context.vision_token_mask
                language_token_mask = context.language_token_mask
            else:
                vision_token_mask, language_token_mask = get_vision_expert_mask(
                    token_type_ids)
            only_has_language = vision_token_mask.numel() == 0

        if only_has_language:
            output = self.language_mlp(hidden_states)
        else:
            output = torch.empty_like(hidden_states)
            output[:, vision_token_mask, :] = self.vision_mlp(
                hidden_states[:, vision_token_mask, :])
            output[:, language_token_mask, :] = self.language_mlp(
                hidden_states[:, language_token_mask, :])
        return output


class PatchedVisionExpertAttention(nn.Module):

    def _distribute_qkv_linear(self, mod: nn.Module, device_mesh: DeviceMesh):
        """distribute qkv linear."""
        num_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, 'num_multi_query_heads', num_heads)
        head_dim = self.config.hidden_size // num_heads
        sections = [
            self.config.hidden_size, num_kv_heads * head_dim,
            num_kv_heads * head_dim
        ]
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
        num_heads = self.config.num_attention_heads // world_size
        num_kv_heads = getattr(self.config, 'num_multi_query_heads',
                               self.config.num_attention_heads) // world_size

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        hidden_size = num_heads * head_dim
        only_has_language = context.is_decoding
        if not context.is_decoding:
            # for embedding splitting
            if hasattr(context, 'vision_token_mask') and hasattr(
                    context, 'language_token_mask'):
                vision_token_mask = context.vision_token_mask
                language_token_mask = context.language_token_mask
            else:
                vision_token_mask, language_token_mask = get_vision_expert_mask(
                    token_type_ids)
            only_has_language = vision_token_mask.numel() == 0

        def __qkv_proj(hidden_states):
            """qkv_proj."""
            if only_has_language:
                mixed_raw_layer = self.language_expert_query_key_value(
                    hidden_states)
            else:
                shape = list(hidden_states.shape)
                shape[-1] = hidden_size + head_dim * num_kv_heads * 2
                mixed_raw_layer = torch.empty(shape,
                                              dtype=hidden_states.dtype,
                                              device=hidden_states.device)

                mixed_raw_layer[:,
                                vision_token_mask, :] = self.vision_expert_query_key_value(
                                    hidden_states[:, vision_token_mask, :])
                mixed_raw_layer[:,
                                language_token_mask, :] = self.language_expert_query_key_value(
                                    hidden_states[:, language_token_mask, :])
            query_states, key_states, value_states = torch.split(
                mixed_raw_layer, [
                    hidden_size, head_dim * num_kv_heads,
                    head_dim * num_kv_heads
                ],
                dim=-1)
            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            """rotary embedding func."""
            scaling_factor = getattr(self.rotary_emb, 'scaling_factor', 1.0)
            inv_freq = self.rotary_emb.inv_freq

            query_states, key_states = fused_rotary_emb(
                query_states[None],
                key_states[None],
                position_ids[None],
                inv_freq=inv_freq,
                scaling_factor=scaling_factor,
                out_q=query_states[None],
                out_k=key_states[None])
            return query_states[0], key_states[0], value_states

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

        if only_has_language:
            attn_output = self.language_expert_dense(context_layer)
        else:
            ctx_shape = list(context_layer.shape)
            ctx_shape[-1] *= world_size
            attn_output = torch.empty(ctx_shape,
                                      dtype=hidden_states.dtype,
                                      device=hidden_states.device)

            attn_output[:, vision_token_mask, :] = self.vision_expert_dense(
                context_layer[:, vision_token_mask, :])
            attn_output[:,
                        language_token_mask, :] = self.language_expert_dense(
                            context_layer[:, language_token_mask, :])

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
        position_ids = _get_cogvlm_position_ids(context)

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            token_type_ids = vision_embedding_indexing.int().unsqueeze(0)
            vision_embedding_indexing = torch.arange(
                vision_embedding_indexing.numel(),
                device=vision_embedding_indexing.device
            )[vision_embedding_indexing]
            # multi-modality
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)
        else:
            token_type_ids = torch.ones_like(
                input_ids, dtype=torch.int,
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
        x: 'torch.BoolTensor(B, L)') -> 'torch.LongTensor(B, L)':
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
    position_id_offsets = vision_input_info.history_image_token_lengths - vision_input_info.history_image_nums * 3
    if inputs.is_decoding:
        position_ids = inputs.history_lengths - position_id_offsets
    else:
        if vision_input_info.input_embeddings is not None and len(
                vision_input_info.input_embeddings) > 0:
            starts = inputs.history_lengths - vision_input_info.history_lengths
            ends = starts + q_seq_length
            token_type_ids = vision_input_info.input_embedding_indexing.to(
                torch.int)
            history_position_lengths = vision_input_info.history_lengths - position_id_offsets
            position_ids_all = history_position_lengths[:,
                                                        None] + build_position_ids(
                                                            token_type_ids)
            position_ids = torch.cat([
                pids[s:e]
                for (pids, s, e) in zip(position_ids_all, starts, ends)
            ])
            vision_token_mask_all, _ = get_vision_expert_mask(token_type_ids)
            vision_token_mask = torch.cat([
                masks[s:e]
                for (masks, s, e) in zip(vision_token_mask_all, starts, ends)
            ])
            mask_indexing = torch.arange(vision_token_mask.shape[-1],
                                         device=vision_token_mask.device)
            vision_token_mask_new = mask_indexing[vision_token_mask]
            language_token_mask_new = mask_indexing[~vision_token_mask]

            context.vision_token_mask = vision_token_mask_new
            context.language_token_mask = language_token_mask_new
        else:
            position_ids = context.attention_mask.long().cumsum(-1) - 1
            position_ids += (inputs.history_lengths -
                             position_id_offsets).unsqueeze(-1)
            device = position_ids.device
            position_ids_1d = [
                ids[:l]
                for ids, l in zip(position_ids.cpu(), q_seq_length.cpu())
            ]
            position_ids = torch.cat(position_ids_1d).to(device)

    return position_ids
