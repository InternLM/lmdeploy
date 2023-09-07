# Copyright (c) OpenMMLab. All rights reserved.

import importlib
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import TRANSFORMERS_DYNAMIC_MODULE_NAME, logging

from lmdeploy.pytorch_poc.kernels import paged_attention_fwd

chatglm_module = importlib.import_module('.chatglm2-6b.modeling_chatglm',
                                         TRANSFORMERS_DYNAMIC_MODULE_NAME)

apply_rotary_pos_emb = chatglm_module.apply_rotary_pos_emb
RotaryEmbedding = chatglm_module.RotaryEmbedding
split_tensor_along_last_dim = chatglm_module.split_tensor_along_last_dim
ChatGLMModel = chatglm_module.ChatGLMModel

import logging

logger = logging.getLogger(__name__)


class PatchedSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h] and returns output of
    the same size.
    """

    # def __init__(self, origin_mod: nn.Module, context: Any):
    #     super().__init__()
    #     self.origin_mod = origin_mod
    #     self.context = context

    # def _allocate_memory(self,
    #                      inference_max_sequence_len,
    #                      batch_size,
    #                      device=None,
    #                      dtype=None):
    #     if self.multi_query_attention:
    #         num_attention_heads = self.num_multi_query_groups_per_partition
    #     else:
    #         num_attention_heads = self.num_attention_heads_per_partition
    #     return torch.empty(
    #         inference_max_sequence_len,
    #         batch_size,
    #         num_attention_heads,
    #         self.hidden_size_per_attention_head,
    #         dtype=dtype,
    #         device=device,
    #     )

    def _contiguous_batching_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]

        # input 要改回1D适应cache和attn

        origin_self = self

        context = self.context.context
        history_lengths = context.history_lengths
        # logger.debug('history_lengths = %s', history_lengths)

        mixed_x_layer = origin_self.query_key_value(hidden_states)

        if origin_self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    origin_self.num_attention_heads_per_partition *
                    origin_self.hidden_size_per_attention_head,
                    origin_self.num_multi_query_groups_per_partition *
                    origin_self.hidden_size_per_attention_head,
                    origin_self.num_multi_query_groups_per_partition *
                    origin_self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(query_layer.size()[:-1] + (
                origin_self.num_attention_heads_per_partition,
                origin_self.hidden_size_per_attention_head,
            ))
            key_layer = key_layer.view(key_layer.size()[:-1] + (
                origin_self.num_multi_query_groups_per_partition,
                origin_self.hidden_size_per_attention_head,
            ))
            value_layer = value_layer.view(value_layer.size()[:-1] + (
                origin_self.num_multi_query_groups_per_partition,
                origin_self.hidden_size_per_attention_head,
            ))
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                origin_self.num_attention_heads_per_partition,
                3 * origin_self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer,
             value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # [b, sq, np, hn]
        query_layer, key_layer, value_layer = [
            k.transpose(0, 1) for k in [query_layer, key_layer, value_layer]
        ]

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            q_start_loc, q_seq_length = self.context.q_seq_info

            q_start_loc: torch.Tensor
            history_lengths = q_seq_length.new_tensor(history_lengths)
            kv_seq_length = q_seq_length + history_lengths
            max_seq_len = q_seq_length.max().item()

            # logger.debug('===Context Fill===')
            # logger.debug('key_states.shape = %s', key_layer.shape)
            # logger.debug('value_states.shape = %s', value_layer.shape)
            # logger.debug('q_start_loc = %s', q_start_loc)
            # logger.debug('q_seq_length = %s', q_seq_length)
            # logger.debug('cache_k = %s', cache_k.shape)

            # kv length 和谁绑定在一起内聚性更好？
            context.fill_cache(
                key_layer[0],
                value_layer[0],
                q_start_loc,
                q_seq_length,
                cache_k,
                cache_v,
            )
            # logger.debug('cache_k.shape = %s', cache_k.shape)
            # torch.cuda.synchronize()

            # key_layer = torch.cat((cache_k, key_layer), dim=0)
            # value_layer = torch.cat((cache_v, value_layer), dim=0)
        # logger.debug('use cache = %s', use_cache)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        # if origin_self.multi_query_attention:
        #     key_layer = key_layer.unsqueeze(-2)
        #     key_layer = key_layer.expand(
        #         -1,
        #         -1,
        #         -1,
        #         origin_self.num_attention_heads_per_partition //
        #         origin_self.num_multi_query_groups_per_partition,
        #         -1,
        #     )
        #     key_layer = key_layer.contiguous().view(key_layer.size()[:2] + (
        #         origin_self.num_attention_heads_per_partition,
        #         origin_self.hidden_size_per_attention_head,
        #     ))
        #     value_layer = value_layer.unsqueeze(-2)
        #     value_layer = value_layer.expand(
        #         -1,
        #         -1,
        #         -1,
        #         origin_self.num_attention_heads_per_partition //
        #         origin_self.num_multi_query_groups_per_partition,
        #         -1,
        #     )
        #     value_layer = value_layer.contiguous().view(
        #         value_layer.size()[:2] + (
        #             origin_self.num_attention_heads_per_partition,
        #             origin_self.hidden_size_per_attention_head,
        #         ))

        # ==================================
        # core attention computation
        # ==================================

        # logger.debug('===Attention===')
        # logger.debug('query_states.shape = %s', query_layer.shape)
        # logger.debug('cache_k.shape = %s', cache_k.shape)
        # logger.debug('max_seq_len = %s', max_seq_len)

        # if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
        #     context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
        #                                                                      is_causal=True)
        # else:
        #     if attention_mask is not None:
        #         attention_mask = ~attention_mask
        #     context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
        #                                                                      attention_mask)
        # context_layer = context_layer.permute(2, 0, 1, 3)
        # new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        # context_layer = context_layer.reshape(*new_context_layer_shape)

        context_layer = torch.empty_like(query_layer)

        block_offsets = context.block_offsets
        block_size = cache_k.size(1)

        # logger.debug('block_offsets %s', block_offsets)

        paged_attention_fwd(query_layer,
                            cache_k,
                            cache_v,
                            context_layer,
                            block_offsets,
                            b_start_loc=q_start_loc,
                            b_seq_len=q_seq_length,
                            b_kv_seq_len=kv_seq_length,
                            max_input_len=max_seq_len,
                            BLOCK=block_size)

        context_layer = context_layer.transpose(1, 0).flatten(-2)
        # new_context_layer_shape = context_layer.size()[:-2] + (origin_self.hidden_size_per_partition,)
        # context_layer = context_layer.reshape(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output = origin_self.dense(context_layer)

        return output, kv_cache

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
        output_attentions=False,
    ):

        use_origin = False
        if use_origin:
            return self.origin_mod(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_cache,
                use_cache,
            )
        else:
            # logger.debug('continuous forwarding')
            return self._contiguous_batching_forward(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_cache,
                use_cache,
            )


class PatchedChatGLMModel(nn.Module):

    # def __init__(self, origin_mod: nn.Module, context: Any):
    #     super().__init__()
    #     self.origin_mod = origin_mod
    #     self.context = context
    #     # for compatibility
    #     self.output_layer = origin_mod.output_layer

    def _contiguous_batching_forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                            ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None):
        orig_self = self.origin_mod
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                orig_self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else orig_self.config.use_cache
        return_dict = return_dict if return_dict is not None else orig_self.config.use_return_dict

        # logger.debug('')
        # logger.debug('=' * 66)
        # logger.debug('Enter GLMModel')
        # logger.debug(f'input_ids in model: {input_ids}')
        # logger.debug(f'position_ids in model: {position_ids}')
        # logger.debug(f'attention_mask in model: {attention_mask}')
        # logger.debug(f'full_attention_mask in model: {full_attention_mask}')

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = orig_self.embedding(input_ids)

        if orig_self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = orig_self.get_prompt(
                    batch_size=batch_size,
                    device=input_ids.device,
                    dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask.new_ones(
                        (batch_size, orig_self.pre_seq_len)), attention_mask
                ],
                                           dim=-1)

        # if past_key_values is None:
        #     logger.debug(f'past_key_values[0][0].shape = None')
        # else:
        #     logger.debug(
        #         f'past_key_values[0][0].shape = {past_key_values[0][0].shape}')

        # logger.debug(
        #     f'full_attention_mask in model before process: {full_attention_mask}'
        # )

        # if full_attention_mask is None:
        #     if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
        #         full_attention_mask = orig_self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # logger.debug(
        #     f'full_attention_mask in model after process: {full_attention_mask}'
        # )

        # Rotary positional embeddings
        rotary_pos_emb = orig_self.rotary_pos_emb(orig_self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = orig_self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states)

        if not return_dict:
            return tuple(v for v in [
                hidden_states, presents, all_hidden_states, all_self_attentions
            ] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def forward(
        self,
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                        ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        use_origin = False
        if use_origin:
            return self.origin_mod(input_ids=input_ids,
                                   position_ids=position_ids,
                                   attention_mask=attention_mask,
                                   full_attention_mask=full_attention_mask,
                                   past_key_values=past_key_values,
                                   inputs_embeds=inputs_embeds,
                                   use_cache=use_cache,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)
        else:
            # print("continuous forwarding")
            return self._contiguous_batching_forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                full_attention_mask=full_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
