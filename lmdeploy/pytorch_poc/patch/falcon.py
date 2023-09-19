# Copyright (c) OpenMMLab. All rights reserved.
# Adapter from: https://huggingface.co/tiiuae/falcon-7b-instruct

import importlib
import logging
import math
import os
import warnings
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions, QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast, TokenClassifierOutput)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.falcon.modeling_falcon import build_alibi_tensor

from lmdeploy.pytorch_poc.kernels import paged_attention_fwd

# from transformers.models.falcon.modeling_falcon import

# from transformers.models.falcon.configuration_falcon import FalconConfig
# from transformers.models.falcon.configuration_falcon import FalconConfig as RWConfig

logger = logging.getLogger(__name__)


# rotary pos emb helpers (torch.jit.script does not seem to support staticmethod...)
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class PatchedFalconRotaryEmbedding(nn.Module):
    """Implementation adapted from Huggingface transformers."""

    def _patched_set_cos_sin_cache(self, seq_len, device, dtype):
        self.seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)

        if dtype in [torch.float16, torch.bfloat16]:
            emb = emb.float()

        # logger.debug(f"emb.shape = {emb.shape}")
        self.cos_cached = emb.cos()[None, :, :]
        self.sin_cached = emb.sin()[None, :, :]

        self.cos_cached = self.cos_cached.type(dtype)
        self.sin_cached = self.sin_cached.type(dtype)

    def patched_cos_sin(self,
                        position_ids: torch.Tensor,
                        device='cpu',
                        dtype=torch.bfloat16) -> torch.Tensor:
        total_length = int(position_ids.max().item()) + 1

        if (self.seq_len_cached is None) or (total_length >
                                             self.seq_len_cached):
            self._patched_set_cos_sin_cache(total_length, device, dtype)
        # position_ids.shape == [1, packed_seq_len]
        return (
            self.cos_cached[:, position_ids[0], None, :],
            self.sin_cached[:, position_ids[0], None, :],
        )

    def _contiguous_batching_forward(self, query: torch.Tensor,
                                     key: torch.Tensor,
                                     position_ids: torch.Tensor):
        # batch, seq_len, *_ = query.shape
        cos, sin = self.patched_cos_sin(position_ids,
                                        device=query.device,
                                        dtype=query.dtype)
        logger.debug(f'sin = {sin}')
        logger.debug(f'cos = {cos}')
        return (
            (query * cos) + (rotate_half(query) * sin),
            (key * cos) + (rotate_half(key) * sin),
        )

    def forward(self, query, key, position_ids_or_past_key_values_length=0):
        use_origin = False

        if use_origin:
            return self.origin_mod(query, key,
                                   position_ids_or_past_key_values_length)
        else:
            # print("continuous forwarding")
            return self._contiguous_batching_forward(
                query, key, position_ids_or_past_key_values_length)


class PatchedFalconAttention(nn.Module):

    def _contiguous_batching_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # prepare inputs for continuous batch forwarding
        context = self.context.context
        history_lengths = context.history_lengths
        q_start_loc, q_seq_length = self.context.q_seq_info
        history_lengths = q_seq_length.new_tensor(history_lengths)
        kv_seq_length = q_seq_length + history_lengths
        max_seq_len = q_seq_length.max().item()
        config = getattr(self, 'config', None)

        fused_qkv = self.query_key_value(
            hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        # num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        # logger.debug(f'query_layer (just after split) {query_layer.size()}=\n {query_layer}')
        # logger.debug(f'key_layer (just after split) {key_layer.size()}=\n {key_layer}')
        # logger.debug(f'value_layer {value_layer.size()}=\n {value_layer}')
        batch_size, query_length, _, _ = query_layer.shape

        # query_layer = query_layer.reshape(batch_size * self.num_heads,
        #                                   query_length, self.head_dim)
        # key_layer = key_layer.reshape(
        #     batch_size * num_kv_heads,
        #     query_length,
        #     self.head_dim,
        # )
        # value_layer = value_layer.reshape(batch_size * num_kv_heads,
        #                                   query_length, self.head_dim)

        # past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]

        if isinstance(self.maybe_rotary, nn.Module):
            query_layer, key_layer = self.maybe_rotary(query_layer, key_layer,
                                                       position_ids)

        # logger.debug(f'query_layer (just after rotary) {query_layer.size()}=\n {query_layer}')
        # logger.debug(f'key_layer (just after rotary) {key_layer.size()}=\n {key_layer}')

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, kv_length, head_dim]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            # key_layer = torch.cat((past_key, key_layer), dim=1)
            # value_layer = torch.cat((past_value, value_layer), dim=1)

            # logger.debug(f'key_layer =\n {key_layer.size()}')
            # logger.debug(f'value_layer =\n {value_layer.size()}')
            # logger.debug(f'past_key =\n {past_key.size()}')
            # logger.debug(f'past_value =\n {past_value.size()}')
            context.fill_cache(
                key_layer[0],
                value_layer[0],
                q_start_loc,
                q_seq_length,
                past_key,
                past_value,
            )

        _, _, kv_length, _ = key_layer.shape
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        # attention_mask_float = (attention_mask * 1.0).masked_fill(
        #     attention_mask, float("-1e9")).to(query_layer.dtype)

        # query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1,
        #                                    self.head_dim)
        # key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1,
        #                                self.head_dim)
        # value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1,
        #                                    self.head_dim)

        attn_output = torch.empty_like(query_layer)
        block_offsets = context.block_offsets
        block_size = past_key.size(1)

        if alibi is None:
            # if hasattr(F, "scaled_dot_product_attention") and not output_attentions:
            #     # TODO: deprecate this once we add FA2 support in Falcon
            #     logger.warning_once(
            #         "The current implementation of Falcon calls `torch.scaled_dot_product_attention` directly, this will be deprecated in the"
            #         " future in favor of the `BetterTransformer` API. Please install the latest optimum library with `pip install -U optimum` and call "
            #         "`model.to_bettertransformer()` to benefit from `torch.scaled_dot_product_attention` and future performance optimizations."
            #     )

            #     attn_output = F.scaled_dot_product_attention(
            #         query_layer_, key_layer_, value_layer_, attention_mask_float, 0.0, is_causal=False
            #     )
            #     attention_scores = None
            # else:
            #     attention_scores = query_layer_ @ key_layer_.transpose(-1, -2)
            #     attention_scores /= math.sqrt(self.head_dim)

            #     attention_scores = F.softmax(
            #         attention_scores + attention_mask_float, dim=-1, dtype=hidden_states.dtype
            #     )
            #     attn_output = attention_scores @ value_layer_

            # attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
            # attn_output = attn_output.permute(0, 2, 1, 3)
            # attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

            logger.debug(f'query_layer_ = {query_layer.shape}\n%s',
                         query_layer)
            logger.debug('block_offsets = \n%s', block_offsets)
            logger.debug(
                f'key_layer filled in  = {past_key[block_offsets[0].item()].shape}\n%s',
                past_key[block_offsets[0].item()])
            logger.debug(
                f'value_layer filled in  = {past_value[block_offsets[0].item()].shape}\n%s',
                past_value[block_offsets[0].item()])
            logger.debug(f'q_start_loc =\n {q_start_loc}')
            logger.debug(f'q_seq_length =\n {q_seq_length}')

            paged_attention_fwd(query_layer,
                                past_key,
                                past_value,
                                attn_output,
                                block_offsets,
                                b_start_loc=q_start_loc,
                                b_seq_len=q_seq_length,
                                b_kv_seq_len=kv_seq_length,
                                max_input_len=max_seq_len,
                                BLOCK=block_size)

            attn_output = attn_output.reshape(batch_size, query_length, -1)
            logger.debug(
                f'attn_output (before dense) {attn_output.size()} = \n%s',
                attn_output)

            output_tensor = self.dense(attn_output)

            if output_attentions:
                return output_tensor, present, None
            else:
                return output_tensor, present

        else:
            # matmul_result = query_layer_ @ key_layer_.transpose(-1, -2)

            # # change view to [batch_size, num_heads, q_length, kv_length]
            # attention_scores = matmul_result.view(batch_size, self.num_heads,
            #                                       query_length, kv_length)

            # # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            # input_dtype = attention_scores.dtype
            # # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            # if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
            #     attention_scores = attention_scores.to(torch.float32)
            # # Matt (HF) note: We could possibly use F.scaled_dot_product_attention here too, by
            # # adding (alibi * self.inv_norm_factor) to attention_mask_float. I think this would be mathematically
            # # equivalent and more performant, but there might be a numerical difference. If you're reading this
            # # and you'd like to experiment and maybe file a PR, feel free!
            # attention_logits = attention_scores + alibi.view(
            #     batch_size, self.num_heads, 1, -1)
            # attention_logits *= self.inv_norm_factor
            # attention_probs = F.softmax(attention_logits +
            #                             attention_mask_float,
            #                             dim=-1,
            #                             dtype=hidden_states.dtype)
            # # [batch_size, num_heads, q_length, kv_length]
            # attention_probs = self.attention_dropout(attention_probs)

            # if head_mask is not None:
            #     attention_probs = attention_probs * head_mask

            # # change view [batch_size, num_heads, q_length, kv_length]
            # attention_probs_reshaped = attention_probs.view(
            #     batch_size, self.num_heads, query_length, kv_length)

            context_layer = torch.empty_like(query_layer)
            paged_attention_fwd(query_layer,
                                past_key,
                                past_value,
                                context_layer,
                                block_offsets,
                                b_start_loc=q_start_loc,
                                b_seq_len=q_seq_length,
                                b_kv_seq_len=kv_seq_length,
                                max_input_len=max_seq_len,
                                BLOCK=block_size)

            # # matmul: [batch_size * num_heads, q_length, head_dim]
            # context_layer = (attention_probs_reshaped @ value_layer_).flatten(
            #     0, 1)

            # change view [batch_size, q_length, num_heads * head_dim]
            # context_layer = self._merge_heads(context_layer)
            context_layer = context_layer.reshape(batch_size, query_length, -1)

            output_tensor = self.dense(context_layer)

            if output_attentions:
                return output_tensor, present, None
            else:
                return output_tensor, present

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):

        position_ids = self.context.position_ids
        use_origin = False
        if use_origin:
            return self.origin_mod(hidden_states, alibi, attention_mask,
                                   layer_past, head_mask, use_cache,
                                   output_attentions)
        else:
            # logger.debug('continuous forwarding')
            return self._contiguous_batching_forward(
                hidden_states, position_ids, alibi, attention_mask, layer_past,
                head_mask, use_cache, output_attentions)


class PatchedFalconDecoderLayer(nn.Module):

    def _contiguous_batching_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states

        if self.config.new_decoder_architecture:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self attention.
        attn_outputs = self.self_attention(
            attention_layernorm_out,
            position_ids=position_ids,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(attention_output,
                                       residual,
                                       self.config.attention_dropout,
                                       training=self.training)
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]

        # MLP.
        mlp_output = self.mlp(mlp_layernorm_out)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(mlp_output,
                             residual,
                             self.config.hidden_dropout,
                             training=self.training)

        if use_cache:
            outputs = (output, ) + outputs
        else:
            outputs = (output, ) + outputs[1:]

        return outputs  # hidden_states, present, attentions

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):

        use_origin = False

        if use_origin:
            return self.origin_mod(hidden_states=hidden_states,
                                   alibi=alibi,
                                   attention_mask=attention_mask,
                                   layer_past=layer_past,
                                   head_mask=head_mask,
                                   use_cache=use_cache,
                                   output_attentions=output_attentions)
        else:
            # print("continuous forwarding")
            return self._contiguous_batching_forward(
                hidden_states=hidden_states,
                position_ids=position_ids,
                alibi=alibi,
                attention_mask=attention_mask,
                layer_past=layer_past,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions)


class PatchedFalconModel(nn.Module):

    def _contiguous_batching_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                        ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...],
               BaseModelOutputWithPastAndCrossAttentions]:

        history_lengths = self.context.context.history_lengths

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_alibi = getattr(self, 'use_alibi', getattr(self, 'alibi', False))
        logger.debug(f'use_alibi = {use_alibi}')

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')

        # if past_key_values is None:
        #     past_key_values = tuple([None] * len(self.h))
        # else:
        #     past_key_values = self._convert_to_rw_cache(past_key_values)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask,
                                       self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        # presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        logger.debug(f'history_lengths = {history_lengths}')
        # past_key_values_length = history_lengths
        # past_key_values_length = 0
        # if past_key_values[0] is not None:
        #     past_key_values_length = past_key_values[0][0].shape[
        #         1]  # 1 because RW-cache, not standard format
        # if attention_mask is None:
        #     attention_mask = torch.ones(
        #         (batch_size, seq_length + past_key_values_length),
        #         device=hidden_states.device)
        # else:
        #     attention_mask = attention_mask.to(hidden_states.device)

        if use_alibi:
            alibi = build_alibi_tensor(attention_mask,
                                       self.num_heads,
                                       dtype=hidden_states.dtype)
        else:
            alibi = None

        # causal_mask = self._prepare_attn_mask(
        #     attention_mask,
        #     input_shape=(batch_size, seq_length),
        #     past_key_values_length=past_key_values_length,
        # )

        seqlen = self.context.position_ids.max().item()
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            logger.debug(f'====SeqLen {seqlen} Decode Layer {i}=======')
            outputs = block(
                hidden_states,
                # position_ids=position_ids,
                layer_past=layer_past,
                attention_mask=None,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

            hidden_states = outputs[0]
            # if use_cache is True:
            #     presents = presents + (outputs[1], )

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1], )

        # Add last hidden state

        logger.debug(f'hidden_states before in_f \n {hidden_states}')

        hidden_states = self.ln_f(hidden_states)

        logger.debug(f'hidden_states after in_f \n {hidden_states}')

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        # if presents is not None:
        #     presents = self._convert_cache_to_standard_format(
        #         presents, batch_size)

        if not return_dict:
            return tuple(v for v in [
                hidden_states, past_key_values, all_hidden_states,
                all_self_attentions
            ] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                        ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...],
               BaseModelOutputWithPastAndCrossAttentions]:

        use_origin = False
        if use_origin:
            return self.origin_mod(input_ids=input_ids,
                                   past_key_values=past_key_values,
                                   attention_mask=attention_mask,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   use_cache=use_cache,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)
        else:
            # print("continuous forwarding")
            return self._contiguous_batching_forward(
                input_ids=input_ids,
                # position_ids=position_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)


class PatchedFalconForCausalLM(nn.Module):

    def _contiguous_batching_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                        ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...],
               BaseModelOutputWithPastAndCrossAttentions]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            position_ids=position_ids,  # add position_ids
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        # remove labels to compute loss

        if not return_dict:
            output = (lm_logits, ) + transformer_outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                        ...]] = None,
        return_dict: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        use_origin: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor, ...],
               BaseModelOutputWithPastAndCrossAttentions]:

        use_origin = False
        if use_origin:
            return self.origin_mod(input_ids=input_ids,
                                   past_key_values=past_key_values,
                                   attention_mask=attention_mask,
                                   head_mask=None,
                                   inputs_embeds=None,
                                   use_cache=False,
                                   labels=None,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)
        else:
            # print("continuous forwarding causal lm")
            return self._contiguous_batching_forward(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
