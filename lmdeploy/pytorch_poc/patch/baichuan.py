# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from lmdeploy.pytorch_poc.kernels import paged_attention_fwd

from .llama import apply_rotary_pos_emb


class BaichuanAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

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
        if self.context.use_origin:
            return self.origin_mod(hidden_states, attention_mask, position_ids,
                                   past_key_value, output_attentions,
                                   use_cache)
        else:
            return self._contiguous_batching_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache)

    def _contiguous_batching_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        assert not output_attentions
        origin_self = self.origin_mod

        context = self.context.context
        history_lengths = context.history_lengths
        max_seq_len = position_ids.size(-1)

        proj = origin_self.W_pack(hidden_states)
        proj = proj.unflatten(
            -1, (3, origin_self.hidden_size)).unsqueeze(0).transpose(
                0, -2).squeeze(-2)
        query_states = proj[0].view(-1, origin_self.num_heads,
                                    origin_self.head_dim)
        key_states = proj[1].view(-1, origin_self.num_heads,
                                  origin_self.head_dim)
        value_states = proj[2].view(-1, origin_self.num_heads,
                                    origin_self.head_dim)

        kv_seq_len = max_seq_len + max(history_lengths)
        # TODO: setup past_key_value with paged attention
        if hasattr(origin_self,
                   'rotary_emb'):  # baichuan-13B has no rotary_emb
            cos, sin = origin_self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)

        kv_seq_length = position_ids[..., -1] + 1
        q_seq_length = kv_seq_length - kv_seq_length.new_tensor(
            history_lengths)
        q_start_loc = q_seq_length.cumsum(0)
        q_start_loc = torch.cat([q_start_loc.new_zeros(1), q_start_loc[:-1]])
        context.fill_cache(
            key_states,
            value_states,
            q_start_loc,
            q_seq_length,
            past_key_value[0],
            past_key_value[1],
        )
        attn_output = torch.empty_like(query_states)

        block_offsets = context.block_offsets
        block_size = past_key_value[0].size(1)
        paged_attention_fwd(query_states,
                            past_key_value[0],
                            past_key_value[1],
                            attn_output,
                            block_offsets,
                            b_start_loc=q_start_loc,
                            b_seq_len=q_seq_length,
                            b_kv_seq_len=kv_seq_length,
                            max_input_len=max_seq_len,
                            BLOCK=block_size)
        attn_output = attn_output.reshape(-1, origin_self.hidden_size)

        attn_output = origin_self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class BaichuanLayer(torch.nn.Module):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.origin_mod.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.origin_mod.self_attn(  # noqa
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.origin_mod.post_attention_layernorm(hidden_states)
        hidden_states = self.origin_mod.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs


class BaichuanForCausalLM(nn.Module):

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = False,
                output_hidden_states: Optional[bool] = False,
                return_dict: Optional[bool] = True,
                **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.origin_mod.config.use_return_dict  # noqa

        # decoder outputs consists of
        # (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.origin_mod.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=kwargs.get('position_ids', None),
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.origin_mod.lm_head(hidden_states)

        loss = None

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
