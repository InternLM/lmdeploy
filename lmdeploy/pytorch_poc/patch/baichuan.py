# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from lmdeploy.pytorch_poc.patch.functional import \
    attention_forward_with_paged_attention

from .llama import apply_rotary_pos_emb


class Attention(nn.Module):
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
            return self.origin_mod(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            assert use_cache
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
        assert not output_attentions
        context = self.context.context
        history_lengths = context.history_lengths

        def _qkv_proj(hidden_states):
            proj = self.W_pack(hidden_states)
            return proj.chunk(3, -1)

        def _rotary_emb_fn(query_states, key_states, value_states):
            if hasattr(self, 'rotary_emb'):
                max_seq_len = position_ids.size(-1)
                kv_seq_len = max_seq_len + max(history_lengths)
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids)
            return query_states, key_states, value_states

        attn_output = attention_forward_with_paged_attention(
            hidden_states,
            history_lengths=history_lengths,
            block_offsets=context.block_offsets,
            num_heads=self.num_heads // world_size,
            num_kv_heads=self.num_heads // world_size,
            head_dim=self.head_dim,
            position_ids=position_ids,
            past_key_value=past_key_value,
            qkv_proj=_qkv_proj,
            o_proj=self.o_proj,
            rotary_emb_fn=_rotary_emb_fn,
        )

        return attn_output, None, past_key_value


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
            return self.origin_mod(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            assert use_cache
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
        assert not output_attentions
        context = self.context.context
        history_lengths = context.history_lengths

        def _qkv_proj(hidden_states):
            proj = self.W_pack(hidden_states)
            return proj.chunk(3, -1)

        _rotary_emb_fn = None

        attn_output = attention_forward_with_paged_attention(
            hidden_states,
            history_lengths=history_lengths,
            block_offsets=context.block_offsets,
            num_heads=self.num_heads // world_size,
            num_kv_heads=self.num_heads // world_size,
            head_dim=self.head_dim,
            position_ids=position_ids,
            past_key_value=past_key_value,
            qkv_proj=_qkv_proj,
            o_proj=self.o_proj,
            rotary_emb_fn=_rotary_emb_fn,
            bias_type='alibi',
        )

        return attn_output, None, past_key_value


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
        (
            hidden_states,
            self_attn_weights,
            present_key_value,
        ) = self.origin_mod.self_attn(  # noqa
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
        return_dict = (return_dict if return_dict is not None else
                       self.origin_mod.config.use_return_dict)  # noqa

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
