# Copyright (c) OpenMMLab. All rights reserved.
import pdb
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DeviceMesh
from transformers.modeling_outputs import BaseModelOutputWithPast

from lmdeploy.pytorch_poc.dist_utils import (colwise_parallelize_linear_fn,
                                             rowwise_parallelize_linear_fn)

from .functional import (apply_rotary_pos_emb,
                         attention_forward_with_paged_attention,
                         attention_forward_with_rerope, repeat_kv, rotate_half)


class LlamaAttention(nn.Module):
    """Rewrite module of LlamaAttention."""

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['q_proj', 'k_proj', 'v_proj']:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['o_proj']:
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
        context = self.context.context

        json_config = self.context.context.json_config
        history_lengths = context.history_lengths

        use_rerope = 'rerope' in json_config and json_config['rerope']
        if use_rerope:

            def apply_rotary_pos_emb_rerope(q, k, cos, sin, position_ids):
                # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
                cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
                sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
                cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
                sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
                q_embed = ((q * cos[:, -q.shape[0]:]) +
                           (rotate_half(q) * sin[:, -q.shape[0]:])
                           ).squeeze(0) if q is not None else None
                k_embed = ((k * cos) + (rotate_half(k) * sin)
                           ).squeeze(0) if k is not None else None
                return q_embed, k_embed

            def apply_rotary_pos_emb_rerope_v2(q, k, cos, sin, position_ids):
                # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
                assert 1 == position_ids.shape[0]

                _, __, seq_len, dim = cos.shape

                cos = cos[0, 0][position_ids].reshape(
                    seq_len, 1, dim)  # [bs, seq_len, dim] to [seq_len, 1, dim]
                sin = sin[0, 0][position_ids].reshape(
                    seq_len, 1, dim)  # [bs, seq_len, dim] to [seq_len, 1, dim]

                if q is not None:
                    q = rotate_half(q).mul_(sin[-q.shape[0]:]).add_(
                        q.mul_(cos[-q.shape[0]:]))
                if k is not None:
                    k = rotate_half(k).mul_(sin).add_(k.mul_(cos))
                return q, k

            def _rotary_emb_context_rerope_fn(query_states, key_states,
                                              value_states, position_ids,
                                              window):
                kv_seq_len, num_dim, dim = key_states.shape

                cos, sin = self.rotary_emb(value_states,
                                           seq_len=max(kv_seq_len, window + 1))

                query_states1, key_states1 = apply_rotary_pos_emb_rerope_v2(
                    query_states, key_states, cos, sin, position_ids)

                query_states2, _ = apply_rotary_pos_emb_rerope_v2(
                    query_states, None, cos, sin, position_ids * 0 + window)

                # repeat k/v heads if n_kv_heads < n_heads
                if self.num_key_value_groups > 1:
                    key_states1 = repeat_kv(key_states1,
                                            self.num_key_value_groups)
                    key_states2 = repeat_kv(key_states,
                                            self.num_key_value_groups)
                    value_states = repeat_kv(value_states,
                                             self.num_key_value_groups)
                else:
                    key_states2 = key_states

                query_states1 = query_states1.transpose(0, 1).reshape(
                    1, num_dim, kv_seq_len, dim).contiguous()
                query_states2 = query_states2.transpose(0, 1).reshape(
                    1, num_dim, kv_seq_len, dim).contiguous()
                key_states1 = key_states1.transpose(0, 1).reshape(
                    1, num_dim, kv_seq_len, dim).contiguous()
                key_states2 = key_states2.transpose(0, 1).reshape(
                    1, num_dim, kv_seq_len, dim).contiguous()
                value_states = value_states.transpose(0, 1).reshape(
                    1, num_dim, kv_seq_len, dim).contiguous()

                return query_states1, query_states2, key_states1, key_states2, value_states

            def _rotary_emb_generate_rerope_fn(key_states, value_states,
                                               position_ids, window):
                kv_seq_len = key_states.shape[0]
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

                position_ids = (position_ids[:, -1] -
                                position_ids).clip(max=window)
                _, key_states = apply_rotary_pos_emb_rerope_v2(
                    None, key_states, cos, -sin, position_ids)
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states,
                                         self.num_key_value_groups)
                return key_states, value_states

            attn_output = attention_forward_with_rerope(
                hidden_states,
                history_lengths=history_lengths,
                block_offsets=context.block_offsets,
                num_heads=self.num_heads // world_size,
                num_kv_heads=self.num_key_value_heads // world_size,
                head_dim=self.head_dim,
                position_ids=position_ids,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                context=context,
                q_proj=self.q_proj,
                k_proj=self.k_proj,
                v_proj=self.v_proj,
                o_proj=self.o_proj,
                rotary_emb_context_fn=_rotary_emb_context_rerope_fn,
                rotary_emb_generate_fn=_rotary_emb_generate_rerope_fn,
            )
        else:

            def _rotary_emb_fn(query_states, key_states, value_states):
                max_seq_len = position_ids.size(-1)
                kv_seq_len = max_seq_len + max(history_lengths)
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids,
                    getattr(context, 'position_ids_1d', None))
                return query_states, key_states, value_states

            attn_output = attention_forward_with_paged_attention(
                hidden_states,
                history_lengths=history_lengths,
                block_offsets=context.block_offsets,
                num_heads=self.num_heads // world_size,
                num_kv_heads=self.num_key_value_heads // world_size,
                head_dim=self.head_dim,
                position_ids=position_ids,
                past_key_value=past_key_value,
                context=context,
                q_proj=self.q_proj,
                k_proj=self.k_proj,
                v_proj=self.v_proj,
                o_proj=self.o_proj,
                rotary_emb_fn=_rotary_emb_fn,
            )
        return attn_output, None, past_key_value

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
        """Rewrite of LlamaAttention.forward."""
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
            return self._contiguous_batching_forward_impl(
                hidden_states,
                position_ids,
                past_key_value,
                output_attentions,
                attention_mask=attention_mask,
                world_size=world_size,
            )


class LlamaMLP(nn.Module):

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['gate_proj', 'up_proj']:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['down_proj']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs


class LlamaModel(nn.Module):

    def _continuous_batching_forward(
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
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of LlamaModel.forward."""
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)

        if use_cache is None:
            use_cache = self.config.use_cache

        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        assert (
            position_ids is not None
        ), 'position_ids can not be none when using continuous batching mode.'
        assert position_ids.dim() == 2

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Attention mask is not necessary in continuous batching
        attention_mask = None

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            past_key_value = (past_key_values[idx]
                              if past_key_values is not None else None)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1], )

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
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
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite of LlamaModel.forward."""
        use_origin = self.context.use_origin
        if use_origin:
            # use origin model
            return self.origin_mod(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
        else:
            return self._continuous_batching_forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
