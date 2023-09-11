# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.functional as F
import transformers
from packaging import version
from torch import nn
from torch.distributed._tensor import (DeviceMesh, DTensor, Replicate,
                                       distribute_tensor)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import rotate_half

from lmdeploy.pytorch_poc.dist_utils import (colwise_parallelize_linear_fn,
                                             rowwise_parallelize_linear_fn,
                                             try_to_local)
from lmdeploy.pytorch_poc.kernels import paged_attention_fwd
from lmdeploy.pytorch_poc.utils import bind_sigature

_tp_from_config = version.parse(
    transformers.__version__) >= version.parse('4.32')


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):

    def try_to_local(tensor):
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        return tensor

    # indexing does not support DTensor
    old_q = q
    old_k = k
    position_ids = try_to_local(position_ids)
    cos = try_to_local(cos)
    sin = try_to_local(sin)
    q = try_to_local(q)
    k = try_to_local(k)

    # The first two dimensions of cos and sin are always 1,
    # so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids]  # [bs, 1, seq_len, dim]
    sin = sin[position_ids]  # [bs, 1, seq_len, dim]
    seq_length = position_ids[..., -1] + 1
    cos = [s[:l] for s, l in zip(cos, seq_length)]
    sin = [s[:l] for s, l in zip(sin, seq_length)]
    cos = torch.cat(cos, 0).unsqueeze(1)
    sin = torch.cat(sin, 0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    # recovery dtensor if necessary
    if isinstance(old_q, DTensor):
        q_embed = DTensor.from_local(q_embed,
                                     device_mesh=old_q.device_mesh,
                                     placements=old_q.placements,
                                     run_check=False)

    if isinstance(old_k, DTensor):
        k_embed = DTensor.from_local(k_embed,
                                     device_mesh=old_k.device_mesh,
                                     placements=old_k.placements,
                                     run_check=False)

    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1,
    repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :].expand(slen, num_key_value_heads,
                                                  n_rep, head_dim)
    return hidden_states.reshape(slen, num_key_value_heads * n_rep, head_dim)


class LlamaRMSNorm(nn.Module):

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        if mod_name != '':
            return

        mod.weight = nn.Parameter(
            distribute_tensor(mod.weight,
                              device_mesh=device_mesh,
                              placements=[Replicate()]))


class LlamaAttention(nn.Module):

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        if mod_name != '':
            return

        assert hasattr(mod, 'q_proj')
        assert hasattr(mod, 'k_proj')
        assert hasattr(mod, 'v_proj')
        assert hasattr(mod, 'o_proj')

        # qkv
        colwise_parallelize_linear_fn(mod.q_proj, device_mesh=device_mesh)
        colwise_parallelize_linear_fn(mod.k_proj, device_mesh=device_mesh)
        colwise_parallelize_linear_fn(mod.v_proj, device_mesh=device_mesh)

        # o
        rowwise_parallelize_linear_fn(mod.o_proj, device_mesh=device_mesh)

    @classmethod
    def _distribute_input_fn(cls, inputs, inputs_dict,
                             device_mesh: DeviceMesh):
        inputs_dict = bind_sigature([
            'hidden_states', 'attention_mask', 'position_ids',
            'past_key_value', 'output_attentions', 'use_cache'
        ], inputs, inputs_dict)
        inputs_dict['position_ids'] = try_to_local(inputs_dict['position_ids'])

        return tuple(), inputs_dict

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        # return outputs
        attn_output, attn_weights, past_key_value = outputs

        local_out = attn_output.to_local()
        dist.all_reduce(local_out)
        attn_output = DTensor.from_local(local_out,
                                         device_mesh=device_mesh,
                                         placements=[Replicate()],
                                         run_check=False)

        return attn_output, attn_weights, past_key_value

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
        origin_config = origin_self.config if _tp_from_config else origin_self

        context = self.context.context
        history_lengths = context.history_lengths
        max_seq_len = position_ids.size(-1)

        if origin_config.pretraining_tp > 1:
            key_value_slicing = (
                origin_self.num_key_value_heads *
                origin_self.head_dim) // origin_config.pretraining_tp
            query_slices = origin_self.q_proj.weight.split(
                (origin_self.num_heads * origin_self.head_dim) //
                origin_config.pretraining_tp,
                dim=0)
            key_slices = origin_self.k_proj.weight.split(key_value_slicing,
                                                         dim=0)
            value_slices = origin_self.v_proj.weight.split(key_value_slicing,
                                                           dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(origin_config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(origin_config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(origin_config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = origin_self.q_proj(hidden_states)
            key_states = origin_self.k_proj(hidden_states)
            value_states = origin_self.v_proj(hidden_states)

        query_states = query_states.view(-1, origin_self.num_heads,
                                         origin_self.head_dim)
        key_states = key_states.view(-1, origin_self.num_key_value_heads,
                                     origin_self.head_dim)
        value_states = value_states.view(-1, origin_self.num_key_value_heads,
                                         origin_self.head_dim)

        kv_seq_len = max_seq_len + max(history_lengths)
        # TODO: setup past_key_value with paged attention
        cos, sin = origin_self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        q_start_loc, q_seq_length = self.context.q_seq_info

        history_lengths = q_seq_length.new_tensor(history_lengths)
        kv_seq_length = q_seq_length + history_lengths

        context.fill_cache(
            key_states,
            value_states,
            q_start_loc,
            q_seq_length,
            past_key_value[0],
            past_key_value[1],
        )

        # TODO: fix GQA
        # # repeat k/v heads if n_kv_heads < n_heads
        # key_states = repeat_kv(key_states, origin_self.num_key_value_groups)
        # value_states = repeat_kv(value_states,
        #                          origin_self.num_key_value_groups)

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

        if origin_config.pretraining_tp > 1:
            attn_output = attn_output.split(origin_self.hidden_size //
                                            origin_config.pretraining_tp,
                                            dim=1)
            o_proj_slices = origin_self.o_proj.weight.split(
                origin_self.hidden_size // origin_config.pretraining_tp, dim=1)
            attn_output = sum([
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(origin_config.pretraining_tp)
            ])
        else:
            attn_output = origin_self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

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
                hidden_states, attention_mask, position_ids, past_key_value,
                output_attentions, use_cache)


class LlamaMLP(nn.Module):

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        if mod_name != '':
            return

        assert hasattr(mod, 'gate_proj')
        assert hasattr(mod, 'up_proj')
        assert hasattr(mod, 'down_proj')

        # gate
        colwise_parallelize_linear_fn(mod.gate_proj, device_mesh=device_mesh)

        # up
        colwise_parallelize_linear_fn(mod.up_proj, device_mesh=device_mesh)

        # down
        rowwise_parallelize_linear_fn(mod.down_proj, device_mesh=device_mesh)

    @classmethod
    def _distribute_output_fn(cls, outputs: DTensor, device_mesh: DeviceMesh):
        # return outputs
        local_out = outputs.to_local()
        dist.all_reduce(local_out)
        outputs = DTensor.from_local(local_out,
                                     device_mesh=device_mesh,
                                     placements=[Replicate()],
                                     run_check=False)

        return outputs

    def forward(self, x):

        if isinstance(x, DTensor):
            gate_out = self.gate_proj(x)
            gate_out.to_local()[...] = self.act_fn(gate_out.to_local())
            down_proj = self.down_proj(gate_out * self.up_proj(x))
        else:
            down_proj = self.origin_mod.forward(x)

        return down_proj


class LlamaDecoderLayer(nn.Module):

    @classmethod
    def _distribute_input_fn(cls, inputs, inputs_dict,
                             device_mesh: DeviceMesh):
        inputs_dict = bind_sigature([
            'hidden_states', 'attention_mask', 'position_ids',
            'past_key_value', 'output_attentions', 'use_cache'
        ], inputs, inputs_dict)

        inputs_dict['hidden_states'] = distribute_tensor(
            inputs_dict['hidden_states'],
            device_mesh=device_mesh,
            placements=[Replicate()])

        if inputs_dict['attention_mask'] is not None:
            inputs_dict['attention_mask'] = distribute_tensor(
                inputs_dict['attention_mask'],
                device_mesh=device_mesh,
                placements=[Replicate()])

        return tuple(), inputs_dict


class LlamaModel(nn.Module):

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        # return outputs

        outputs.last_hidden_state = try_to_local(outputs.last_hidden_state)

        hidden_states = outputs.hidden_states
        if hidden_states is not None and len(hidden_states) > 0:
            outputs.hidden_states = tuple(
                try_to_local(state) for state in hidden_states)

        return outputs

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
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = (use_cache
                     if use_cache is not None else self.config.use_cache)

        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        assert position_ids is not None, (
            'position_ids can not be none when using continuous batching mode.'
        )
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

            past_key_value = past_key_values[
                idx] if past_key_values is not None else None
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

        use_origin = self.context.use_origin

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids '
                             'and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            assert input_ids.dim() == 2
        elif inputs_embeds is not None:
            assert inputs_embeds.dim() == 3
        else:
            raise ValueError(
                'You have to specify '
                'either decoder_input_ids or decoder_inputs_embeds')

        if use_origin:
            # use origin model
            return self.origin_mod(input_ids, attention_mask, position_ids,
                                   past_key_values, inputs_embeds, use_cache,
                                   output_attentions, output_hidden_states,
                                   return_dict)
        else:
            return self._continuous_batching_forward(
                input_ids, attention_mask, position_ids, past_key_values,
                inputs_embeds, use_cache, output_attentions,
                output_hidden_states, return_dict)
