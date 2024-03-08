# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..dist_utils import (colwise_parallelize_linear_fn,
                          rowwise_parallelize_linear_fn, try_to_local)
from ..kernels import fill_kv_cache, paged_attention_fwd


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor):
    """apply rotary."""
    dtype = t.dtype
    t_ = t.float()
    t_ = (t_ * freqs.cos()) + (rotate_half(t_) * freqs.sin())
    return t_.to(dtype)


class PatchedQWenAttention(nn.Module):

    def _distribute_partition_fn(self, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['c_attn']:
            for name, param in mod.named_parameters():
                splited_param = param.split(self.hidden_size, dim=0)
                updated_param = []
                for p in splited_param:
                    dist_tensor = distribute_tensor(p, device_mesh, [Shard(0)])
                    dist_tensor = try_to_local(dist_tensor)
                    updated_param.append(dist_tensor)
                param = torch.cat(updated_param)
                dist_param = torch.nn.Parameter(param)
                mod.register_parameter(name, dist_param)
        elif mod_name in ['c_proj']:
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
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of QWenAttention.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """

        def __qkv_proj(hidden_states):
            """qkv_proj."""
            qkv_states = self.c_attn(hidden_states)
            b, seq_len, _ = qkv_states.size()
            query_states, key_states, value_states = qkv_states.chunk(3, dim=2)
            num_heads = self.num_heads // world_size
            query_states = query_states.view(b, seq_len, num_heads,
                                             self.head_dim)
            key_states = key_states.view(b, seq_len, num_heads, self.head_dim)
            value_states = value_states.view(b, seq_len, num_heads,
                                             self.head_dim)
            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            """rotary embedding func."""
            max_seq_len = position_ids.size(-1)
            kv_seq_len = max_seq_len + max(history_lengths)
            if (self.use_dynamic_ntk
                    and kv_seq_len == hidden_states.size()[1]):
                import math
                context_value = math.log(kv_seq_len / self.seq_length, 2) + 1
                ntk_alpha = 2**math.ceil(context_value) - 1
                ntk_alpha = max(ntk_alpha, 1)
                self._ntk_cached = ntk_alpha
            else:
                ntk_alpha = self._ntk_cached
            rotary_pos_emb = self.rotary_emb(
                kv_seq_len, ntk_alpha=ntk_alpha).to(hidden_states.device)
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = (rotary_pos_emb, ) * 2
            q_pos_emb, k_pos_emb = rotary_pos_emb

            q_pos_emb = q_pos_emb[:, position_ids_1d, :, :]
            k_pos_emb = k_pos_emb[:, position_ids_1d, :, :]
            query_states = apply_rotary_pos_emb(query_states, q_pos_emb)
            key_states = apply_rotary_pos_emb(key_states, k_pos_emb)

            return query_states, key_states, value_states

        context = self.context.context
        history_lengths = context.history_lengths
        block_offsets = context.block_offsets
        position_ids_1d = context.position_ids_1d

        query_states, key_states, value_states = __qkv_proj(hidden_states)
        query_states, key_states, value_states = __rotary_emb_fn(
            query_states, key_states, value_states)
        query_states = query_states.flatten(0, 1)
        key_states = key_states.flatten(0, 1)
        value_states = value_states.flatten(0, 1)
        q_start_loc = context.q_start_loc
        q_seq_length = context.q_seq_length

        fill_kv_cache(key_states,
                      value_states,
                      past_key_value[0],
                      past_key_value[1],
                      q_start_loc,
                      q_seq_length,
                      block_offsets=block_offsets,
                      history_lengths=history_lengths,
                      context=context)

        attn_output = query_states
        kv_seq_length = context.kv_seq_length
        max_seq_len = position_ids.size(-1)
        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_seq_len,
        )
        attn_output = attn_output.flatten(1, 2)
        attn_output = self.c_proj(attn_output)
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            world_size=world_size,
        )


class PatchedQWenMLP(nn.Module):

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['w1', 'w2']:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['c_proj']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs


class PatchedQWenBlock(nn.Module):

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        layernorm_output = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            layernorm_output,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        attn_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        residual = hidden_states
        layernorm_input = attn_output + residual

        layernorm_output = self.ln_2(layernorm_input)

        residual = layernorm_input
        mlp_output = self.mlp(layernorm_output)
        hidden_states = residual + mlp_output
        outputs = (hidden_states, ) + outputs[1:]
        return outputs


class PatchedQWenModel(nn.Module):

    def _continuous_batching_forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of LlamaModel.forward."""
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # Attention mask is not necessary in continuous batching
        hidden_states = inputs_embeds

        # decoder layers
        for idx, decoder_layer in enumerate(self.h):
            past_key_value = (past_key_values[idx]
                              if past_key_values is not None else None)
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_value,
            hidden_states=None,
            attentions=None,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite of LlamaModel.forward."""
        return self._continuous_batching_forward(
            input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )


class PatchedRMSNorm(nn.Module):
    """Rewrite RMSNorm."""

    def forward(self, hidden_states):
        """forward."""
        # torch.nn.functional.normalize based implementation might leads
        # to wrong output
        from ..kernels import rms_norm
        ret = rms_norm(hidden_states, self.weight, self.eps)

        return ret
