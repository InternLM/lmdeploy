# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DeviceMesh
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..kernels import apply_rotary_pos_emb, fill_kv_cache, paged_attention_fwd
from ..weight_loader.dist_utils import (colwise_parallelize_linear,
                                        colwise_split_parallelize_linear,
                                        rowwise_parallelize_linear)


class PatchedQWenAttention(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        for mod_name in ['c_attn']:
            w_pack_out = self.c_attn.out_features
            sections = [w_pack_out // 3] * 3
            colwise_split_parallelize_linear(getattr(self, mod_name),
                                             sections,
                                             loader,
                                             rank=rank,
                                             world_size=world_size,
                                             prefix=mod_name)
        for mod_name in ['c_proj']:
            rowwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        rotary_pos_emb_list: Optional[List[List[torch.Tensor]]] = None,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of QWenAttention.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        context = self.context.context
        block_offsets = context.block_offsets
        position_ids = context.position_ids
        position_ids_1d = context.position_ids_1d
        max_kv_seq_length = context.max_kv_seq_length
        max_q_seq_length = context.max_q_seq_length
        kv_seq_length = context.kv_seq_length
        q_start_loc = context.q_start_loc
        q_seq_length = context.q_seq_length

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
            assert len(rotary_pos_emb_list) == 1, 'do not support dynamic ntk'
            cos, sin = rotary_pos_emb_list[0]
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                position_ids=position_ids,
                position_ids_1d=position_ids_1d)

            return query_states, key_states, value_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)
        if rotary_pos_emb_list is not None:
            query_states, key_states, value_states = __rotary_emb_fn(
                query_states, key_states, value_states)
        if max_kv_seq_length > self.seq_length and self.use_logn_attn:
            if self.logn_tensor.device != query_states.device or \
                    self.logn_tensor.dtype != query_states.dtype:
                self.logn_tensor = self.logn_tensor.to(
                    query_states.device).type_as(query_states)
            logn_tensor = self.logn_tensor[:, position_ids_1d, :, :]
            query_states = query_states * logn_tensor.expand_as(query_states)

        query_states = query_states.flatten(0, 1)
        key_states = key_states.flatten(0, 1)
        value_states = value_states.flatten(0, 1)

        fill_kv_cache(key_states,
                      value_states,
                      past_key_value[0],
                      past_key_value[1],
                      q_start_loc,
                      q_seq_length,
                      kv_seq_length=kv_seq_length,
                      max_q_seq_length=max_q_seq_length,
                      block_offsets=block_offsets)

        attn_output = query_states

        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_q_seq_length,
        )
        attn_output = attn_output.flatten(1, 2)
        attn_output = self.c_proj(attn_output)
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        rotary_pos_emb_list: Optional[List[List[torch.Tensor]]] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            past_key_value=layer_past,
            rotary_pos_emb_list=rotary_pos_emb_list,
            world_size=world_size,
        )


class PatchedQWenMLP(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['w1', 'w2']:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        for mod_name in ['c_proj']:
            rowwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs


class PatchedQWenModel(nn.Module):

    def _continuous_batching_forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of LlamaModel.forward."""
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # Attention mask is not necessary in continuous batching
        hidden_states = inputs_embeds

        context = self.context.context
        max_kv_seq_length = context.max_kv_seq_length
        # do not support use_dynamic_ntk
        ntk_alpha_list = [1.0]
        self.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
        rotary_pos_emb_list = [
            self.rotary_emb(max_kv_seq_length, ntk_alpha=ntk_alpha)
            for ntk_alpha in ntk_alpha_list
        ]
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                rotary_pos_emb_list=rotary_pos_emb_list,
            )
            hidden_states = outputs[0]

        hidden_states = self.ln_f(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
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
