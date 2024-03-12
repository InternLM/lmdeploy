# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from:
# https://huggingface.co/tiiuae/falcon-7b-instruct
# https://github.com/huggingface/transformers/blob/v4.33-release/src/transformers/models/falcon/modeling_falcon.py  # noqa

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint
from torch.distributed._tensor import DeviceMesh
from transformers.modeling_outputs import \
    BaseModelOutputWithPastAndCrossAttentions

from ..dist_utils import (colwise_parallelize_linear_fn,
                          rowwise_parallelize_linear_fn)
from ..kernels import (alibi_paged_attention_fwd, apply_rotary_pos_emb,
                       fill_kv_cache, fused_rotary_emb, paged_attention_fwd)


class PatchedFalconAttention(nn.Module):

    # @classmethod
    def _distribute_partition_fn(self, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""

        world_size = dist.get_world_size()

        if mod_name in ['query_key_value']:
            if self.new_decoder_architecture:
                # e.g. 40b-instruct, GQA
                # split qkv across groups
                # no finer-grained partitioning
                mod.weight.data = mod.weight.reshape(
                    -1,  # num groups
                    (self.num_heads + self.num_kv_heads * 2) * self.head_dim,
                    self.hidden_size,
                )
            elif self.multi_query:
                # e.g. 7b-instruct, MQA
                # split to q, copy kv
                weight = mod.weight.unflatten(0, (-1, self.head_dim))
                q_weight = weight[:self.num_heads]
                kv_weight = weight[-2:]
                q_weight_shards = q_weight.chunk(world_size, 0)
                weight_shards = []
                for q in q_weight_shards:
                    # only shard q heads but
                    # copy single k/v head to all ranks
                    weight_shards.append(q)
                    weight_shards.append(kv_weight)
                mod.weight.data = torch.cat(weight_shards, dim=0)
                # here we keep the weight to be 3D,
                # so that column parallel will split it
                # into integer-numbered heads

            # no bias for 7b-instruct and 40b-instruct
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

            if self.new_decoder_architecture or self.multi_query:
                # return to 2D for later matmul
                mod.weight.data = mod.weight.data.reshape(-1, self.hidden_size)

        elif mod_name in ['dense']:
            if self.new_decoder_architecture:
                # e.g. 40b-instruct, GQA
                mod.weight.data = mod.weight.reshape(
                    self.hidden_size,
                    -1,  # num groups
                    self.num_heads * self.head_dim,
                )
            elif self.multi_query:
                # e.g. 7b-instruct, MQA
                mod.weight.data = mod.weight.reshape(self.hidden_size, -1,
                                                     self.head_dim)

            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

            if self.new_decoder_architecture or self.multi_query:
                mod.weight.data = mod.weight.reshape(self.hidden_size, -1)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _split_heads(
        self, fused_qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the last dimension into (num_heads, head_dim), results share
        same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*):
                [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim]
            key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if self.new_decoder_architecture:
            # e.g. 40b-instruct model
            batch, seq_len, _ = fused_qkv.shape
            qkv = fused_qkv.view(batch, seq_len, -1,
                                 self.num_heads // self.num_kv_heads + 2,
                                 self.head_dim)
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            # because cache_engine & kernel
            # already handled grouped attention
            # removing broadcast make it faster and more memory-saving
            # key = torch.broadcast_to(key, query.shape)
            # value = torch.broadcast_to(value, query.shape)

            query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
            return query, key, value
        elif not self.multi_query:
            # e.g. rw-1b model
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length,
                                       self.num_heads // dist.get_world_size(),
                                       3, self.head_dim)
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[...,
                                                                         2, :]
        else:
            # e.g. 7b-instruct model
            fused_qkv = fused_qkv.unflatten(-1, (-1, self.head_dim))
            split_shape = (fused_qkv.size(-2) - 2, 1, 1)
            return fused_qkv.split(split_shape, dim=-2)

    def _contiguous_batching_forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ):
        # prepare inputs for continuous batch forwarding
        context = self.context.context
        q_start_loc = context.q_start_loc
        q_seq_length = context.q_seq_length
        kv_seq_length = context.kv_seq_length
        max_q_seq_length = context.max_q_seq_length
        block_offsets = context.block_offsets
        position_ids_1d = context.position_ids_1d
        max_kv_seq_length = context.max_kv_seq_length

        def __maybe_rotary_fn(query_states, key_states, value_states):
            scaling_factor = 1.0
            inv_freq = self.maybe_rotary.inv_freq
            query_states, key_states = fused_rotary_emb(
                query_states[None],
                key_states[None],
                position_ids_1d[None],
                inv_freq=inv_freq,
                scaling_factor=scaling_factor,
                out_q=query_states[None],
                out_k=key_states[None])
            return query_states[0], key_states[0], value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            """rotary embedding func."""
            cos, sin = self.rotary_emb(value_states.transpose(0, 1),
                                       max_kv_seq_length)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, context.position_ids,
                position_ids_1d)
            return query_states, key_states, value_states

        fused_qkv = self.query_key_value(hidden_states)

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        query_layer = query_layer.flatten(0, 1)
        key_layer = key_layer.flatten(0, 1)
        value_layer = value_layer.flatten(0, 1)
        if hasattr(self, 'maybe_rotary'):
            query_layer, key_layer, value_layer = __maybe_rotary_fn(
                query_layer, key_layer, value_layer)
        elif hasattr(self, 'rotary_emb'):
            query_layer, key_layer, value_layer = __rotary_emb_fn(
                query_layer, key_layer, value_layer)

        past_key, past_value = layer_past
        fill_kv_cache(
            key_layer.contiguous(),
            value_layer.contiguous(),
            past_key,
            past_value,
            q_start_loc,
            q_seq_length,
            kv_seq_length=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            block_offsets=block_offsets,
        )

        attn_output = query_layer

        if not alibi:
            paged_attention_fwd(q=query_layer,
                                k=past_key,
                                v=past_value,
                                o=attn_output,
                                block_offsets=block_offsets,
                                q_start_loc=q_start_loc,
                                q_seqlens=q_seq_length,
                                kv_seqlens=kv_seq_length,
                                max_seqlen=max_q_seq_length)

        else:
            num_heads_full = self.num_heads
            head_offset = 0
            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                head_offset = self.num_heads // world_size * rank
            alibi_paged_attention_fwd(q=query_layer,
                                      k=past_key,
                                      v=past_value,
                                      o=attn_output,
                                      block_offsets=block_offsets,
                                      b_start_loc=q_start_loc,
                                      b_seq_len=q_seq_length,
                                      b_kv_seq_len=kv_seq_length,
                                      max_input_len=max_q_seq_length,
                                      head_offset=head_offset,
                                      num_heads=num_heads_full,
                                      alibi_scale=self.inv_norm_factor)

        attn_output = attn_output[None].flatten(-2, -1)
        output_tensor = self.dense(attn_output)

        if output_attentions:
            return output_tensor, layer_past, None
        else:
            return output_tensor, layer_past

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
        return self._contiguous_batching_forward(hidden_states, alibi,
                                                 layer_past)


class PatchedFalconMLP(nn.Module):

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['dense_h_to_4h']:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['dense_4h_to_h']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs


class PatchedFalconModel(nn.Module):

    def _contiguous_batching_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                        ...]] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...],
               BaseModelOutputWithPastAndCrossAttentions]:

        output_attentions = False
        use_cache = True
        use_alibi = getattr(self, 'use_alibi', getattr(self, 'alibi', False))

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        head_mask = self.get_head_mask(head_mask,
                                       self.config.num_hidden_layers)

        hidden_states = inputs_embeds

        # Compute alibi tensor: check build_alibi_tensor documentation

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=None,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=use_alibi,
            )
            hidden_states = outputs[0]

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        return self._contiguous_batching_forward(
            input_ids=input_ids, past_key_values=past_key_values)


class PatchedFalconForCausalLM(nn.Module):

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
        use_origin: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...],
               BaseModelOutputWithPastAndCrossAttentions]:
        """Forward function, patched to ignore position_ids."""

        outputs = self.origin_mod(input_ids=input_ids,
                                  past_key_values=past_key_values,
                                  attention_mask=attention_mask,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict)
        return outputs
