# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from:
# https://huggingface.co/tiiuae/falcon-7b-instruct
# https://github.com/huggingface/transformers/blob/v4.33-release/src/transformers/models/falcon/modeling_falcon.py  # noqa

import logging
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint
from torch.distributed._tensor import DeviceMesh
from transformers.modeling_outputs import \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.models.falcon.modeling_falcon import build_alibi_tensor

from ..dist_utils import (colwise_parallelize_linear_fn,
                          rowwise_parallelize_linear_fn)
from ..kernels import (alibi_paged_attention_fwd, fill_kv_cache,
                       paged_attention_fwd)

logger = logging.getLogger()


# rotary pos emb helpers
# (torch.jit.script does not seem to support staticmethod...)
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

        self.cos_cached = emb.cos()[None, :, :]
        self.sin_cached = emb.sin()[None, :, :]

        self.cos_cached = self.cos_cached.type(dtype)
        self.sin_cached = self.sin_cached.type(dtype)

    def patched_cos_sin(self,
                        position_ids_1d: torch.Tensor,
                        device='cpu',
                        dtype=torch.bfloat16) -> torch.Tensor:
        total_length = int(position_ids_1d.max().item()) + 1

        if (self.seq_len_cached is None) or (total_length >
                                             self.seq_len_cached):
            self._patched_set_cos_sin_cache(total_length, device, dtype)
        # position_ids.shape == [1, packed_seq_len]
        # position_ids_1d.shape == [packed_seq_len]
        return (
            self.cos_cached[:, position_ids_1d, None, :],
            self.sin_cached[:, position_ids_1d, None, :],
        )

    def _contiguous_batching_forward(self, query: torch.Tensor,
                                     key: torch.Tensor,
                                     position_ids_1d: torch.Tensor):
        # batch, seq_len, *_ = query.shape
        cos, sin = self.patched_cos_sin(position_ids_1d,
                                        device=query.device,
                                        dtype=query.dtype)
        return (
            (query * cos) + (rotate_half(query) * sin),
            (key * cos) + (rotate_half(key) * sin),
        )

    def forward(self, query, key, position_ids_or_past_key_values_length=0):
        """forward."""
        return self._contiguous_batching_forward(
            query, key, position_ids_or_past_key_values_length)


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
                weight = mod.weight.reshape(
                    -1,  # num groups
                    (self.num_heads + self.num_kv_heads * 2) * self.head_dim,
                    self.hidden_size,
                )
            elif self.multi_query:
                # e.g. 7b-instruct, MQA
                # split to q, copy kv
                weight = mod.weight.reshape(
                    -1,
                    self.head_dim,
                    self.hidden_size,
                )
                q_weight = weight[:self.num_heads]
                k_weight = weight[self.num_heads:self.num_heads + 1]
                v_weight = weight[self.num_heads + 1:self.num_heads + 2]
                q_weight_shards = torch.tensor_split(q_weight,
                                                     world_size,
                                                     dim=0)
                weight_shards = []
                for q in q_weight_shards:
                    # only shard q heads but
                    # copy single k/v head to all ranks
                    weight_shards.append(q)
                    weight_shards.append(k_weight)
                    weight_shards.append(v_weight)
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
                weight = mod.weight.reshape(
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
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            if not dist.is_initialized():
                num_head = self.num_heads
            else:
                # this trick will, for example, split 11 into [4, 4, 3]
                # following the way column parallel linear splitting
                # non-dividable dims
                num_head = self.num_heads - dist.get_rank() - 1
                num_head = 1 + num_head // dist.get_world_size()
            fused_qkv = fused_qkv.view(batch_size, seq_length, num_head + 2,
                                       self.head_dim)
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[
                ..., [-1], :]

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
        q_start_loc = context.q_start_loc
        q_seq_length = context.seq_length
        history_lengths = q_seq_length.new_tensor(history_lengths)
        kv_seq_length = q_seq_length + history_lengths
        max_seq_len = q_seq_length.max().item()

        fused_qkv = self.query_key_value(
            hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        if isinstance(self.maybe_rotary, nn.Module):
            position_ids_1d = self.context.context.position_ids_1d
            query_layer, key_layer = self.maybe_rotary(query_layer, key_layer,
                                                       position_ids_1d)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, kv_length, head_dim]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]

            fill_kv_cache(key_layer.contiguous(),
                          value_layer.contiguous(),
                          past_key,
                          past_value,
                          q_start_loc,
                          q_seq_length,
                          block_offsets=context.block_offsets,
                          history_lengths=context.history_lengths,
                          context=context)

        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        attn_output = torch.empty_like(query_layer)
        block_offsets = context.block_offsets
        block_size = past_key.size(1)

        if alibi is None:
            paged_attention_fwd(q=query_layer,
                                k=past_key,
                                v=past_value,
                                o=attn_output,
                                block_offsets=block_offsets,
                                b_start_loc=q_start_loc,
                                b_seq_len=q_seq_length,
                                b_kv_seq_len=kv_seq_length,
                                max_input_len=max_seq_len)

        else:
            alibi_paged_attention_fwd(q=query_layer,
                                      k=past_key,
                                      v=past_value,
                                      o=attn_output,
                                      block_offsets=block_offsets,
                                      b_start_loc=q_start_loc,
                                      b_seq_len=q_seq_length,
                                      b_kv_seq_len=kv_seq_length,
                                      max_input_len=max_seq_len,
                                      alibi_scale=self.inv_norm_factor,
                                      BLOCK=block_size)

        attn_output = attn_output.reshape(batch_size, query_length, -1)

        output_tensor = self.dense(attn_output)

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
        position_ids = self.context.context.position_ids
        return self._contiguous_batching_forward(hidden_states, position_ids,
                                                 alibi, attention_mask,
                                                 layer_past, head_mask,
                                                 use_cache, output_attentions)


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

        # history_lengths = self.context.context.history_lengths

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # noqa
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache  # noqa
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # noqa
        use_alibi = getattr(self, 'use_alibi', getattr(self, 'alibi', False))

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'  # noqa
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')

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
        if use_alibi:
            alibi = build_alibi_tensor(attention_mask,
                                       self.num_heads,
                                       dtype=hidden_states.dtype)
        else:
            alibi = None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

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

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1], )

        # Add last hidden state

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

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
        return self._contiguous_batching_forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)


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
