# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py   # noqa: E501

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..dist_utils import (colwise_parallelize_linear,
                          rowwise_parallelize_linear_fn, try_to_local)
from ..kernels import paged_attention_fwd
from .functional import fill_kv_cache


class PatchedRMSNorm(nn.Module):
    """Rewrite RMSNorm."""

    def forward(self, hidden_states):
        """forward."""
        # torch.nn.functional.normalize based implementation might leads
        # to wrong output
        from ..kernels import rms_norm
        ret = rms_norm(hidden_states.permute(1, 0, 2), self.weight, self.eps)
        return ret.permute(1, 0, 2)


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    tensor_list = tensor.chunk(num_partitions, dim=-1)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def apply_rotary_pos_emb(x: torch.Tensor,
                         rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, hn = x.size(0), x.size(-1)
    xslice = x[..., :hn // 2]
    rope_cache = rope_cache[:sq]
    xshaped = xslice.unflatten(-1, (-1, 2))
    rope_cache = rope_cache.unsqueeze(2)

    # inplace
    torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] -
            xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] +
            xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
        out=xshaped,
    )
    return x


class PatchedSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h] and returns output of
    the same size.
    """

    def _distribute_qkv_linear(self, mod: nn.Module, device_mesh: DeviceMesh):
        """distribute qkv linear."""
        sections = [
            self.num_attention_heads_per_partition *
            self.hidden_size_per_attention_head,
            self.num_multi_query_groups_per_partition *
            self.hidden_size_per_attention_head,
            self.num_multi_query_groups_per_partition *
            self.hidden_size_per_attention_head,
        ]
        for name, param in mod.named_parameters():
            splited_param = param.split(sections, dim=0)
            updated_param = []
            for p in splited_param:
                dist_tensor = distribute_tensor(p, device_mesh, [Shard(0)])
                dist_tensor = try_to_local(dist_tensor)
                updated_param.append(dist_tensor)
            param = torch.cat(updated_param)
            dist_param = torch.nn.Parameter(param)
            mod.register_parameter(name, dist_param)

    def _distribute_qkv_lora_linear(self, module: nn.Module,
                                    device_mesh: DeviceMesh):
        """distribute qkv lora linear."""
        to_local = True
        self._distribute_qkv_linear(
            module.base_layer,
            device_mesh=device_mesh,
        )
        for mod in module.lora_A.values():
            colwise_parallelize_linear(mod,
                                       device_mesh=device_mesh,
                                       to_local=to_local)
        for mod in module.lora_B.values():
            self._distribute_qkv_linear(
                mod,
                device_mesh=device_mesh,
            )
        module._tp_mode = 'colwise'

    def _distribute_partition_fn(self, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['query_key_value']:
            from peft.tuners.lora import Linear as LoraLinear
            if isinstance(mod, LoraLinear):
                self._distribute_qkv_lora_linear(mod, device_mesh)
            else:
                self._distribute_qkv_linear(mod, device_mesh)
        elif mod_name in ['dense']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor]] = None,
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

        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()

        context = self.context.context
        history_lengths = context.history_lengths
        max_seq_length = context.max_seq_length
        q_start_loc = context.q_start_loc
        q_seq_length = context.q_seq_length
        kv_seq_length = context.kv_seq_length
        block_offsets = context.block_offsets

        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition *
                    self.hidden_size_per_attention_head // world_size,
                    self.num_multi_query_groups_per_partition *
                    self.hidden_size_per_attention_head // world_size,
                    self.num_multi_query_groups_per_partition *
                    self.hidden_size_per_attention_head // world_size,
                ],
                dim=-1,
            )
            query_layer = query_layer.unflatten(
                -1, (-1, self.hidden_size_per_attention_head))
            key_layer = key_layer.unflatten(
                -1, (-1, self.hidden_size_per_attention_head))
            value_layer = value_layer.unflatten(
                -1, (-1, self.hidden_size_per_attention_head))
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition // world_size,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer,
             value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding)
        query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # [b, sq, np, hn]
        query_layer, key_layer, value_layer = [
            k.transpose(0, 1) for k in [query_layer, key_layer, value_layer]
        ]

        # adjust key and value for inference
        cache_k, cache_v = kv_cache
        fill_kv_cache(key_layer[0],
                      value_layer[0],
                      cache_k,
                      cache_v,
                      q_start_loc,
                      q_seq_length,
                      block_offsets=block_offsets,
                      history_lengths=history_lengths,
                      context=context)

        # ==================================
        # core attention computation
        # ==================================

        context_layer = query_layer
        paged_attention_fwd(query_layer,
                            cache_k,
                            cache_v,
                            context_layer,
                            block_offsets,
                            q_start_loc=q_start_loc,
                            q_seqlens=q_seq_length,
                            kv_seqlens=kv_seq_length,
                            max_seqlen=max_seq_length)

        context_layer = context_layer.transpose(1, 0).flatten(-2)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

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
        return self._contiguous_batching_forward(
            hidden_states,
            rotary_pos_emb,
            kv_cache,
        )


class MLP(nn.Module):

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['dense_h_to_4h']:
            for name, param in mod.named_parameters():
                dist_tensor = distribute_tensor(param.unflatten(0, (2, -1)),
                                                device_mesh, [Shard(1)])
                dist_tensor = try_to_local(dist_tensor)
                dist_param = torch.nn.Parameter(dist_tensor.flatten(0, 1))
                mod.register_parameter(name, dist_param)
        elif mod_name in ['dense_4h_to_h']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs


class PatchedChatGLMModel(nn.Module):

    def _contiguous_batching_forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                            ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None):
        output_hidden_states = False
        use_cache = True

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size,
                                                  device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            context = self.context.context
            position_ids_1d = context.position_ids_1d
            rotary_pos_emb = rotary_pos_emb[position_ids_1d[None]]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        (hidden_states, presents, all_hidden_states,
         all_self_attentions) = self.encoder(
             inputs_embeds,
             full_attention_mask,
             rotary_pos_emb=rotary_pos_emb,
             kv_caches=past_key_values,
             use_cache=use_cache,
             output_hidden_states=output_hidden_states)

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
        return self._contiguous_batching_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            full_attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
