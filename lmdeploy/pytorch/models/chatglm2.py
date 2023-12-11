# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py   # noqa: E501

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..dist_utils import rowwise_parallelize_linear_fn, try_to_local
from ..kernels import paged_attention_fwd
from .functional import fill_kv_cache


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
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor,
                         rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, _, np, _ = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] -
            xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] +
            xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


class PatchedSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h] and returns output of
    the same size.
    """

    def _distribute_partition_fn(self, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['query_key_value']:
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
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
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
        origin_self = self

        context = self.context.context
        history_lengths = context.history_lengths
        mixed_x_layer = origin_self.query_key_value(hidden_states)

        if origin_self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    origin_self.num_attention_heads_per_partition *
                    origin_self.hidden_size_per_attention_head // world_size,
                    origin_self.num_multi_query_groups_per_partition *
                    origin_self.hidden_size_per_attention_head // world_size,
                    origin_self.num_multi_query_groups_per_partition *
                    origin_self.hidden_size_per_attention_head // world_size,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(query_layer.size()[:-1] + (
                origin_self.num_attention_heads_per_partition // world_size,
                origin_self.hidden_size_per_attention_head,
            ))
            key_layer = key_layer.view(key_layer.size()[:-1] + (
                origin_self.num_multi_query_groups_per_partition // world_size,
                origin_self.hidden_size_per_attention_head,
            ))
            value_layer = value_layer.view(value_layer.size()[:-1] + (
                origin_self.num_multi_query_groups_per_partition // world_size,
                origin_self.hidden_size_per_attention_head,
            ))
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                origin_self.num_attention_heads_per_partition // world_size,
                3 * origin_self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer,
             value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # [b, sq, np, hn]
        query_layer, key_layer, value_layer = [
            k.transpose(0, 1) for k in [query_layer, key_layer, value_layer]
        ]

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            q_start_loc = context.q_start_loc
            q_seq_length = context.q_seq_length

            q_start_loc: torch.Tensor
            history_lengths = q_seq_length.new_tensor(history_lengths)
            kv_seq_length = q_seq_length + history_lengths
            max_seq_len = q_seq_length.max().item()
            fill_kv_cache(key_layer[0],
                          value_layer[0],
                          cache_k,
                          cache_v,
                          q_start_loc,
                          q_seq_length,
                          block_offsets=context.block_offsets,
                          history_lengths=history_lengths)

        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        # ==================================
        # core attention computation
        # ==================================

        context_layer = torch.empty_like(query_layer)

        block_offsets = context.block_offsets
        block_size = cache_k.size(1)

        paged_attention_fwd(query_layer,
                            cache_k,
                            cache_v,
                            context_layer,
                            block_offsets,
                            b_start_loc=q_start_loc,
                            b_seq_len=q_seq_length,
                            b_kv_seq_len=kv_seq_length,
                            max_input_len=max_seq_len,
                            BLOCK=block_size)

        context_layer = context_layer.transpose(1, 0).flatten(-2)

        # =================
        # Output. [sq, b, h]
        # =================

        output = origin_self.dense(context_layer)

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
            attention_mask,
            rotary_pos_emb,
            kv_cache,
            use_cache,
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
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None):
        orig_self = self.origin_mod
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                orig_self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else orig_self.config.use_cache  # noqa: E501
        return_dict = return_dict if return_dict is not None else orig_self.config.use_return_dict  # noqa: E501

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = orig_self.embedding(input_ids)

        if orig_self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = orig_self.get_prompt(
                    batch_size=batch_size,
                    device=input_ids.device,
                    dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask.new_ones(
                        (batch_size, orig_self.pre_seq_len)), attention_mask
                ],
                                           dim=-1)

        # Rotary positional embeddings
        rotary_pos_emb = orig_self.rotary_pos_emb(orig_self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        (hidden_states, presents, all_hidden_states,
         all_self_attentions) = orig_self.encoder(
             inputs_embeds,
             full_attention_mask,
             rotary_pos_emb=rotary_pos_emb,
             kv_caches=past_key_values,
             use_cache=use_cache,
             output_hidden_states=output_hidden_states)

        if not return_dict:
            return tuple(v for v in [
                hidden_states, presents, all_hidden_states, all_self_attentions
            ] if v is not None)

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
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
