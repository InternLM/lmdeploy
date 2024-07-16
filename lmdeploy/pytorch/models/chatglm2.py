# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py   # noqa: E501

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)

from ..kernels import fill_kv_cache, paged_attention_fwd
from ..weight_loader.dist_utils import (colwise_split_parallelize_linear,
                                        rowwise_parallelize_linear)


class PatchedRMSNorm(nn.Module):
    """Rewrite RMSNorm."""

    def forward(self, hidden_states):
        """forward."""
        # torch.nn.functional.normalize based implementation might leads
        # to wrong output
        from ..kernels import rms_norm

        ret = rms_norm(hidden_states, self.weight, self.eps)
        return ret


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
    # x: [b, sq, np, hn]
    # rope_cache: [b, sq, dim/4, 2]
    sq, hn = x.size(1), x.size(-1)
    xslice = x[..., :hn // 2]
    rope_cache = rope_cache[:, :sq]
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

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        sections = [
            self.num_attention_heads_per_partition *
            self.hidden_size_per_attention_head,
            self.num_multi_query_groups_per_partition *
            self.hidden_size_per_attention_head,
            self.num_multi_query_groups_per_partition *
            self.hidden_size_per_attention_head,
        ]
        colwise_split_parallelize_linear(self.query_key_value,
                                         sections,
                                         loader,
                                         rank=rank,
                                         world_size=world_size,
                                         prefix='query_key_value')
        rowwise_parallelize_linear(self.dense,
                                   loader,
                                   rank=rank,
                                   world_size=world_size,
                                   prefix='dense')

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
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
        # hidden_states: [b, sq, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [b, sq, h] --> [b, sq, (np * 3 * hn)]

        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()

        context = self.context.context
        max_q_seq_length = context.max_q_seq_length
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

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_layer, key_layer,
             value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding)
        query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        cache_k, cache_v = kv_cache
        fill_kv_cache(
            key_layer[0],
            value_layer[0],
            cache_k,
            cache_v,
            q_start_loc,
            q_seq_length,
            kv_seq_length=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            block_offsets=block_offsets,
        )

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
                            max_seqlen=max_q_seq_length)

        context_layer = context_layer.flatten(-2)

        # =================
        # Output. [b, sq, h]
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

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        w_pack_out = self.dense_h_to_4h.out_features
        sections = [w_pack_out // 2] * 2
        colwise_split_parallelize_linear(self.dense_h_to_4h,
                                         sections,
                                         loader,
                                         rank=rank,
                                         world_size=world_size,
                                         prefix='dense_h_to_4h')
        rowwise_parallelize_linear(self.dense_4h_to_h,
                                   loader,
                                   rank=rank,
                                   world_size=world_size,
                                   prefix='dense_4h_to_h')

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
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
        assert input_ids is not None
        context = self.context.context
        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        output_hidden_states = False
        use_cache = True

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            # multi-modality
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        hidden_states = inputs_embeds

        if getattr(self, 'pre_seq_len', None) is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size,
                                                  device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        # glm-4v
        if getattr(self, 'vision', None) is not None:
            from .cogvlm import _get_cogvlm_position_ids
            position_ids_1d = _get_cogvlm_position_ids(context)
        else:
            position_ids_1d = context.position_ids_1d

        rotary_pos_emb = rotary_pos_emb[position_ids_1d[None]]

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

    def forward(self,
                input_ids,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.BoolTensor] = None,
                full_attention_mask: Optional[torch.BoolTensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor,
                                                      torch.Tensor],
                                                ...]] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                use_cache: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs):
        return self._contiguous_batching_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            full_attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )


class PatchedEmbedding(nn.Module):

    def forward(self, input_ids):
        """Rewrite to not transpose hidden_statens for all models."""
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # If the input flag for fp32 residual connection is set,
        # convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class PatchedChatGLMForConditionalGeneration(nn.Module):

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                return_last_logit: Optional[bool] = False,
                **kwargs):
        """rewrite to not transpose logits for all models."""
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.transformer.output_layer(hidden_states)

        loss = None

        if not return_dict:
            output = (lm_logits, ) + transformer_outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
