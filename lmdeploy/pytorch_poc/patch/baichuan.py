# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from lmdeploy.pytorch_poc.dist_utils import (rowwise_parallelize_linear_fn,
                                             try_to_local)
from lmdeploy.pytorch_poc.patch.functional import \
    attention_forward_with_paged_attention

from .llama import apply_rotary_pos_emb


def _attention_partition_fn(mod_name: str, mod: nn.Module,
                            device_mesh: DeviceMesh):
    """A function for attention partition."""
    if mod_name in ['W_pack']:
        for name, param in mod.named_parameters():
            param = param.unflatten(0, (3, -1))
            dist_tensor = distribute_tensor(param, device_mesh, [Shard(1)])
            dist_tensor = try_to_local(dist_tensor)
            dist_tensor = dist_tensor.flatten(0, 1)
            dist_param = torch.nn.Parameter(dist_tensor)
            mod.register_parameter(name, dist_param)
    elif mod_name in ['o_proj']:
        rowwise_parallelize_linear_fn(mod,
                                      device_mesh=device_mesh,
                                      to_local=True)


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        return _attention_partition_fn(mod_name, mod, device_mesh)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

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
        """Rewrite of Attention.forward."""
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
        """Rewrite implementation of Attention.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
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

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        return _attention_partition_fn(mod_name, mod, device_mesh)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of BaichuanAttention.forward."""
        if self.context.use_origin:
            return self.origin_mod(
                hidden_states,
                attention_mask,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            world_size = 1
            if dist.is_initialized():
                world_size = dist.get_world_size()
            return self._contiguous_batching_forward(
                hidden_states,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                world_size=world_size,
            )

    def _contiguous_batching_forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of BaichuanAttention.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        assert not output_attentions
        context = self.context.context
        position_ids = context.position_ids
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


class BaichuanModel(nn.Module):

    def _continuous_batching_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of BaichuanModel.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        """Rewrite of BaichuanModel.forward."""
        use_origin = self.context.use_origin
        if use_origin:
            # use origin model
            return self.origin_mod(
                input_ids,
                attention_mask,
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
                past_key_values,
                inputs_embeds,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
