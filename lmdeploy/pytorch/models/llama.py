# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from lmdeploy.pytorch.layers import (ApplyRotaryEmb, Attention, EmbeddingType,
                                     RMSNorm, SiluAndMul, build_linear,
                                     build_merged_linear,
                                     build_rotary_embedding)
from lmdeploy.pytorch.model_inputs import StepContextManager

from ..weight_loader.dist_utils import (colwise_parallelize_linear,
                                        rowwise_parallelize_linear)


class LlamaAttention(nn.Module):
    """Rewrite module of LlamaAttention."""

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        self.ctx_mgr = ctx_mgr
        self.num_heads = origin.num_heads // world_size
        self.num_kv_heads = origin.num_key_value_heads // world_size
        self.head_dim = origin.head_dim

        # qkv
        self.qkv_proj = build_merged_linear(origin.q_proj,
                                            origin.k_proj,
                                            origin.v_proj,
                                            ctx_mgr=ctx_mgr,
                                            free_origin=True)

        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_kv_heads,
            v_head_size=self.head_dim,
        )

        self.o_proj = build_linear(origin.o_proj,
                                   ctx_mgr=ctx_mgr,
                                   all_reduce=world_size > 1)

    @staticmethod
    def _load_weights(mod, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['q_proj', 'k_proj', 'v_proj']:
            colwise_parallelize_linear(getattr(mod, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        rowwise_parallelize_linear(mod.o_proj,
                                   loader,
                                   rank=rank,
                                   world_size=world_size,
                                   prefix='o_proj')

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of LlamaAttention.forward."""
        context = self.ctx_mgr.current_context()

        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        qkv_states = qkv_states.unflatten(-1, (-1, self.head_dim))
        query_states, key_states, value_states = qkv_states.split((
            self.num_heads,
            self.num_kv_heads,
            self.num_kv_heads,
        ),
                                                                  dim=1)

        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            context.attn_meta,
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaMLP(nn.Module):

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        # gate up
        self.gate_up_proj = build_merged_linear(
            origin.gate_proj,
            origin.up_proj,
            ctx_mgr=ctx_mgr,
            free_origin=True,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_linear(origin.down_proj,
                                      ctx_mgr=ctx_mgr,
                                      all_reduce=True)

    @staticmethod
    def _load_weights(mod: nn.Module, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['gate_proj', 'up_proj']:
            colwise_parallelize_linear(getattr(mod, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        rowwise_parallelize_linear(mod.down_proj,
                                   loader,
                                   rank=rank,
                                   world_size=world_size,
                                   prefix='down_proj')

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class LlamaDecoderLayer(nn.Module):

    def __init__(self, origin: nn.Module, layer_idx: int,
                 ctx_mgr: StepContextManager):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = LlamaAttention(origin.self_attn, ctx_mgr)
        self.mlp = LlamaMLP(origin.mlp, ctx_mgr)

        # norm
        input_layernorm = origin.input_layernorm
        self.input_layernorm = RMSNorm(input_layernorm.weight,
                                       input_layernorm.variance_epsilon)
        post_attention_layernorm = origin.post_attention_layernorm
        self.post_attention_layernorm = RMSNorm(
            post_attention_layernorm.weight,
            post_attention_layernorm.variance_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class LlamaModel(nn.Module):

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        self.ctx_mgr = ctx_mgr
        self.embed_tokens = origin.embed_tokens
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(layer, idx, ctx_mgr)
            for idx, layer in enumerate(origin.layers)
        ])
        norm = origin.norm
        self.norm = RMSNorm(norm.weight, norm.variance_epsilon)

        rotary_emb = origin.layers[0].self_attn.rotary_emb
        rotary_name = type(rotary_emb).__name__
        if rotary_name in [
                'LlamaRotaryEmbedding', 'LlamaLinearScalingRotaryEmbedding'
        ]:
            emb_type = EmbeddingType.LinearScaling
        elif rotary_name == 'LlamaDynamicNTKScalingRotaryEmbedding':
            emb_type = EmbeddingType.DynamicNTKScaling
        self.rotary_emb = build_rotary_embedding(
            rotary_emb.dim,
            rotary_emb.max_position_embeddings,
            rotary_emb.base,
            rotary_emb.scaling_factor,
            emb_type,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite of LlamaModel.forward."""
        context = self.ctx_mgr.current_context()
        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        hidden_states = inputs_embeds
        residual = None
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)
        for idx, decoder_layer in enumerate(self.layers):

            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class LlamaForCausalLM(nn.Module):

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        self.ctx_mgr = ctx_mgr
        self.model = LlamaModel(origin.model, ctx_mgr)
        self.lm_head = build_linear(origin.lm_head)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return logits
