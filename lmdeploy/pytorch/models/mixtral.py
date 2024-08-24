# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, EmbeddingType,
                                 RMSNorm, build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_colwise_linear,
                                        build_merged_colwise_linear,
                                        build_rowwise_linear)
from lmdeploy.pytorch.nn.moe import SoftmaxTopK, build_moe_from_mlp
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from ..weight_loader.dist_utils import (colwise_parallelize_linear,
                                        rowwise_parallelize_linear)


class MixtralAttention(nn.Module):
    """mixtral attention."""

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        is_tp = world_size > 1
        self.ctx_mgr = ctx_mgr
        self.num_heads = origin.num_heads // world_size
        self.num_kv_heads = origin.num_key_value_heads // world_size
        self.head_dim = origin.head_dim

        # qkv
        self.qkv_proj = build_merged_colwise_linear(
            origin.q_proj,
            origin.k_proj,
            origin.v_proj,
            ctx_mgr=ctx_mgr,
            is_tp=is_tp,
        )
        del origin.q_proj, origin.k_proj, origin.v_proj

        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.window_size = origin.config.sliding_window or -1
        self.attn_fwd = Attention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_kv_heads,
            v_head_size=self.head_dim,
            sliding_window=self.window_size,
        )

        self.o_proj = build_rowwise_linear(
            origin.o_proj,
            ctx_mgr=ctx_mgr,
            is_tp=is_tp,
        )

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
        attn_metadata: Any = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of LlamaAttention.forward."""
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        qkv_states = qkv_states.unflatten(-1, (-1, self.head_dim))
        query_states, key_states, value_states = qkv_states.split(
            (
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
            ),
            dim=1,
        )

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
            attn_metadata,
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


class MixtralBLockSparseTop2MLP(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['w1', 'w3']:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        rowwise_parallelize_linear(self.w2,
                                   loader,
                                   rank=rank,
                                   world_size=world_size,
                                   prefix='w2')


class MixtralSparseMoeBlock(nn.Module):
    """mixtral sparse moe block."""

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        is_tp = world_size > 1
        self.is_tp = is_tp
        self.top_k = origin.top_k
        self.gate = build_colwise_linear(
            origin.gate,
            ctx_mgr=ctx_mgr,
            is_tp=is_tp,
        )
        self.softmax_topk = SoftmaxTopK(self.top_k)

        gates = [exp.w1 for exp in origin.experts]
        ups = [exp.w3 for exp in origin.experts]
        downs = [exp.w2 for exp in origin.experts]
        self.fused_moe = build_moe_from_mlp(gates,
                                            ups,
                                            downs,
                                            top_k=self.top_k,
                                            renormalize=True)

    def forward(self, hidden_states: torch.Tensor):
        """forward."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        topk_weights, topk_ids = self.softmax_topk(router_logits)
        out_states = self.fused_moe(
            hidden_states,
            topk_weights,
            topk_ids,
        )

        out_states = out_states.reshape(batch_size, sequence_length, -1)

        if self.is_tp:
            dist.all_reduce(out_states)
        return out_states, router_logits


class MixtralDecoderLayer(nn.Module):
    """mixtral decoder layer."""

    def __init__(self, origin: nn.Module, layer_idx: int,
                 ctx_mgr: StepContextManager):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = MixtralAttention(origin.self_attn, ctx_mgr)
        self.block_sparse_moe = MixtralSparseMoeBlock(origin.block_sparse_moe,
                                                      ctx_mgr)

        # norm
        input_layernorm = origin.input_layernorm
        is_w8a8 = hasattr(input_layernorm, 'from_float')
        self.input_layernorm = RMSNorm(
            input_layernorm.weight.size(0),
            input_layernorm.variance_epsilon,
            dtype=input_layernorm.weight.dtype,
            device=input_layernorm.weight.device,
            is_w8a8=is_w8a8,
        )
        load_weight(self.input_layernorm.weight, input_layernorm.weight)
        post_attention_layernorm = origin.post_attention_layernorm
        is_w8a8 = hasattr(post_attention_layernorm, 'from_float')
        self.post_attention_layernorm = RMSNorm(
            post_attention_layernorm.weight.size(0),
            post_attention_layernorm.variance_epsilon,
            dtype=post_attention_layernorm.weight.dtype,
            device=post_attention_layernorm.weight.device,
            is_w8a8=is_w8a8,
        )
        load_weight(self.post_attention_layernorm.weight,
                    post_attention_layernorm.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
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
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states, _ = self.block_sparse_moe(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class MixtralModel(nn.Module):
    """mixtral model."""

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        self.ctx_mgr = ctx_mgr
        self.embed_tokens = origin.embed_tokens
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(layer, idx, ctx_mgr)
            for idx, layer in enumerate(origin.layers)
        ])
        norm = origin.norm
        is_w8a8 = hasattr(norm, 'from_float')
        self.norm = RMSNorm(norm.weight.size(0),
                            norm.variance_epsilon,
                            dtype=norm.weight.dtype,
                            device=norm.weight.device,
                            is_w8a8=is_w8a8)
        load_weight(self.norm.weight, norm.weight)

        emb_type = EmbeddingType.LinearScaling
        config = origin.config
        rope_dim = config.hidden_size // config.num_attention_heads
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        scaling_factor = 1.0
        self.rotary_emb = build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            scaling_factor,
            emb_type,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite of LlamaModel.forward."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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
                attn_metadata=attn_metadata,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.embed_tokens


class MixtralForCausalLM(nn.Module):
    """mixture model for causalLM."""

    support_cuda_graph = True

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        self.ctx_mgr = ctx_mgr
        self.model = MixtralModel(origin.model, ctx_mgr)
        self.lm_head = build_rowwise_linear(origin.lm_head)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
