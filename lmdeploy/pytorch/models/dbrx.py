# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, LayerNorm,
                                 RopeType, build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import build_qkv_proj, build_rowwise_linear
from lmdeploy.pytorch.nn.moe import FusedMoE, SoftmaxTopK
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight


class DbrxAttention(nn.Module):
    """Rewrite module of DbrxAttention."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        attn_config = config.attn_config
        quantization_config = getattr(config, 'quantization_config', None)
        hidden_size = config.d_model
        num_heads = config.n_heads
        num_key_value_heads = attn_config.kv_n_heads
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)

        # packed qkv
        self.Wqkv = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            num_heads,
            head_dim,
            num_kv_heads=num_key_value_heads,
        )

        # o_proj
        self.out_proj = build_rowwise_linear(hidden_size,
                                             hidden_size,
                                             bias=False,
                                             quant_config=quantization_config,
                                             dtype=dtype,
                                             device=device,
                                             is_tp=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of forward."""
        # qkv proj
        qkv_states = self.Wqkv(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.Wqkv.split_qkv(
            qkv_states)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )

        # attention
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

        # o proj
        attn_output = self.out_proj(attn_output)
        return attn_output


class DbrxRouter(nn.Module):
    """router."""

    def __init__(self,
                 hidden_size: int,
                 moe_num_experts: int,
                 moe_top_k: int,
                 moe_normalize_expert_weights: Optional[float],
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()

        self.layer = build_rowwise_linear(
            hidden_size,
            moe_num_experts,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
        )

        self.softmax_topk = SoftmaxTopK(moe_top_k)

        self.moe_normalize_expert_weights = moe_normalize_expert_weights

    def forward(self, hidden_states: torch.Tensor):
        """forward."""
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        weights = self.layer(hidden_states)
        top_weights, top_experts = self.softmax_topk(weights)

        top_weights_scale = (torch.norm(top_weights,
                                        p=self.moe_normalize_expert_weights,
                                        dim=-1,
                                        keepdim=True)
                             if self.moe_normalize_expert_weights is not None
                             else 1.0)
        top_weights = top_weights / top_weights_scale

        return top_weights, top_experts


class DbrxExperts(nn.Module):
    """experts."""

    def __init__(self,
                 hidden_size: int,
                 ffn_hidden_size: int,
                 moe_num_experts: int,
                 ffn_act_fn: dict,
                 moe_top_k: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()

        act_fn_name = ffn_act_fn.get('name', None)
        assert act_fn_name == 'silu'

        self.mlp = FusedMoE(
            hidden_size,
            ffn_hidden_size,
            moe_num_experts,
            top_k=moe_top_k,
            renormalize=False,
            dtype=dtype,
            device=device,
            all_reduce=True,
        )

    def forward(self, hidden_states: torch.Tensor, top_weights: torch.Tensor,
                top_experts: torch.Tensor):
        """forward."""
        batch_size = hidden_states.size(0)
        hidden_states = hidden_states.flatten(0, 1)
        out_states = self.mlp(
            hidden_states,
            top_weights,
            top_experts,
        )
        out_states = out_states.unflatten(0, (batch_size, -1))

        return out_states


class DbrxFFN(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        ffn_config = config.ffn_config
        self.router = DbrxRouter(
            hidden_size=config.d_model,
            moe_num_experts=ffn_config.moe_num_experts,
            moe_top_k=ffn_config.moe_top_k,
            moe_normalize_expert_weights=ffn_config.
            moe_normalize_expert_weights,
            dtype=dtype,
            device=device,
        )

        self.experts = DbrxExperts(
            hidden_size=config.d_model,
            ffn_hidden_size=ffn_config.ffn_hidden_size,
            moe_num_experts=ffn_config.moe_num_experts,
            ffn_act_fn=ffn_config.ffn_act_fn,
            moe_top_k=ffn_config.moe_top_k,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        """forward."""
        top_weights, top_experts = self.router(x)
        out = self.experts(x, top_weights, top_experts)
        return out


class DbrxNormAttentionNorm(nn.Module):
    """dbrx norm attention norm."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx

        self.norm_1 = LayerNorm(
            config.d_model,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.attn = DbrxAttention(
            config=config,
            dtype=dtype,
            device=device,
        )
        self.norm_2 = LayerNorm(
            config.d_model,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        residual_states: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):
        """forward."""
        if residual_states is None:
            residual_states = hidden_states
            hidden_states = self.norm_1(hidden_states)
        else:
            hidden_states, residual_states = self.norm_1(
                hidden_states, residual_states)

        hidden_states = self.attn(
            hidden_states,
            rotary_pos_emb,
            past_key_value,
            attn_metadata,
        )

        hidden_states, residual_states = self.norm_2(hidden_states,
                                                     residual_states)
        return hidden_states, residual_states


class DbrxBlock(nn.Module):
    """decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx

        # build attention layer
        self.norm_attn_norm = DbrxNormAttentionNorm(config,
                                                    layer_idx,
                                                    dtype=dtype,
                                                    device=device)

        # builf MLP
        self.ffn = DbrxFFN(config, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):
        # Self Attention
        hidden_states, residual = self.norm_attn_norm(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            residual_states=residual,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states = self.ffn(hidden_states)

        return hidden_states, residual


class DbrxModel(nn.Module):
    """model."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size,
                                config.d_model,
                                self.padding_idx,
                                dtype=dtype,
                                device=device)

        # build all decode layers
        self.blocks = nn.ModuleList([
            DbrxBlock(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.n_layers)
        ])

        # build norm
        self.norm_f = LayerNorm(config.d_model,
                                bias=False,
                                dtype=dtype,
                                device=device)

        # build rotary embedding
        emb_type = RopeType.LinearScaling
        rope_dim = config.d_model // config.n_heads
        rope_max_pos_emb = config.max_seq_len
        rope_base = config.attn_config.rope_theta
        self.rotary_emb = build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            emb_type=emb_type,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Rewrite of LlamaModel.forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.blocks):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        # norm
        hidden_states, _ = self.norm_f(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.wte


class DbrxForCausalLM(nn.Module):
    """ModelForCausalLM."""

    support_cuda_graph = True

    packed_modules_mapping = {
        'qkv_proj': [
            'q_proj',
            'k_proj',
            'v_proj',
        ],
        'gate_up_proj': [
            'gate_proj',
            'up_proj',
        ],
    }

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # build model
        self.transformer = DbrxModel(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """model forward, return logits."""
        hidden_states = self.transformer(
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
        return self.transformer.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        # modify from vllm
        config = self.config

        ffn_config = config.ffn_config
        num_experts = ffn_config.moe_num_experts

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name
                    or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue

            if '.experts' in name:
                loaded_weight = loaded_weight.unflatten(0, (num_experts, -1))
                if '.w1' in name:
                    name = name.replace('.w1', '.gate_up_weights')
                    param = params_dict[name]
                    for exp_id in range(num_experts):
                        weight = loaded_weight[exp_id]
                        load_weight(param,
                                    weight,
                                    expert_id=exp_id,
                                    shard_id='gate')
                elif '.v1' in name:
                    name = name.replace('.v1', '.gate_up_weights')
                    param = params_dict[name]
                    for exp_id in range(num_experts):
                        weight = loaded_weight[exp_id]
                        load_weight(param,
                                    weight,
                                    expert_id=exp_id,
                                    shard_id='up')
                elif '.w2' in name:
                    name = name.replace('.w2', '.down_weights')
                    param = params_dict[name]
                    for exp_id in range(num_experts):
                        weight = loaded_weight[exp_id].t()
                        load_weight(param,
                                    weight,
                                    expert_id=exp_id,
                                    shard_id='down')
            elif '.Wqkv' in name:
                param = params_dict[name]
                q, k, v = param.weight_spliter(loaded_weight)
                load_weight(param, q, shard_id='q')
                load_weight(param, k, shard_id='k')
                load_weight(param, v, shard_id='v')
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)
