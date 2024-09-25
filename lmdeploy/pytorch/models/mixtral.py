# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RMSNorm, RopeType,
                                 build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import build_qkv_proj, build_rowwise_linear
from lmdeploy.pytorch.nn.moe import FusedMoE, SoftmaxTopK
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


class MixtralAttention(nn.Module):
    """mixtral attention."""

    def __init__(self,
                 config: Any,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = None

        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads

        # qkv
        self.qkv_proj = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.window_size = config.sliding_window or -1
        self.attn_fwd = Attention(
            num_heads,
            head_dim,
            num_kv_heads=num_key_value_heads,
            v_head_size=head_dim,
            sliding_window=self.window_size,
        )

        # o_proj
        self.o_proj = build_rowwise_linear(num_heads * head_dim,
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
        """Rewrite of LlamaAttention.forward."""
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(
            qkv_states)

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


class MixtralSparseMoeBlock(nn.Module):
    """mixtral sparse moe block."""

    def __init__(self,
                 config: Any,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        self.gate = build_rowwise_linear(
            self.hidden_dim,
            self.num_experts,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
        )

        self.softmax_topk = SoftmaxTopK(self.top_k)

        self.experts = FusedMoE(
            self.hidden_dim,
            self.ffn_dim,
            self.num_experts,
            top_k=self.top_k,
            renormalize=True,
            dtype=dtype,
            device=device,
            all_reduce=True,
        )

    def forward(self, hidden_states: torch.Tensor):
        """forward."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        topk_weights, topk_ids = self.softmax_topk(router_logits)
        out_states = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
        )

        out_states = out_states.reshape(batch_size, sequence_length, -1)
        return out_states, router_logits


class MixtralDecoderLayer(nn.Module):
    """mixtral decoder layer."""

    def __init__(self,
                 config: Any,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = None

        # build attention layer
        self.self_attn = MixtralAttention(config, dtype=dtype, device=device)
        self.block_sparse_moe = MixtralSparseMoeBlock(config,
                                                      dtype=dtype,
                                                      device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):

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

    def __init__(self,
                 config: Any,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            quant_config=None,
                            dtype=dtype,
                            device=device)

        emb_type = RopeType.LinearScaling
        rope_dim = config.hidden_size // config.num_attention_heads
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        scaling_factor = 1.0
        self.rotary_emb = build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            scaling_factor,
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
        """forward."""
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


class MixtralForCausalLM(nn.Module, CudaGraphMixin):
    """mixture model for causalLM."""

    def __init__(self,
                 config: Any,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.model = MixtralModel(config, dtype=dtype, device=device)
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
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.lm_head(hidden_states)

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        """support cudagraph."""
        seq_lens = input_ids.size(1)
        if seq_lens <= 512:
            return True
        return False

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
        ]

        num_experts = self.config.num_local_experts
        expert_params_mapping = []
        for exp_id in range(num_experts):
            gate_param = ('.experts.gate_up_weights',
                          f'.experts.{exp_id}.w1.weight', exp_id, 'gate')
            up_param = ('.experts.gate_up_weights',
                        f'.experts.{exp_id}.w3.weight', exp_id, 'up')
            down_param = ('.experts.down_weights',
                          f'.experts.{exp_id}.w2.weight', exp_id, 'down')
            expert_params_mapping += [gate_param, up_param, down_param]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name
                    or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            for (param_name, weight_name, expert_id,
                 shard_id) in expert_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param,
                            loaded_weight,
                            expert_id=expert_id,
                            shard_id=shard_id)
                break
            else:
                for (param_name, weight_name,
                     shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    load_weight(param, loaded_weight, shard_id=shard_id)
                    break
                else:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
