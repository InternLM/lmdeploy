# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.models.llama4 import Llama4Config, Llama4TextConfig

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import ApplyRotaryEmb, Attention, RMSNorm, RopeType, SiluAndMul, build_rotary_embedding
from lmdeploy.pytorch.nn.linear import build_merged_colwise_linear, build_qkv_proj, build_rowwise_linear
from lmdeploy.pytorch.nn.moe import build_fused_moe
from lmdeploy.pytorch.nn.rotary_embedding import Llama3Parameters
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


class Llama4TextAttention(nn.Module):
    """attention."""

    def __init__(self,
                 config: Llama4TextConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)

        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_rope = int((layer_idx + 1) % 4 != 0)  # rope unused for dense layers
        self.attn_bias = config.attention_bias

        # qkv
        self.qkv_proj = build_qkv_proj(
            config.hidden_size,
            num_q_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            head_size=self.head_dim,
            bias=self.attn_bias,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        self.attn_fwd = Attention(
            self.num_attention_heads,
            self.head_dim,
            num_kv_heads=self.num_key_value_heads,
            v_head_size=self.head_dim,
        )

        # o_proj
        self.o_proj = build_rowwise_linear(config.num_attention_heads * self.head_dim,
                                           config.hidden_size,
                                           bias=self.attn_bias,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """forward."""
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)

        if self.use_rope:
            cos, sin = rotary_pos_emb
            # TODO: fuse apply rotary pos emb
            query_states = query_states.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
            key_states = key_states.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
            query_states, key_states = self.apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )
            query_states = query_states.unflatten(-1, (2, -1)).transpose(-1, -2).flatten(-2)
            key_states = key_states.unflatten(-1, (2, -1)).transpose(-1, -2).flatten(-2)

        if hasattr(self, 'qk_norm'):
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


class Llama4TextMLP(nn.Module):
    """attention."""

    def __init__(self,
                 config: Llama4TextConfig,
                 intermediate_size: int = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 is_tp: bool = True,
                 all_reduce: bool = True):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = config.intermediate_size

        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)

        mlp_bias = False
        mlp_args = dict(
            bias=mlp_bias,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=is_tp,
        )
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            **mlp_args,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_rowwise_linear(
            intermediate_size,
            config.hidden_size,
            all_reduce=all_reduce,
            **mlp_args,
        )

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class Llama4TextMoe(nn.Module):
    """attention."""

    def __init__(self, config: Llama4TextConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts

        self.router = build_rowwise_linear(
            self.hidden_dim,
            self.num_experts,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=None,
        )
        self.experts = build_fused_moe(
            self.hidden_dim,
            self.ffn_dim,
            self.num_experts,
            top_k=1,
            renormalize=False,
            dtype=dtype,
            device=device,
            all_reduce=False,
            quant_config=quantization_config,
        )
        self.shared_expert = Llama4TextMLP(config, dtype=dtype, device=device, is_tp=True, all_reduce=False)

        dist_ctx = dist.get_dist_manager().current_context()
        self.tp = dist_ctx.tp

    def forward(self, hidden_states: torch.Tensor):
        """forward."""
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.router(hidden_states)

        topk_weights, topk_ids = torch.topk(router_logits, self.top_k, dim=-1)
        input_weight = topk_weights.float().sigmoid().to(hidden_states.dtype)

        moe_hidden_states = hidden_states[:, None, :] * input_weight[:, :, None]
        moe_hidden_states = moe_hidden_states.view(-1, hidden_dim)
        topk_weights = torch.ones_like(input_weight).reshape(-1, 1)
        topk_ids = topk_ids.reshape(-1, 1)

        out_states = self.experts(
            moe_hidden_states,
            topk_weights,
            topk_ids,
        )

        out_states = out_states.reshape(-1, self.top_k, hidden_dim)
        out_states = out_states.sum(1)

        shared_states = self.shared_expert(hidden_states)
        out_states += shared_states
        out_states = out_states.reshape(batch, seq_len, -1)

        if self.tp > 1:
            dist.all_reduce(out_states)

        return out_states


class Llama4TextDecoderLayer(nn.Module):
    """decoder layer."""

    def __init__(self,
                 config: Llama4TextConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = Llama4TextAttention(config, layer_idx, dtype=dtype, device=device)
        self.use_chunked_attention = int((layer_idx + 1) % 4 != 0)  # <=> use rope
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:  # the 128E model interleaves dense / sparse
            self.feed_forward = Llama4TextMoe(config, dtype=dtype, device=device)
        else:
            self.feed_forward = Llama4TextMLP(config,
                                              intermediate_size=config.intermediate_size_mlp,
                                              dtype=dtype,
                                              device=device)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):
        """forward."""

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class Llama4TextModel(nn.Module):
    """llama4 text model."""

    def __init__(self, config: Llama4TextConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)
        self.layers = nn.ModuleList([
            Llama4TextDecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)

        self.rotary_emb = self.build_llama4_rotary_embedding(config)

    @staticmethod
    def build_llama4_rotary_embedding(config: Llama4TextConfig):
        """build llama4 rotary embedding."""

        scaling_factor = 1.0
        rope_dim = config.hidden_size // config.num_attention_heads
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        llama3_params = None
        rope_scaling = config.rope_scaling
        if rope_scaling is None:
            emb_type = RopeType.Default
        else:
            emb_type = RopeType.Llama3
            low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
            high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
            llama3_params = Llama3Parameters(low_freq_factor, high_freq_factor)

        return build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            scaling_factor,
            llama3_params=llama3_params,
            emb_type=emb_type,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        **kwargs,
    ):
        """model forward."""
        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Llama4ForCausalLM(nn.Module):

    def __init__(self,
                 config: Llama4TextConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.model = Llama4TextModel(config, dtype=dtype, device=device)
        self.vocab_size = config.vocab_size
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            device=device,
                                            dtype=dtype)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        **kwargs,
    ):
        """model forward."""
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        return outputs

    def get_input_embeddings(self):
        """input embeddings."""
        return self.model.embed_tokens

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.lm_head(hidden_states)


class Llama4ForConditionalGeneration(nn.Module, CudaGraphMixin):

    def __init__(self,
                 config: Llama4Config,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr

        # TODO: add vision model
        self.vision_model = None

        # TODO: add projector
        self.multi_modal_projector = None

        self.language_model = Llama4ForCausalLM(config.text_config, ctx_mgr, dtype=dtype, device=device)
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def get_input_embeddings(self):
        """input embeddings."""
        return self.language_model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        pixel_values: torch.FloatTensor = None,
        **kwargs,
    ):
        """model forward."""

        # TODO: add vision

        inputs_embeds = self.get_input_embeddings()(input_ids)

        return self.language_model(
            inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
        )

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.language_model.get_logits(hidden_states)

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

        # TODO: add vision inputs

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
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        num_experts = self.config.text_config.num_local_experts

        params_dict = dict(self.named_parameters())
        device = next(iter(params_dict.values())).device
        for name, loaded_weight in weights:
            # TODO: support vision
            if 'vision_model' in name:
                continue
            elif 'multi_modal_projector' in name:
                continue

            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue

            if '.experts' in name:
                if '.gate_up_proj' in name:
                    loaded_weight = loaded_weight.to(device)
                    name = name.replace('.gate_up_proj', '.gate_up.weight')
                    param = params_dict[name]
                    for exp_id in range(num_experts):
                        weight_gate, weight_up = loaded_weight[exp_id].chunk(2, -1)
                        load_weight(param, weight_gate.t(), expert_id=exp_id, shard_id='gate')
                        load_weight(param, weight_up.t(), expert_id=exp_id, shard_id='up')
                elif '.down_proj' in name:
                    loaded_weight = loaded_weight.to(device)
                    name = name.replace('.down_proj', '.down.weight')
                    param = params_dict[name]
                    for exp_id in range(num_experts):
                        weight = loaded_weight[exp_id].t()
                        load_weight(param, weight, expert_id=exp_id, shard_id='down')
            else:
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    load_weight(param, loaded_weight, shard_id=shard_id)
                    break
                else:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
