# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (Attention, RMSNorm, RopeType, SiluAndMul,
                                 build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_colwise_linear,
                                        build_merged_colwise_linear,
                                        build_rowwise_linear)
from lmdeploy.pytorch.nn.rotary_embedding import (ApplyRotaryEmb,
                                                  LongRoPEScalingParameters)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


# TODO use MLA of pytorch engine
class MiniCPMAttention(nn.Module):
    """minicpm3 attention."""

    def __init__(self,
                 config: Any,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = None
        self.q_lora_rank = config.q_lora_rank
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.hidden_size // config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        if self.q_lora_rank is None:
            self.q_proj = build_colwise_linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
                is_tp=True,
            )
        else:
            self.q_a_proj = build_colwise_linear(
                self.hidden_size,
                config.q_lora_rank,
                bias=config.attention_bias,
                dtype=dtype,
                device=device,
                is_tp=False,
            )
            self.q_a_layernorm = RMSNorm(config.q_lora_rank,
                                         1e-6,
                                         quant_config=quantization_config,
                                         dtype=dtype,
                                         device=device)
            self.q_b_proj = build_colwise_linear(
                config.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
                is_tp=True,
            )

        self.kv_a_proj_with_mqa = build_colwise_linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=False,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank,
                                      1e-6,
                                      quant_config=quantization_config,
                                      dtype=dtype,
                                      device=device)
        self.kv_b_proj = build_colwise_linear(
            config.kv_lora_rank,
            self.num_heads *
            (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
        )

        self.apply_rotary_pos_emb = ApplyRotaryEmb()
        self.softmax_scale = self.q_head_dim**(-0.5)
        self.attn_fwd = Attention(self.num_heads,
                                  config.kv_lora_rank + self.qk_rope_head_dim,
                                  scale=self.softmax_scale,
                                  num_kv_heads=config.num_key_value_heads)

        self.o_proj = build_rowwise_linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            is_tp=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        num_heads = self.num_heads // world_size
        bsz, q_len, _ = hidden_states.size()

        # qkv_proj
        bsz, q_len, _ = hidden_states.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)
        kv = (self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(
            bsz, q_len, num_heads, self.qk_nope_head_dim + self.v_head_dim))

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        q_pe, k_pe = self.apply_rotary_pos_emb(
            q_pe,
            k_pe,
            cos,
            sin,
            inplace=True,
        )

        query_states = k_pe.new_empty(bsz, q_len, self.num_heads,
                                      self.q_head_dim)
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = k_pe.new_empty(bsz, q_len, self.num_heads,
                                    self.q_head_dim)
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = torch.nn.functional.pad(
                value_states, [0, self.q_head_dim - self.v_head_dim])

        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            inplace=False,
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, :self.v_head_dim]

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads *
                                          self.v_head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output


class MiniCPMMLP(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_rowwise_linear(config.intermediate_size,
                                              config.hidden_size,
                                              bias=False,
                                              quant_config=quantization_config,
                                              dtype=dtype,
                                              device=device,
                                              is_tp=True)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class MiniCPMDecoderLayer(nn.Module):
    """decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attn = MiniCPMAttention(config, dtype=dtype, device=device)

        # builf MLP
        self.mlp = MiniCPMMLP(config, dtype=dtype, device=device)

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
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        attn_metadata: Any = None,
    ):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        hidden_states = residual + hidden_states * (
            self.scale_depth / math.sqrt(self.num_hidden_layers))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (
            self.scale_depth / math.sqrt(self.num_hidden_layers))

        outputs = (hidden_states, residual)
        return outputs


class MiniCPM3Model(nn.Module):
    """model."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.scale_emb = config.scale_emb

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)

        # build all decode layers
        self.layers = nn.ModuleList([
            MiniCPMDecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            dtype=dtype,
                            device=device)
        # build rotary embedding
        emb_type = RopeType.LinearScaling
        rope_dim = config.qk_rope_head_dim
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        rope_scaling = config.rope_scaling
        if rope_scaling is not None:
            scaling_type = rope_scaling['type']
            assert scaling_type in ['longrope', 'su']
            emb_type = RopeType.LongRoPEScaling
            ori_pos_emb = getattr(config, 'original_max_position_embeddings',
                                  rope_max_pos_emb)

            longrope_params = LongRoPEScalingParameters(
                short_factor=rope_scaling['short_factor'],
                long_factor=rope_scaling['long_factor'],
                original_max_position_embeddings=ori_pos_emb)
            self.rotary_emb = build_rotary_embedding(
                rope_dim,
                rope_max_pos_emb,
                rope_base,
                longrope_params=longrope_params,
                emb_type=emb_type,
            )
        else:
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
            inputs_embeds = self.embed_tokens(input_ids) * self.scale_emb

        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)
        # decoding
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, _ = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                attn_metadata=attn_metadata,
            )

        # norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.embed_tokens


class MiniCPM3ForCausalLM(nn.Module, CudaGraphMixin):
    """rewrote model of MiniCPM3ForCausalLM."""

    packed_modules_mapping = {
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
        # build LLamaModel
        self.model = MiniCPM3Model(config, dtype=dtype, device=device)
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
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

        logits = self.lm_head(
            hidden_states /
            (self.config.hidden_size / self.config.dim_model_base))
        return logits

    def update_weights(self):
        """update weights."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # ('.qkv_proj', '.q_proj', 'q'),
            # ('.qkv_proj', '.k_proj', 'k'),
            # ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name
                    or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
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
