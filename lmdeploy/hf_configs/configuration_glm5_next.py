# Copyright (c) OpenMMLab. All rights reserved.
"""Local Transformers configuration for GLM-5.3-Flash.

Transformers releases that predate GLM-5.3 do not know the ``glm5_next``
model type.  Keeping the configuration local lets LMDeploy read official
checkpoints without executing remote modeling code.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig

_GLM5_NEXT_TOP_LEVEL_CONFIG_KEYS = (
    'architectures',
    'vocab_size',
    'hidden_size',
    'head_dim',
    'intermediate_size',
    'moe_intermediate_size',
    'num_hidden_layers',
    'num_attention_heads',
    'num_key_value_heads',
    'hidden_act',
    'max_position_embeddings',
    'rms_norm_eps',
    'use_cache',
    'pad_token_id',
    'bos_token_id',
    'eos_token_id',
    'rope_theta',
    'rope_scaling',
    'rope_parameters',
    'partial_rotary_factor',
    'tie_word_embeddings',
    'attention_bias',
    'attention_dropout',
    'n_routed_experts',
    'num_experts_per_tok',
    'n_shared_experts',
    'n_group',
    'topk_group',
    'norm_topk_prob',
    'routed_scaling_factor',
    'scoring_func',
    'topk_method',
    'first_k_dense_replace',
    'moe_layer_freq',
    'moe_router_dtype',
    'output_router_logits',
    'router_aux_loss_coef',
    'q_lora_rank',
    'kv_lora_rank',
    'qk_head_dim',
    'qk_nope_head_dim',
    'qk_rope_head_dim',
    'v_head_dim',
    'mla_use_nope',
    'swiglu_limit',
    'mhc',
    'hc_mult',
    'hc_sinkhorn_iters',
    'hc_eps',
    'num_nextn_predict_layers',
    'linear_attn_config',
    'linear_head_dim',
    'linear_num_heads',
    'linear_conv_kernel_dim',
    'linear_lower_bound',
    'gate_lower_bound',
    'index_head_dim',
    'index_topk',
    'index_kpool',
    'index_kpool_always_select_tail',
    'index_kpool_compress',
    'index_n_heads',
    'index_topk_freq',
    'index_topk_pattern',
    'index_skip_topk_offset',
    'index_share_for_mtp_iteration',
    'indexer_rope_interleave',
    'indexer_types',
    'layer_types',
    'mlp_layer_types',
    'initializer_range',
    'quantization_config',
)

_GLM5_NEXT_OUTER_ONLY_CONFIG_KEYS = frozenset({
    'architectures',
    'quantization_config',
})


class Glm5NextTextConfig(PretrainedConfig):
    """Configuration of the GLM-5.3 text tower."""

    model_type = 'glm5_next_text'
    base_config_key = 'text_config'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self,
                 vocab_size: int = 154880,
                 hidden_size: int = 4096,
                 head_dim: int | None = 0,
                 intermediate_size: int = 12288,
                 moe_intermediate_size: int = 2048,
                 num_hidden_layers: int = 45,
                 num_attention_heads: int = 64,
                 num_key_value_heads: int | None = 64,
                 hidden_act: str = 'silu',
                 max_position_embeddings: int = 1048576,
                 rms_norm_eps: float = 1e-5,
                 use_cache: bool = True,
                 pad_token_id: int | None = None,
                 bos_token_id: int | None = None,
                 eos_token_id: int | list[int] | None = None,
                 rope_theta: float = 800000.0,
                 rope_scaling: dict[str, Any] | None = None,
                 rope_parameters: dict[str, Any] | None = None,
                 partial_rotary_factor: float = 1.0,
                 tie_word_embeddings: bool = False,
                 attention_bias: bool = False,
                 attention_dropout: float = 0.0,
                 n_routed_experts: int | None = 288,
                 num_experts_per_tok: int = 8,
                 n_shared_experts: int | None = 1,
                 n_group: int = 1,
                 topk_group: int = 1,
                 norm_topk_prob: bool = True,
                 routed_scaling_factor: float = 2.5,
                 scoring_func: str = 'sigmoid',
                 topk_method: str = 'noaux_tc',
                 first_k_dense_replace: int = 3,
                 moe_layer_freq: int | None = 1,
                 moe_router_dtype: str = 'float32',
                 output_router_logits: bool = False,
                 router_aux_loss_coef: float = 0.001,
                 q_lora_rank: int | None = 1536,
                 kv_lora_rank: int = 512,
                 qk_head_dim: int | None = 256,
                 qk_nope_head_dim: int = 256,
                 qk_rope_head_dim: int = 0,
                 v_head_dim: int = 256,
                 mla_use_nope: bool = True,
                 swiglu_limit: float | None = 10.0,
                 mhc: bool = True,
                 hc_mult: int = 4,
                 hc_sinkhorn_iters: int = 20,
                 hc_eps: float = 1e-6,
                 num_nextn_predict_layers: int = 1,
                 linear_attn_config: dict[str, Any] | None = None,
                 linear_head_dim: int = 128,
                 linear_num_heads: int = 64,
                 linear_conv_kernel_dim: int = 4,
                 linear_lower_bound: float | None = None,
                 gate_lower_bound: float | None = -5.0,
                 index_head_dim: int | None = 128,
                 index_topk: int | None = 2048,
                 index_kpool: int = 4,
                 index_kpool_always_select_tail: bool = True,
                 index_kpool_compress: bool = True,
                 index_n_heads: int | None = 32,
                 index_topk_freq: int = 1,
                 index_topk_pattern: str | None = None,
                 index_skip_topk_offset: int | None = None,
                 index_share_for_mtp_iteration: bool = True,
                 indexer_rope_interleave: bool = True,
                 indexer_types: list[str] | None = None,
                 layer_types: list[str] | None = None,
                 mlp_layer_types: list[str] | None = None,
                 initializer_range: float = 0.02,
                 **kwargs):
        if rope_scaling is None and rope_parameters is not None:
            rope_scaling = rope_parameters
        if rope_parameters is not None:
            rope_theta = rope_parameters.get('rope_theta', rope_theta)
            partial_rotary_factor = rope_parameters.get(
                'partial_rotary_factor', partial_rotary_factor)

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if qk_head_dim is None:
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # Some released configs serialize this optional field as null.  The
        # GLM-5.3 layout uses MoE on every layer after the dense prefix.
        if moe_layer_freq is None:
            moe_layer_freq = 1

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_parameters
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.moe_router_dtype = moe_router_dtype
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.swiglu_limit = swiglu_limit
        self.mhc = mhc
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.index_kpool_compress = index_kpool_compress
        self.index_n_heads = index_n_heads
        self.index_topk_freq = index_topk_freq
        self.index_topk_pattern = index_topk_pattern
        self.index_skip_topk_offset = index_skip_topk_offset
        self.index_share_for_mtp_iteration = index_share_for_mtp_iteration
        self.indexer_rope_interleave = indexer_rope_interleave
        self.indexer_types = indexer_types
        self.layer_types = layer_types
        self.mlp_layer_types = mlp_layer_types
        self.initializer_range = initializer_range
        self.linear_lower_bound = linear_lower_bound
        self.gate_lower_bound = (gate_lower_bound
                                 if gate_lower_bound is not None else
                                 linear_lower_bound)

        if linear_attn_config is None:
            if layer_types is None:
                kda_layers = [
                    layer_idx for layer_idx in range(num_hidden_layers)
                    if layer_idx % 4 != 3
                ]
            else:
                kda_layers = [
                    layer_idx for layer_idx, layer_type in enumerate(layer_types)
                    if layer_type == 'linear_attention'
                ]
            kda_layer_set = set(kda_layers)
            linear_attn_config = {
                'full_attn_layers': [
                    layer_idx for layer_idx in range(num_hidden_layers)
                    if layer_idx not in kda_layer_set
                ],
                'head_dim': linear_head_dim,
                'kda_layers': kda_layers,
                'num_heads': linear_num_heads,
                'short_conv_kernel_size': linear_conv_kernel_dim,
                'gate_lower_bound': self.gate_lower_bound,
            }
        else:
            linear_attn_config = dict(linear_attn_config)

        linear_head_dim = linear_attn_config.get('head_dim',
                                                 linear_head_dim)
        linear_num_heads = linear_attn_config.get('num_heads',
                                                  linear_num_heads)
        linear_conv_kernel_dim = linear_attn_config.get(
            'short_conv_kernel_size', linear_conv_kernel_dim)
        if gate_lower_bound is None:
            gate_lower_bound = linear_attn_config.get('gate_lower_bound',
                                                       linear_lower_bound)
        linear_attn_config.setdefault('head_dim', linear_head_dim)
        linear_attn_config.setdefault('num_heads', linear_num_heads)
        linear_attn_config.setdefault('short_conv_kernel_size',
                                      linear_conv_kernel_dim)
        linear_attn_config.setdefault('gate_lower_bound', gate_lower_bound)

        self.linear_head_dim = linear_head_dim
        self.linear_num_heads = linear_num_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.gate_lower_bound = gate_lower_bound
        self.linear_attn_config = linear_attn_config

        super().__init__(pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings,
                         **kwargs)
        if rope_parameters is not None or rope_scaling is not None:
            self.rope_parameters = rope_parameters or rope_scaling

    def is_kda_layer(self, layer_idx: int) -> bool:
        """Return whether ``layer_idx`` uses KDA instead of sparse MLA."""
        return (self.linear_attn_config is not None
                and layer_idx in self.linear_attn_config['kda_layers'])

    @property
    def linear_layer_ids(self) -> list[int]:
        return [
            layer_idx for layer_idx in range(self.num_hidden_layers)
            if self.is_kda_layer(layer_idx)
        ]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            layer_idx for layer_idx in range(self.num_hidden_layers)
            if not self.is_kda_layer(layer_idx)
        ]

    @property
    def nextn_layer_ids(self) -> list[int]:
        return [
            self.num_hidden_layers + layer_idx
            for layer_idx in range(self.num_nextn_predict_layers or 0)
        ]


class Glm5NextVisionConfig(GlmOcrVisionConfig):
    """GLM-OCR vision configuration with clamped SwiGLU."""

    model_type = 'glm5_next_vision'

    def __init__(self, swiglu_limit: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.swiglu_limit = swiglu_limit


class Glm5NextConfig(PretrainedConfig):
    """Top-level GLM-5.3 multimodal configuration."""

    model_type = 'glm5_next'
    sub_configs = {
        'vision_config': Glm5NextVisionConfig,
        'text_config': Glm5NextTextConfig,
    }
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self,
                 text_config: dict[str, Any] | PretrainedConfig | None = None,
                 vision_config: dict[str, Any] | PretrainedConfig | None = None,
                 image_token_id: int = 154854,
                 video_token_id: int = 154855,
                 image_start_token_id: int = 154830,
                 image_end_token_id: int = 154831,
                 video_start_token_id: int = 154832,
                 video_end_token_id: int = 154833,
                 **kwargs):
        top_level_text_config = {
            key: kwargs[key]
            for key in _GLM5_NEXT_TOP_LEVEL_CONFIG_KEYS
            if key in kwargs and key not in _GLM5_NEXT_OUTER_ONLY_CONFIG_KEYS
        }
        if isinstance(text_config, dict):
            text_config = {**top_level_text_config, **text_config}
            text_config = Glm5NextTextConfig(**text_config)
        elif text_config is None:
            text_config = Glm5NextTextConfig(**top_level_text_config)
        self.text_config = text_config

        if isinstance(vision_config, dict):
            vision_config = Glm5NextVisionConfig(**vision_config)
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        if getattr(self.text_config, 'quantization_config', None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(**kwargs)

        # Existing LMDeploy configuration builders still read language fields
        # from the outer config.  Mirror language fields without overwriting
        # top-level model-selection or quantization metadata.
        for key in _GLM5_NEXT_TOP_LEVEL_CONFIG_KEYS:
            if (key not in _GLM5_NEXT_OUTER_ONLY_CONFIG_KEYS
                    and hasattr(self.text_config, key)):
                setattr(self, key, getattr(self.text_config, key))
