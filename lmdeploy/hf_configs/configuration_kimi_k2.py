# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class KimiK2Config(PretrainedConfig):
    """Text configuration for standalone Kimi-K2 draft checkpoints."""

    model_type = 'kimi_k2'

    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        hidden_act: str = 'silu',
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 262144,
        rope_theta: float = 50000.0,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads)
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_parameters = (
            rope_parameters if rope_parameters is not None else rope_scaling)
        self.attention_bias = attention_bias
        super().__init__(**kwargs)
