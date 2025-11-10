# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import ModelConfig

from .builder import AutoModelConfigBuilder


class DefaultModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return True

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""

        # for multi-modal models, get the language model config to build model config
        if hasattr(hf_config, 'text_config'):
            hf_config = hf_config.text_config
        elif hasattr(hf_config, 'llm_config'):
            hf_config = hf_config.llm_config

        head_dim = getattr(hf_config, 'head_dim', None)
        head_dim = head_dim or hf_config.hidden_size // hf_config.num_attention_heads

        # head_dim should not be None
        hf_config.head_dim = head_dim
        num_attention_heads = hf_config.num_attention_heads
        num_key_value_heads = getattr(hf_config, 'num_key_value_heads', num_attention_heads)
        use_sliding_window = getattr(hf_config, 'use_sliding_window', True)
        sliding_window = -1
        if use_sliding_window:
            sliding_window = getattr(hf_config, 'sliding_window', sliding_window) or -1
        tp = kwargs.get('tp', 1)
        # update num_kv_heads for tp mode
        num_key_value_heads = cls.update_num_kv_heads(hf_config, tp, num_key_value_heads)

        return ModelConfig(
            hidden_size=hf_config.hidden_size,
            num_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            sliding_window=sliding_window,
            head_dim=head_dim,
            k_head_dim=head_dim,
            v_head_dim=head_dim,
            vocab_size=hf_config.vocab_size,
            llm_config=hf_config,
        )
