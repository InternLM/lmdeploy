# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import ModelConfig

from .builder import AutoModelConfigBuilder


class GemmaModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'gemma'

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        """build gemma."""
        return ModelConfig(hidden_size=hf_config.hidden_size,
                           num_layers=hf_config.num_hidden_layers,
                           num_attention_heads=hf_config.num_attention_heads,
                           num_key_value_heads=hf_config.num_key_value_heads,
                           bos_token_id=hf_config.bos_token_id,
                           eos_token_id=hf_config.eos_token_id,
                           head_dim=hf_config.head_dim,
                           vocab_size=hf_config.vocab_size)
