# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import ModelConfig

from .builder import AutoModelConfigBuilder


class DBRXModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'dbrx'

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        """build."""
        hidden_size = hf_config.d_model
        num_heads = hf_config.n_heads
        head_dim = hidden_size // num_heads
        eos_token_id = getattr(hf_config, 'eos_token_id', None)
        if eos_token_id is None:
            eos_token_id = 100257
        bos_token_id = getattr(hf_config, 'bos_token_id', None)
        if bos_token_id is None:
            bos_token_id = eos_token_id
        return ModelConfig(
            hidden_size=hidden_size,
            num_layers=hf_config.n_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=hf_config.attn_config.kv_n_heads,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            head_dim=head_dim,
            vocab_size=hf_config.vocab_size)
