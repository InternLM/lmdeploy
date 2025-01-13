# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import ModelConfig

from .builder import AutoModelConfigBuilder


class MedusaModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.architectures[0] == 'MedusaModel'

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
        from transformers import AutoConfig
        base_config = AutoConfig.from_pretrained(
            hf_config.base_model_name_or_path)
        head_dim = base_config.hidden_size // base_config.num_attention_heads
        # config is wrong
        # https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3/blob/main/config.json#L3
        hf_config.medusa_num_heads = 5
        medusa_num_heads = hf_config.medusa_num_heads
        medusa_num_layers = hf_config.medusa_num_layers
        if getattr(hf_config, 'hidden_size', None) is None:
            setattr(hf_config, 'hidden_size', base_config.hidden_size)
        if getattr(hf_config, 'vocab_size', None) is None:
            setattr(hf_config, 'vocab_size', base_config.vocab_size)
        return ModelConfig(
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            num_layers=base_config.num_hidden_layers,
            num_key_value_heads=base_config.num_key_value_heads,
            bos_token_id=base_config.bos_token_id,
            eos_token_id=base_config.eos_token_id,
            head_dim=head_dim,
            vocab_size=base_config.vocab_size,
            hf_config=hf_config,
            medusa_num_heads=medusa_num_heads,
            medusa_num_layers=medusa_num_layers,
        )
