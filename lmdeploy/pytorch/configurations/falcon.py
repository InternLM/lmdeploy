# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import ModelConfig

from .builder import AutoModelConfigBuilder


class FalconModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'falcon'

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        """build falcon."""
        num_attention_heads = hf_config.num_attention_heads
        if hf_config.new_decoder_architecture:
            # 40b-instruct, GQA
            kv_head = hf_config.num_kv_heads
        if hf_config.multi_query:
            # 7b-instruct, MQA
            kv_head = 1
        else:
            # rw-1b, MHA
            kv_head = num_attention_heads
        head_dim = hf_config.hidden_size // num_attention_heads
        return ModelConfig(
            hidden_size=hf_config.hidden_size,
            num_layers=hf_config.num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=kv_head,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            head_dim=head_dim,
            multi_query_attention=hf_config.multi_query,
            vocab_size=hf_config.vocab_size,
        )
