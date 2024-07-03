# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class CogVLMModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        model_arch = getattr(hf_config, 'architectures', [None])[0]
        return model_arch == 'CogVLMForCausalLM'

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        """build."""
        import torch
        cfg = DefaultModelConfigBuilder.build(hf_config)
        if getattr(hf_config, 'num_multi_query_heads', None):
            cfg.num_key_value_heads = hf_config.num_multi_query_heads
        cfg.unused_modules = ['model.vision']
        torch_dtype = 'bfloat16' if torch.cuda.is_bf16_supported(
        ) else 'float16'
        hf_config.torch_dtype = torch_dtype
        return cfg
