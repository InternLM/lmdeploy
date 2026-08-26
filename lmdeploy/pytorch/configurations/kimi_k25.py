# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .kimi_k2 import KimiK2ModelConfigBuilder


class KimiK25ModelConfigBuilder(AutoModelConfigBuilder):
    """Build the PyTorch engine config for Kimi-K2.5/K2.6."""

    @classmethod
    def condition(cls, hf_config):
        """Match the outer KimiK25 multimodal configuration."""
        return getattr(hf_config, 'model_type', None) == 'kimi_k25'

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """Reuse the Kimi-K2 DeepSeek MLA text configuration."""
        if not hasattr(hf_config, 'text_config'):
            raise ValueError('KimiK25 config must define `text_config`.')

        text_config = hf_config.text_config
        if hasattr(hf_config, 'quantization_config') and not hasattr(
                text_config, 'quantization_config'):
            text_config.quantization_config = hf_config.quantization_config

        cfg = KimiK2ModelConfigBuilder.build(text_config, model_path, **kwargs)

        text_dtype = getattr(text_config, 'dtype', None)
        if text_dtype is not None:
            hf_config.dtype = text_dtype

        # The outer config owns vision/media settings, while the nested config
        # is the only valid input for the DeepSeek language-model builder.
        cfg.hf_config = hf_config
        cfg.llm_config = text_config
        return cfg
