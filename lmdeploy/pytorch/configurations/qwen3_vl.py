# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class Qwen3VLModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'qwen3_vl' or hf_config.model_type == 'qwen3_vl_moe'

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
        if hasattr(hf_config, 'quantization_config') and not hasattr(hf_config.text_config, 'quantization_config'):
            setattr(hf_config.text_config, 'quantization_config', hf_config.quantization_config)
        cfg = DefaultModelConfigBuilder.build(hf_config.text_config, model_path, **kwargs)
        setattr(hf_config, 'dtype', hf_config.text_config.dtype)
        cfg.hf_config = hf_config
        return cfg
