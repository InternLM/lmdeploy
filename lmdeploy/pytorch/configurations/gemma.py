# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class GemmaModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['gemma', 'gemma2', 'gemma3_text']

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """Build gemma."""
        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)
        cfg.head_dim = hf_config.head_dim
        return cfg


class GemmaVLModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        model_arch = hf_config.architectures[0] if hf_config.architectures else None
        return model_arch == 'Gemma3ForConditionalGeneration'

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """Build gemma."""
        hf_config.text_config.architectures = ['Gemma3ForCausalLM']
        cfg = DefaultModelConfigBuilder.build(hf_config.text_config, model_path, **kwargs)
        # gemma 3 does not enable sliding window on every layers
        cfg.sliding_window = -1
        cfg.hf_config = hf_config
        return cfg
