# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class GemmaModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['gemma', 'gemma2']

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build gemma."""
        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)
        cfg.head_dim = hf_config.head_dim
        return cfg
