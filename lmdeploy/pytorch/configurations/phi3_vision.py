# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class Phi3VisionModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'phi3_v'

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        """build."""
        cfg = DefaultModelConfigBuilder.build(hf_config)
        cfg.unused_modules = ['model.vision_embed_tokens']
        return cfg
