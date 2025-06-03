# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class Llama4ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['llama4']

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """Build llama4."""
        cfg = DefaultModelConfigBuilder.build(hf_config.text_config, model_path, **kwargs)
        cfg.hf_config = hf_config

        return cfg
