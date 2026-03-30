# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class GptOSSModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['gpt_oss']

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """Build gemma."""
        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)
        # gpt_oss 3 does not enable sliding window on every layers
        cfg.sliding_window = -1
        cfg.hf_config = hf_config
        return cfg
