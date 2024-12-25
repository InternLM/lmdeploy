# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class MLlamaModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.architectures[0] == 'MllamaForConditionalGeneration'

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build llava hf."""
        cfg = DefaultModelConfigBuilder.build(hf_config.text_config,
                                              model_path, **kwargs)
        cfg.hf_config = hf_config
        return cfg
