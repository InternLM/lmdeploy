# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class Qwen3OmniModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'qwen3_omni_moe'

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
        cfg = DefaultModelConfigBuilder.build(hf_config.thinker_config.text_config, model_path, **kwargs)
        cfg.hf_config = hf_config
        return cfg
