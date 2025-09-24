# Copyright (c) OpenMMLab. All rights reserved.
from .default import AutoModelConfigBuilder, DefaultModelConfigBuilder


class SDARModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['sdar', 'sdar_moe']

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)
        cfg.dllm_mask_token = 151669
        cfg.model_paradigm = 'dllm'
        return cfg
