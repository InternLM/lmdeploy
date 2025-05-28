# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class DeepseekVLV2ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['deepseek_vl_v2']

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """Build deepseek-vl2."""

        if hf_config.language_config.use_mla:
            from .deepseek_v2 import DeepseekV2ModelConfigBuilder
            cfg = DeepseekV2ModelConfigBuilder.build(hf_config.language_config, model_path, **kwargs)
            cfg.hf_config = hf_config
        else:
            # deepseek-vl2-tiny uses MHA, rather than MLA
            # in this case, we use DefaultModelConfigBuilder
            cfg = DefaultModelConfigBuilder.build(hf_config.language_config, model_path, **kwargs)
            cfg.hf_config = hf_config

        return cfg
