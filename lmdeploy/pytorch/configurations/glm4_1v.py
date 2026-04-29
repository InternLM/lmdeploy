# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class Glm4vModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        return hf_config.model_type == 'glm4v'

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
        bos_token_id = getattr(hf_config, 'bos_token_id', None)
        hf_config.text_config.bos_token_id = bos_token_id
        cfg = DefaultModelConfigBuilder.build(hf_config.text_config, model_path, **kwargs)
        cfg.hf_config = hf_config
        return cfg
