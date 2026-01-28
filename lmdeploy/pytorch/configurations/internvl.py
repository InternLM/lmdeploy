# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class InternVLModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.architectures[0] == 'InternVLChatModel'

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """Build internvl hf."""
        # hack quantization_config
        if hasattr(hf_config, 'quantization_config') and not hasattr(hf_config.llm_config, 'quantization_config'):
            setattr(hf_config.llm_config, 'quantization_config', hf_config.quantization_config)
        cfg = DefaultModelConfigBuilder.build(hf_config.llm_config, model_path, **kwargs)
        cfg.hf_config = hf_config
        return cfg
