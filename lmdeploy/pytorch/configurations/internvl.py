# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder, ProxyAutoModel
from .default import DefaultModelConfigBuilder


class InternVLModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.architectures[0] == 'InternVLChatModel'

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        """build llava hf."""
        from transformers import AutoModel
        cfg = DefaultModelConfigBuilder.build(hf_config.llm_config)
        cfg.unused_modules = ['InternVisionModel']
        cfg.hf_config = hf_config
        cfg.auto_model_cls = ProxyAutoModel(AutoModel)
        return cfg
