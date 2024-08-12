# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder, ProxyAutoModel
from .default import DefaultModelConfigBuilder


class InternVLProxyAutoModel(ProxyAutoModel):

    def __init__(self, model_cls=None) -> None:
        super().__init__(model_cls)

    def from_pretrained(self, pretrained_model_name_or_path, *args, **kwargs):
        if 'device_map' in kwargs:
            return self.model_cls.from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                            trust_remote_code=True)
        return self.model_cls.from_config(config, *args, **kwargs)


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
        cfg.auto_model_cls = InternVLProxyAutoModel(AutoModel)
        return cfg
