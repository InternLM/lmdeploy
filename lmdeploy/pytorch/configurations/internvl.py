# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class ProxyAutoModel:
    """wrapper of auto model class."""

    def __init__(self) -> None:
        """init."""
        from transformers import AutoModel
        self.model_cls = AutoModel

    def from_config(self, *args, **kwargs):
        """wrap from_config."""
        if hasattr(self.model_cls, '_from_config'):
            if 'trust_remote_code' in kwargs:
                kwargs.pop('trust_remote_code')
            return self.model_cls._from_config(*args, **kwargs)
        else:
            return self.model_cls.from_config(*args, **kwargs)

    def from_pretrained(self, *args, **kwargs):
        """wrap from_pretrained."""
        from transformers import AutoConfig
        if hasattr(self.model_cls,
                   '_from_config') and 'trust_remote_code' in kwargs:
            kwargs.pop('trust_remote_code')
        config = AutoConfig.from_pretrained(args[0], trust_remote_code=True)
        quantization_config = getattr(config.llm_config, 'quantization_config',
                                      None)
        if quantization_config is not None:
            quantization_config['modules_to_not_convert'] = [
                'lm_head', 'vision_model'
            ]
            config.quantization_config = quantization_config
            kwargs['config'] = config
        return self.model_cls.from_pretrained(*args, **kwargs)


class InternVLModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.architectures[0] == 'InternVLChatModel'

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        """build llava hf."""
        cfg = DefaultModelConfigBuilder.build(hf_config.llm_config)
        cfg.unused_modules = ['InternVisionModel']
        cfg.hf_config = hf_config
        cfg.auto_model_cls = ProxyAutoModel()
        return cfg
