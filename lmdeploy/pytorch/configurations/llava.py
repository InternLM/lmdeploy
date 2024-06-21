# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder, ProxyAutoModel
from .default import DefaultModelConfigBuilder


class LlavaModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.architectures[0] in [
            'LlavaLlamaForCausalLM', 'LlavaMistralForCausalLM'
        ]

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        """build."""
        arch = hf_config.architectures[0]
        if arch == 'LlavaLlamaForCausalLM':
            from llava.model.language_model.llava_llama import LlavaConfig
            from llava.model.language_model.llava_llama import \
                LlavaLlamaForCausalLM as LlavaModel

            # reload hf_config due to model_type='llava' is already
            # registered in transformers
            hf_config = LlavaConfig.from_pretrained(model_path)
        elif arch == 'LlavaMistralForCausalLM':
            from llava.model.language_model.llava_mistral import \
                LlavaMistralForCausalLM as LlavaModel
        cfg = DefaultModelConfigBuilder.build(hf_config)
        cfg.auto_model_cls = ProxyAutoModel(LlavaModel)
        cfg.unused_modules = ['model.vision_tower', 'model.mm_projector']
        return cfg
