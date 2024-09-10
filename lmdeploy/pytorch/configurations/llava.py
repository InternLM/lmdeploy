# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
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
        if arch in ['LlavaLlamaForCausalLM', 'LlavaMistralForCausalLM']:
            from llava.model.language_model.llava_llama import LlavaConfig

            # reload hf_config due to model_type='llava' is already
            # registered in transformers
            hf_config = LlavaConfig.from_pretrained(model_path)
        cfg = DefaultModelConfigBuilder.build(hf_config)
        return cfg
