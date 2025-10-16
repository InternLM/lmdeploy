# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class Qwen3MoEModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['qwen3_moe']

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """Build qwen3 moe."""
        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)
        num_moe_layers = None
        num_experts_per_tok = getattr(hf_config, 'num_experts_per_tok', None)
        if num_experts_per_tok is not None:
            num_moe_layers = hf_config.num_hidden_layers - len(getattr(hf_config, 'mlp_only_layers', []))
        cfg.num_experts_per_tok = num_experts_per_tok
        cfg.num_moe_layers = num_moe_layers
        return cfg
