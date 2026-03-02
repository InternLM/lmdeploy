# Copyright (c) OpenMMLab. All rights reserved.
from .deepseek_v2 import DeepseekV2ModelConfigBuilder


class Glm4MoeLiteModelConfigBuilder(DeepseekV2ModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['glm4_moe_lite']

    @classmethod
    def build(cls, hf_config, model_path: str = None, is_draft_model: bool = False, spec_method: str = None, **kwargs):
        """build."""
        # set default attrs
        if not hasattr(hf_config, 'scoring_func'):
            hf_config.scoring_func = 'sigmoid'
        if not hasattr(hf_config, 'moe_layer_freq'):
            hf_config.moe_layer_freq = 1
        return super().build(hf_config,
                             model_path=model_path,
                             is_draft_model=is_draft_model,
                             spec_method=spec_method,
                             **kwargs)
