# Copyright (c) OpenMMLab. All rights reserved.
from .deepseek_v2 import DeepseekV2ModelConfigBuilder
from .default import DefaultModelConfigBuilder


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


class Glm4MoeModelConfigBuilder(DefaultModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['glm4_moe']

    @classmethod
    def build(cls, hf_config, model_path: str = None, is_draft_model: bool = False, spec_method: str = None, **kwargs):
        """build."""

        num_layers = hf_config.num_hidden_layers
        model_paradigm = 'ar'

        if spec_method is not None:
            assert spec_method == 'deepseek_mtp'

        # draft model cfg
        if is_draft_model:
            num_layers = hf_config.num_nextn_predict_layers
            hf_config.architectures[0] = 'Glm4MoeMTPModel'
            # remove for correct mapping when building the patched model
            if hasattr(hf_config, 'auto_map'):
                del hf_config.auto_map

        if is_draft_model or spec_method is not None:
            model_paradigm = 'ar_spec'

        cfg = super().build(hf_config,
                            model_path=model_path,
                            is_draft_model=is_draft_model,
                            spec_method=spec_method,
                            **kwargs)
        cfg.model_paradigm = model_paradigm
        cfg.num_layers = num_layers
        return cfg
