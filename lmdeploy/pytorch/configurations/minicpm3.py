# Copyright (c) OpenMMLab. All rights reserved.

from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class MiniCPM3ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.architectures[0] in ['MiniCPM3ForCausalLM']

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
        head_dim = (hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim)

        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)
        cfg.head_dim = head_dim
        cfg.k_head_dim = head_dim
        cfg.v_head_dim = head_dim

        return cfg
