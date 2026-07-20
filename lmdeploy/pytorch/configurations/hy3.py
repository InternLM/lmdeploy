# Copyright (c) OpenMMLab. All rights reserved.

from .default import DefaultModelConfigBuilder


class Hy3ModelConfigBuilder(DefaultModelConfigBuilder):
    """Model config builder for Hy3."""

    @classmethod
    def condition(cls, hf_config):
        """Match Hy3 configurations."""
        return hf_config.model_type == 'hy_v3'
