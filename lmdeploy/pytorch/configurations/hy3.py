# Copyright (c) OpenMMLab. All rights reserved.

from .default import DefaultModelConfigBuilder


class Hy3ModelConfigBuilder(DefaultModelConfigBuilder):
    """Model config builder for Hy3."""

    @classmethod
    def condition(cls, hf_config):
        """Match Hy3 configurations."""
        return hf_config.model_type == 'hy_v3'

    @classmethod
    def build(
        cls,
        hf_config,
        model_path: str = None,
        is_draft_model: bool = False,
        spec_method: str = None,
        **kwargs,
    ):
        """Build target or MTP draft configuration."""
        if spec_method is not None and spec_method != 'hy3_mtp':
            raise ValueError(f'Unsupported speculative method for Hy3: {spec_method}')

        num_mtp_layers = int(getattr(hf_config, 'num_nextn_predict_layers', 0) or 0)
        if (is_draft_model or spec_method == 'hy3_mtp') and num_mtp_layers < 1:
            raise ValueError('Hy3 MTP requires at least one checkpoint MTP layer.')

        config = super().build(
            hf_config,
            model_path=model_path,
            **kwargs,
        )

        if is_draft_model:
            hf_config.architectures[0] = 'HYV3MTP'
            if hasattr(hf_config, 'auto_map'):
                del hf_config.auto_map
            config.num_layers = num_mtp_layers

        if is_draft_model or spec_method is not None:
            config.model_paradigm = 'ar_spec'

        return config
