# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.pytorch import envs as _envs
from lmdeploy.utils import get_logger

from .deepseek_v32 import DeepseekV32ModelConfigBuilder

logger = get_logger('lmdeploy')

class GlmMoeDsaModelConfigBuilder(DeepseekV32ModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'glm_moe_dsa'

    @classmethod
    def build(cls, hf_config, model_path: str | None = None, **kwargs):
        """build."""
        is_draft_model = kwargs.get('is_draft_model', False)
        quantization_config = getattr(hf_config, 'quantization_config', None)
        is_lmdeploy_patched_fp8 = (quantization_config is not None
                                   and quantization_config.get('quant_method') == 'fp8'
                                   and quantization_config.get('lmdeploy_patched', False))
        if _envs.fp8_moe_only and is_lmdeploy_patched_fp8:
            quantization_config['fp8_quant_scope'] = 'moe_only'
            logger.info('Enable fp8_quant_scope=moe_only for glm_moe_dsa because LMDEPLOY_FP8_MOE_ONLY=1 '
                        'and the FP8 quantization config is LMDeploy-synthesized.')

        if hf_config.qk_head_dim != hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim:
            hf_config.qk_rope_head_dim = hf_config.qk_head_dim - hf_config.qk_nope_head_dim
        hf_config.head_dim = hf_config.qk_rope_head_dim

        config = super().build(hf_config, model_path=model_path, **kwargs)
        if is_draft_model:
            hf_config.architectures[0] = 'GlmMoeDsaMTPModel'
        return config
