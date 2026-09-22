# Copyright (c) OpenMMLab. All rights reserved.

from .builder import AutoModelConfigBuilder
from .qwen3_5 import Qwen3_5ModelConfigBuilder


def _is_meta_moe_config(hf_config):
    """Detect the MetaMoE Qwen3.5-MoE checkpoint family."""
    architectures = getattr(hf_config, 'architectures', []) or []
    if any(arch in ['MetaMoeForConditionalGeneration', 'MetaMoeMTPModel'] for arch in architectures):
        return True

    text_config = getattr(hf_config, 'text_config', None)
    if text_config is None or getattr(hf_config, 'model_type', None) != 'qwen3_5_moe':
        return False

    num_experts = getattr(text_config, 'num_experts', None)
    mtp_num_experts = getattr(text_config, 'mtp_num_experts', None)
    return num_experts == 2560 and mtp_num_experts == 256


class MetaMoeModelConfigBuilder(Qwen3_5ModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return _is_meta_moe_config(hf_config)

    @classmethod
    def build(cls,
              hf_config,
              model_path: str = None,
              tp: int = 1,
              is_draft_model: bool = False,
              spec_method: str = None,
              num_spec_tokens: int = 0,
              **kwargs):
        """build."""
        cfg = super().build(
            hf_config,
            model_path=model_path,
            tp=tp,
            is_draft_model=is_draft_model,
            spec_method=spec_method,
            num_spec_tokens=num_spec_tokens,
            **kwargs,
        )

        text_config = hf_config.text_config
        if is_draft_model:
            if getattr(hf_config, 'architectures', None):
                hf_config.architectures[0] = 'Qwen3_5MTPModel'
            text_config.num_experts = text_config.mtp_num_experts
            text_config.num_experts_per_tok = text_config.mtp_num_experts_per_tok
        else:
            if getattr(hf_config, 'architectures', None):
                hf_config.architectures[0] = 'MetaMoeForConditionalGeneration'
            text_config.num_meta_moe_blocks = 4

        return cfg


AutoModelConfigBuilder._sub_classes.remove(MetaMoeModelConfigBuilder)
AutoModelConfigBuilder._sub_classes.insert(0, MetaMoeModelConfigBuilder)
