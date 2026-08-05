# Copyright (c) OpenMMLab. All rights reserved.

from .builder import AutoModelConfigBuilder
from .qwen3_5 import Qwen3_5ModelConfigBuilder


class InternS2MobiusModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'interns2_mobius'

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
        # InternS2Mobius checkpoints are bf16, but newer transformers may
        # drop the nested text_config.dtype field from config.json when loading
        # trust_remote_code configs. Set a model-family default before the
        # generic dtype resolver falls back to fp16; otherwise the bf16 SSM
        # state cache and fp16 activations mismatch during warmup.
        if getattr(hf_config, 'dtype', None) is None and getattr(hf_config, 'torch_dtype', None) is None:
            hf_config.dtype = 'bfloat16'
        text_config = hf_config.text_config
        if getattr(text_config, 'dtype', None) is None and getattr(text_config, 'torch_dtype', None) is None:
            text_config.dtype = getattr(hf_config, 'dtype', 'bfloat16')

        cfg = Qwen3_5ModelConfigBuilder.build(
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
                hf_config.architectures[0] = 'InternS2MobiusForConditionalGeneration'
            text_config.num_blocks = getattr(text_config, 'num_blocks', 4)

        return cfg
