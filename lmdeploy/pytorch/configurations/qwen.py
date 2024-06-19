# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class QwenModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'qwen'

    @classmethod
    def build(cls, hf_config, model_path: str = None):
        """build."""
        import torch
        cfg = DefaultModelConfigBuilder.build(hf_config)
        if cfg.bos_token_id is None:
            cfg.bos_token_id = 151644
        if cfg.eos_token_id is None:
            cfg.eos_token_id = 151645

        is_bf16_supported = torch.cuda.is_bf16_supported()
        torch_dtype = 'bfloat16' if is_bf16_supported else 'float16'
        if hf_config.bf16 and is_bf16_supported:
            torch_dtype = 'bfloat16'
        elif hf_config.fp16:
            torch_dtype = 'float16'
        hf_config.torch_dtype = torch_dtype
        return cfg
