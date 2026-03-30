# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .deepseek_v2 import DeepseekV2ModelConfigBuilder


def _check_env_v32(device: str = 'cuda'):
    """Environment check."""
    if device != 'cuda':
        return

    # check cuda
    try:
        import fast_hadamard_transform  # noqa: F401
    except ImportError:
        raise ImportError('Deepseek V3.2 requires <fast_hadamard_transform>.')

    try:
        import flash_mla  # noqa: F401
    except ImportError:
        raise ImportError('Deepseek V3.2 requires <flash_mla>.')

    if not hasattr(flash_mla, 'flash_mla_sparse_fwd'):
        raise RuntimeError('Latest flash_mla is required: https://github.com/deepseek-ai/FlashMLA.')


class DeepseekV32ModelConfigBuilder(DeepseekV2ModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['deepseek_v32', 'glm_moe_dsa']

    @classmethod
    def build(cls, hf_config, model_path: str | None = None, **kwargs):
        """build."""
        config = DeepseekV2ModelConfigBuilder.build(hf_config, model_path=model_path, **kwargs)

        assert hf_config.use_flash_mla, 'DeepSeek-V3.2 requires flash_mla to be available.'
        index_k_shape = ([hf_config.index_head_dim], torch.float8_e4m3fn)
        index_k_scale_shape = ([1], torch.float32)
        config.cache_shapes = [index_k_shape, index_k_scale_shape]
        config.use_mla_fp8_cache = True
        config.mla_index_topk = hf_config.index_topk
        config.check_env_func = _check_env_v32
        return config
