# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .deepseek_v2 import DeepseekV2ModelConfigBuilder


def normalize_glm_moe_dsa_config(hf_config):
    """Normalize GLM-MoE-DSA config fields for DeepSeek-V3.2 reuse."""
    if hf_config.model_type != 'glm_moe_dsa':
        return

    qk_head_dim = getattr(hf_config, 'qk_head_dim', None)
    if qk_head_dim is not None and qk_head_dim != hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim:
        qk_rope_head_dim = qk_head_dim - hf_config.qk_nope_head_dim
        if qk_rope_head_dim <= 0:
            raise ValueError('Invalid GLM-MoE-DSA qk head dimensions.')
        hf_config.qk_rope_head_dim = qk_rope_head_dim

    hf_config.qk_head_dim = hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim
    hf_config.head_dim = hf_config.qk_rope_head_dim

    indexer_types = getattr(hf_config, 'indexer_types', None)
    if indexer_types is not None:
        indexer_types = list(indexer_types)
    else:
        pattern = getattr(hf_config, 'index_topk_pattern', None)
        if pattern is not None:
            if isinstance(pattern, str):
                pattern_map = {'F': 'full', 'S': 'shared'}
                indexer_types = [pattern_map[item.upper()] for item in pattern]
            else:
                indexer_types = list(pattern)
        else:
            freq = max(getattr(hf_config, 'index_topk_freq', 1), 1)
            offset = getattr(hf_config, 'index_skip_topk_offset', 2)
            indexer_types = [
                'full' if max(layer_idx - offset + 1, 0) % freq == 0 else 'shared'
                for layer_idx in range(hf_config.num_hidden_layers)
            ]

    if len(indexer_types) != hf_config.num_hidden_layers:
        raise ValueError('GLM-MoE-DSA indexer_types length must match num_hidden_layers.')
    if any(item not in ('full', 'shared') for item in indexer_types):
        raise ValueError('GLM-MoE-DSA indexer_types only supports "full" and "shared".')
    if indexer_types[0] != 'full':
        raise ValueError('The first GLM-MoE-DSA layer must be a full indexer layer.')

    hf_config.indexer_types = indexer_types


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
        normalize_glm_moe_dsa_config(hf_config)
        config = DeepseekV2ModelConfigBuilder.build(hf_config, model_path=model_path, **kwargs)

        assert hf_config.use_flash_mla, 'DeepSeek-V3.2 requires flash_mla to be available.'
        index_k_shape = ([hf_config.index_head_dim], torch.float8_e4m3fn)
        index_k_scale_shape = ([1], torch.float32)
        config.cache_shapes = [index_k_shape, index_k_scale_shape]
        config.use_mla_fp8_cache = True
        config.mla_index_topk = hf_config.index_topk
        config.check_env_func = _check_env_v32
        return config
