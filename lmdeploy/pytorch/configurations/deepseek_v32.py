# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.config import BlockCacheSpec, ModelConfig
from lmdeploy.pytorch.consts import DSA_INDEX_CACHE_NAME, dsa_packed_index_cache_shape

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


def _finalize_v32_cache_specs(model_config: ModelConfig, block_size: int):
    """Give DeepGEMM a physically contiguous paged DSA index cache."""
    hf_config = model_config.hf_config
    indexer_types = getattr(hf_config, 'indexer_types', None)
    if model_config.num_layers == hf_config.num_hidden_layers and indexer_types:
        layer_ids = [
            layer_id for layer_id, indexer_type in enumerate(indexer_types)
            if indexer_type != 'shared'
        ]
    else:
        # Draft-model layer ids are local to its own cache engine.
        layer_ids = list(range(model_config.num_layers))
    model_config.cache_shapes = []
    model_config.block_cache_specs = [
        BlockCacheSpec(DSA_INDEX_CACHE_NAME,
                       layer_ids,
                       dsa_packed_index_cache_shape(
                           block_size, hf_config.index_head_dim),
                       torch.uint8)
    ]


class DeepseekV32ModelConfigBuilder(DeepseekV2ModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'deepseek_v32'

    @classmethod
    def build(cls, hf_config, model_path: str | None = None, **kwargs):
        """build."""
        is_draft_model = kwargs.get('is_draft_model', False)
        config = DeepseekV2ModelConfigBuilder.build(hf_config, model_path=model_path, **kwargs)

        assert hf_config.use_flash_mla, 'DeepSeek-V3.2 requires flash_mla to be available.'
        index_k_shape = ([hf_config.index_head_dim], torch.float8_e4m3fn)
        index_k_scale_shape = ([1], torch.float32)
        config.cache_shapes = [index_k_shape, index_k_scale_shape]
        config.mla_kv_cache_dtype = 'fp8_ds_mla'
        config.mla_index_topk = hf_config.index_topk
        config.check_env_func = _check_env_v32
        config.post_build_func = _finalize_v32_cache_specs
        if is_draft_model:
            hf_config.architectures[0] = 'DeepseekV32MTPModel'
        return config
