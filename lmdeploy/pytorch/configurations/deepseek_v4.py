# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util

import torch

from lmdeploy.pytorch.config import BlockCacheSpec, ModelConfig, StateCacheSpec
from lmdeploy.pytorch.consts import V4_FLASHMLA_D_NOPE, V4_FLASHMLA_D_ROPE, V4_FLASHMLA_NUM_TILES
from lmdeploy.utils import get_logger

from .builder import AutoModelConfigBuilder

logger = get_logger('lmdeploy')


V4_PACKED_TOKEN_DIM = V4_FLASHMLA_D_NOPE + 2 * V4_FLASHMLA_D_ROPE + V4_FLASHMLA_NUM_TILES + 1
V4_SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)


def _normalize_v4_block_size(block_size: int) -> int:
    """Return the logical block size required by DeepSeek-V4 kernels."""
    if block_size < 256:
        block_size = 256
    if block_size % 128 != 0:
        block_size = ((block_size + 127) // 128) * 128
        if block_size < 256:
            block_size = 256
    return block_size


def _get_v4_cache_layers(hf_config):
    """Normalize compression ratios and return layer-id partitions."""
    num_layers = hf_config.num_hidden_layers
    compress_ratios = getattr(hf_config, 'compress_ratios', None)
    if compress_ratios is None:
        compress_ratios = [0] * num_layers
    else:
        compress_ratios = list(compress_ratios)

    if len(compress_ratios) > num_layers:
        extra_ratios = compress_ratios[num_layers:]
        if any(r != 0 for r in extra_ratios):
            raise ValueError('DeepSeek-V4 compress_ratios has extra non-zero entries beyond '
                             f'num_hidden_layers={num_layers}: {extra_ratios}.')
        logger.warning('DeepSeek-V4 compress_ratios has %s entries but num_hidden_layers is %s. '
                       'Ignoring trailing zero entries.', len(compress_ratios), num_layers)
        compress_ratios = compress_ratios[:num_layers]
    elif len(compress_ratios) < num_layers:
        logger.warning('DeepSeek-V4 compress_ratios has %s entries but num_hidden_layers is %s. '
                       'Padding missing entries with 0.', len(compress_ratios), num_layers)
        compress_ratios = compress_ratios + [0] * (num_layers - len(compress_ratios))

    invalid_ratios = sorted({r for r in compress_ratios if r not in V4_SUPPORTED_COMPRESS_RATIOS})
    if invalid_ratios:
        raise ValueError('DeepSeek-V4 compress_ratios only supports '
                         f'{V4_SUPPORTED_COMPRESS_RATIOS}, but got {invalid_ratios}.')

    hf_config.compress_ratios = compress_ratios
    all_layers = list(range(num_layers))
    ratio4_layers = [i for i, r in enumerate(compress_ratios) if r == 4]
    ratio128_layers = [i for i, r in enumerate(compress_ratios) if r == 128]
    return all_layers, ratio4_layers, ratio128_layers


def _check_env_v4(device: str = 'cuda'):
    """Environment check for DeepSeek-V4."""
    if device != 'cuda':
        return

    try:
        import flash_mla  # noqa: F401
    except ImportError as e:
        raise ImportError('DeepSeek-V4 requires <flash_mla> to be installed.') from e

    try:
        import deep_gemm  # noqa: F401
    except ImportError as e:
        raise ImportError('DeepSeek-V4 requires <deep_gemm> to be installed.') from e

    try:
        import fast_hadamard_transform  # noqa: F401
    except ImportError as e:
        raise ImportError('DeepSeek-V4 requires <fast_hadamard_transform> to be installed.') from e

    if importlib.util.find_spec('tile_kernels') is None:
        raise ImportError('DeepSeek-V4 requires <tile_kernels> to be installed.')


def _finalize_v4_cache_specs(model_config: ModelConfig, block_size: int):
    normalized_block_size = _normalize_v4_block_size(block_size)
    if normalized_block_size != block_size:
        logger.warning(f'DeepSeek-V4 requires block_size >= 256 and a multiple of 128. '
                       f'Adjusting block_size from {model_config.block_size} to {normalized_block_size}.')
        model_config.block_size = normalized_block_size
        block_size = normalized_block_size

    hf_config = model_config.hf_config
    _, ratio4_layers, ratio128_layers = _get_v4_cache_layers(hf_config)

    block_specs = []
    if ratio4_layers:
        entries_r4 = block_size // 4
        index_head_dim = getattr(hf_config, 'index_head_dim', 128)
        block_specs.append(
            BlockCacheSpec('v4_compressed_kv_r4_fp8', ratio4_layers, (entries_r4, V4_PACKED_TOKEN_DIM),
                           torch.float8_e4m3fn))
        block_specs.append(
            BlockCacheSpec('v4_index_kv_r4', ratio4_layers, (entries_r4, index_head_dim), torch.float8_e4m3fn))
        block_specs.append(
            BlockCacheSpec('v4_index_kv_r4_scale', ratio4_layers, (entries_r4, 1), torch.float32))
    if ratio128_layers:
        entries_r128 = block_size // 128
        block_specs.append(
            BlockCacheSpec('v4_compressed_kv_r128_fp8', ratio128_layers, (entries_r128, V4_PACKED_TOKEN_DIM),
                           torch.float8_e4m3fn))

    model_config.block_cache_specs = block_specs


def update_cache_config(cache_config):
    original_block_size = cache_config.block_size
    original_kernel_block_size = cache_config.kernel_block_size
    block_size = _normalize_v4_block_size(original_block_size)
    if block_size != original_block_size:
        logger.warning(f'DeepSeek-V4 requires block_size >= 256 and a multiple of 128. '
                       f'Adjusting block_size from {original_block_size} to {block_size}.')
        cache_config.block_size = block_size
    if cache_config.kernel_block_size != block_size:
        logger.warning('DeepSeek-V4 requires kernel_block_size to match block_size. '
                       f'Adjusting kernel_block_size from {original_kernel_block_size} to {block_size}.')
        cache_config.kernel_block_size = block_size
    # V4 manages its sliding window via ring-buffer state caches internally.
    # Setting window_size=-1 selects DefaultBlockManager so blocks are not
    # dropped and kv_seqlens are not reduced by num_ignored_history.
    cache_config.window_size = -1


class DeepseekV4ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'deepseek_v4'

    @classmethod
    def build(cls, hf_config, model_path: str | None = None, tp: int = 1, **kwargs):
        """Build model config.

        Declares real V4 cache footprint via block_cache_specs / state_cache_specs and removes the dummy KV cache shapes
        used during bring-up.
        """
        bos_token_id = getattr(hf_config, 'bos_token_id', None)
        head_dim = getattr(hf_config, 'head_dim', 512)
        num_layers = hf_config.num_hidden_layers
        all_layers, ratio4_layers, ratio128_layers = _get_v4_cache_layers(hf_config)

        config = ModelConfig(
            hidden_size=hf_config.hidden_size,
            num_layers=num_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(hf_config, 'num_key_value_heads', 1),
            bos_token_id=bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            head_dim=head_dim,
            sliding_window=hf_config.sliding_window,
            vocab_size=hf_config.vocab_size,
            model_paradigm='ar',
            use_standard_kv_cache=False,
        )

        # ---- block cache specs ----
        # block_cache_specs depend on the final token block_size, so they are
        # materialized in a post-build hook after ModelConfig.block_size is set.
        config.block_cache_specs = []

        # ---- state cache specs ----
        state_specs = []
        state_specs.append(
            StateCacheSpec('v4_window_kv_fp8', (hf_config.sliding_window, V4_PACKED_TOKEN_DIM), torch.float8_e4m3fn,
                           layer_ids=all_layers))
        if ratio4_layers:
            # overlap compressor scratch for Attention (kv_state + score_state)
            # rows = 2 * ratio = 8, state_dim = 2 * head_dim
            # compress_state shape = (2 * rows, state_dim) = (16, 2 * head_dim)
            state_specs.append(
                StateCacheSpec('v4_compress_state_r4', (16, 2 * head_dim), torch.float32, layer_ids=ratio4_layers))
            index_head_dim = getattr(hf_config, 'index_head_dim', 128)
            # Indexer also has its own compressor (overlap=True because ratio==4)
            # rows = 2 * ratio = 8, state_dim = 2 * index_head_dim
            # compress_state shape = (2 * rows, state_dim) = (16, 2 * index_head_dim)
            state_specs.append(
                StateCacheSpec('v4_compress_state_r4_idx',
                               (16, 2 * index_head_dim),
                               torch.float32,
                               layer_ids=ratio4_layers))

        if ratio128_layers:
            # rows = ratio = 128, state_dim = head_dim
            # compress_state shape = (2 * rows, state_dim) = (256, head_dim)
            state_specs.append(
                StateCacheSpec('v4_compress_state_r128', (256, head_dim), torch.float32, layer_ids=ratio128_layers))

        config.state_cache_specs = state_specs
        # backward-compat bridge to keep scheduler.is_ssm working
        config.states_shapes = [(tuple(spec.shape), spec.dtype) for spec in state_specs]

        config.check_env_func = _check_env_v4
        config.post_build_func = _finalize_v4_cache_specs
        config.update_cache_config_func = update_cache_config
        return config
