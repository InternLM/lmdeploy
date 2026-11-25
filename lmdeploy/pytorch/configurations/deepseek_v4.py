# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
from dataclasses import dataclass

import torch

from lmdeploy.pytorch.config import ModelConfig, StateCacheSpec
from lmdeploy.pytorch.consts import (
    V4_FLASHMLA_RING_STORAGE_ALIGNMENT,
    V4_PACKED_TOKEN_DIM,
)
from lmdeploy.utils import get_logger

from .builder import AutoModelConfigBuilder

logger = get_logger('lmdeploy')


V4_BLOCK_SIZE = 256
V4_SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
V4_SUPPORTED_LAYER_TYPES = (
    'sliding_attention',
    'compressed_sparse_attention',
    'heavily_compressed_attention',
)


@dataclass(frozen=True)
class V4RingGeometry:
    """Logical visibility and physical storage for the V4 local KV ring."""

    logical_window_size: int
    ring_capacity: int
    ring_storage_capacity: int


def get_v4_ring_geometry(hf_config, spec_method: str | None = None,
                         num_spec_tokens: int = 0) -> V4RingGeometry:
    """Resolve a speculation-safe ring without changing SWA semantics."""
    logical_window_size = int(hf_config.sliding_window)
    # Every AR-spec target verifies ``num_spec_tokens + 1`` causal rows in one
    # step. The rectangular V4 executor writes the speculative rows before it
    # reads all position-specific windows, so all methods need W+N physical
    # ring positions; this is not DSpark-specific.
    spec_extra = int(num_spec_tokens) if spec_method is not None else 0
    if logical_window_size < 1:
        raise ValueError(f'DeepSeek-V4 sliding_window must be positive, got {logical_window_size}.')
    if spec_extra < 0:
        raise ValueError(f'DeepSeek-V4 speculative token count must be non-negative, got {spec_extra}.')
    ring_capacity = logical_window_size + spec_extra
    # FlashMLA accepts W+N logical positions, but its split value/scale FP8
    # layout requires each state-slot extent to keep the scale region aligned.
    alignment = V4_FLASHMLA_RING_STORAGE_ALIGNMENT
    ring_storage_capacity = ((ring_capacity + alignment - 1) // alignment *
                             alignment)
    return V4RingGeometry(logical_window_size, ring_capacity,
                          ring_storage_capacity)

def get_v4_compress_ratios(hf_config) -> list[int]:
    """Translate the native Transformers layer schema to compression ratios."""
    num_layers = hf_config.num_hidden_layers
    layer_types = list(hf_config.layer_types)
    if len(layer_types) != num_layers:
        raise ValueError('DeepSeek-V4 requires one layer_type per hidden layer, but got '
                         f'{len(layer_types)} layer types for {num_layers} layers.')

    invalid_layer_types = sorted(set(layer_types).difference(V4_SUPPORTED_LAYER_TYPES))
    if invalid_layer_types:
        raise ValueError('DeepSeek-V4 layer_types only supports '
                         f'{V4_SUPPORTED_LAYER_TYPES}, but got {invalid_layer_types}.')

    compress_rates = hf_config.compress_rates
    compressed_layer_types = set(layer_types).difference({'sliding_attention'})
    missing_rates = sorted(compressed_layer_types.difference(compress_rates))
    if missing_rates:
        raise ValueError(f'DeepSeek-V4 compress_rates is missing layer types: {missing_rates}.')

    compress_ratios = [
        0 if layer_type == 'sliding_attention' else compress_rates[layer_type]
        for layer_type in layer_types
    ]

    invalid_ratios = sorted({r for r in compress_ratios if r not in V4_SUPPORTED_COMPRESS_RATIOS})
    if invalid_ratios:
        raise ValueError('DeepSeek-V4 compression only supports ratios '
                         f'{V4_SUPPORTED_COMPRESS_RATIOS}, but got {invalid_ratios}.')
    return compress_ratios


def _get_v4_cache_layers(hf_config):
    """Return layer-id partitions for each V4 compression ratio."""
    num_layers = hf_config.num_hidden_layers
    compress_ratios = get_v4_compress_ratios(hf_config)
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


def update_cache_config(cache_config):
    original_block_size = cache_config.block_size
    original_kernel_block_size = cache_config.kernel_block_size
    block_size = V4_BLOCK_SIZE
    if block_size != original_block_size:
        logger.warning(f'DeepSeek-V4 requires block_size={V4_BLOCK_SIZE}. '
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
        """Build model config with configuration-owned V4 state caches."""
        bos_token_id = getattr(hf_config, 'bos_token_id', None)
        head_dim = getattr(hf_config, 'head_dim', 512)
        spec_method = kwargs.get('spec_method')
        num_spec_tokens = kwargs.get('num_spec_tokens', 0)
        ring_geometry = get_v4_ring_geometry(hf_config, spec_method,
                                             num_spec_tokens)
        # Model constructors receive the HF config, whereas cache allocation
        # consumes ModelConfig. Record the same immutable geometry in both.
        hf_config.v4_ring_storage_capacity = \
            ring_geometry.ring_storage_capacity
        is_dspark_draft = bool(kwargs.get('is_draft_model', False)
                               and spec_method == 'dspark')
        if is_dspark_draft:
            target_layer_ids = getattr(hf_config, 'dspark_target_layer_ids', None)
            if not target_layer_ids:
                raise ValueError('Bundled DeepSeek-V4 DSpark requires dspark_target_layer_ids.')
            num_layers = len(target_layer_ids)
            all_layers = list(range(num_layers))
            ratio4_layers = []
            ratio128_layers = []
        else:
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
            v4_ring_storage_capacity=ring_geometry.ring_storage_capacity,
        )

        # ---- state cache specs ----
        state_specs = []
        state_specs.append(
            StateCacheSpec('v4_window_kv_fp8', (ring_geometry.ring_storage_capacity, V4_PACKED_TOKEN_DIM),
                           torch.float8_e4m3fn,
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
        config.update_cache_config_func = update_cache_config
        return config
