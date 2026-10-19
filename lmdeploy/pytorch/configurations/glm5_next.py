# Copyright (c) OpenMMLab. All rights reserved.
"""PyTorch engine configuration for GLM-5.3-Flash."""

from functools import partial

import torch

from lmdeploy.pytorch.config import StateCacheSpec
from lmdeploy.pytorch.consts import (
    GDN_CONV_STATE_CACHE_NAME,
    GDN_REPLAY_BETA_CACHE_NAME,
    GDN_REPLAY_CHECKPOINT_CACHE_NAME,
    GDN_REPLAY_CURSOR_CACHE_NAME,
    GDN_REPLAY_D_CACHE_NAME,
    GDN_REPLAY_G_CACHE_NAME,
    GDN_REPLAY_K_CACHE_NAME,
    GDN_REPLAY_RAW_K_CACHE_NAME,
    GDN_REPLAY_RAW_V_CACHE_NAME,
    GLM5_KDA_CONV_STATE,
    GLM5_KDA_RECURRENT_STATE,
    GLM5_KPOOL_TAIL_K_STATE,
    GLM5_KPOOL_TAIL_SCORE_STATE,
)
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.utils import get_logger

from .builder import AutoModelConfigBuilder
from .deepseek_v32 import DeepseekV32ModelConfigBuilder
from .qwen3_next import _check_env_qwen3_next

_GLM5_LINEAR_LAYER_TYPE = 'linear_attention'
_GLM5_FULL_LAYER_TYPES = frozenset({
    'deepseek_sparse_attention',
    'full_attention',
})


def update_cache_config(cache_config, token_lookahead=0):
    """Keep the 64-token MLA / 16-entry KPool owner under one allocator
    reference."""
    if cache_config.role != EngineRole.Hybrid:
        raise ValueError('GLM-5.3 logical64/kernel64 cache does not support PD migration.')
    if (cache_config.block_size, cache_config.kernel_block_size) != (64, 64):
        get_logger('lmdeploy').warning(
            'GLM-5.3 uses block_size=64 and kernel_block_size=64 '
            '(requested %s/%s).', cache_config.block_size, cache_config.kernel_block_size)
    cache_config.block_size = 64
    cache_config.kernel_block_size = 64
    cache_config.prefix_cache_block_aligned = True
    cache_config.prefix_cache_token_lookahead = token_lookahead
    if token_lookahead and cache_config.prefix_cache_decode_state_interval:
        raise ValueError('GLM-5.3 MTP prefix caching only publishes prefill checkpoints.')
    if cache_config.prefix_cache_decode_state_interval % 64:
        raise ValueError('GLM-5.3 prefix_cache_decode_state_interval must be a multiple of 64.')


def is_glm5_kda_layer(text_config, layer_idx: int) -> bool:
    """Return the normalized hybrid-layer kind across process boundaries.

    ``_resolve_glm5_linear_config`` materializes ``linear_layer_ids`` as an
    instance field, so it survives Ray serialization.  In contrast, the
    compatibility method installed on a native Transformers config class is
    process-local and is not available in a freshly spawned worker.
    """
    linear_layer_ids = getattr(text_config, 'linear_layer_ids', None)
    if linear_layer_ids is None:
        _, linear_layer_ids, _ = _resolve_glm5_linear_config(text_config)
    return int(layer_idx) in linear_layer_ids


def _compat_is_kda_layer(text_config, layer_idx: int) -> bool:
    """Provide the legacy predicate for native Transformers configs."""
    return is_glm5_kda_layer(text_config, layer_idx)


def _set_layer_ids_compat(text_config, name: str,
                          layer_ids: list[int]) -> None:
    """Expose a resolved layer map without breaking legacy properties."""
    try:
        setattr(text_config, name, layer_ids)
    except AttributeError:
        # The LMDeploy-owned legacy config exposes these as read-only
        # properties.  Its properties read the normalized legacy dict below.
        existing = list(getattr(text_config, name))
        if existing != layer_ids:
            raise ValueError(
                f'GLM-5.3 {name}={existing} conflicts with layer_types '
                f'({layer_ids}).')


def _resolve_glm5_linear_config(text_config):
    """Normalize official and legacy GLM-5.3 hybrid-layer schemas.

    Transformers 5.16 represents the hybrid layout with ``layer_types`` and
    linear-attention geometry with scalar fields.  Older checkpoints and the
    LMDeploy fallback config expose ``linear_attn_config`` plus convenience
    layer-id properties.  Keep ``layer_types`` authoritative when present and
    materialize the legacy view consumed by the existing model components.
    """
    num_layers = int(text_config.num_hidden_layers)
    legacy = dict(getattr(text_config, 'linear_attn_config', None) or {})
    layer_types = getattr(text_config, 'layer_types', None)

    if layer_types is not None:
        layer_types = list(layer_types)
        if len(layer_types) != num_layers:
            raise ValueError(
                'GLM-5.3 requires one layer_type per hidden layer, but got '
                f'{len(layer_types)} layer types for {num_layers} layers.')
        invalid_types = sorted(
            set(layer_types).difference({_GLM5_LINEAR_LAYER_TYPE},
                                        _GLM5_FULL_LAYER_TYPES))
        if invalid_types:
            raise ValueError(
                'GLM-5.3 layer_types only supports linear_attention, '
                'deepseek_sparse_attention, or full_attention, but got '
                f'{invalid_types}.')
        linear_layer_ids = [
            layer_idx for layer_idx, layer_type in enumerate(layer_types)
            if layer_type == _GLM5_LINEAR_LAYER_TYPE
        ]
        full_attention_layer_ids = [
            layer_idx for layer_idx, layer_type in enumerate(layer_types)
            if layer_type != _GLM5_LINEAR_LAYER_TYPE
        ]
    else:
        linear_layer_ids = getattr(text_config, 'linear_layer_ids', None)
        full_attention_layer_ids = getattr(
            text_config, 'full_attention_layer_ids', None)
        if linear_layer_ids is None:
            linear_layer_ids = legacy.get('kda_layers')
        if full_attention_layer_ids is None:
            full_attention_layer_ids = legacy.get('full_attn_layers')

        all_layer_ids = set(range(num_layers))
        if linear_layer_ids is None and full_attention_layer_ids is None:
            # Match the original GLM-5.3 fallback config for checkpoints that
            # predate both explicit schemas.
            linear_layer_ids = [
                layer_idx for layer_idx in range(num_layers)
                if layer_idx % 4 != 3
            ]
        if linear_layer_ids is None:
            linear_layer_ids = sorted(
                all_layer_ids.difference(full_attention_layer_ids))
        if full_attention_layer_ids is None:
            full_attention_layer_ids = sorted(
                all_layer_ids.difference(linear_layer_ids))
        linear_layer_ids = list(linear_layer_ids)
        full_attention_layer_ids = list(full_attention_layer_ids)

    expected_layer_ids = list(range(num_layers))
    resolved_layer_ids = linear_layer_ids + full_attention_layer_ids
    if sorted(resolved_layer_ids) != expected_layer_ids:
        raise ValueError(
            'GLM-5.3 linear/full layer ids must form an exact partition of '
            f'[0, {num_layers}), got linear={linear_layer_ids}, '
            f'full={full_attention_layer_ids}.')

    def _linear_value(native_name: str, legacy_name: str, default=None):
        value = getattr(text_config, native_name, None)
        if value is None:
            value = legacy.get(legacy_name, default)
        return value

    num_heads = _linear_value('linear_num_heads', 'num_heads')
    head_dim = _linear_value('linear_head_dim', 'head_dim')
    conv_kernel_size = _linear_value('linear_conv_kernel_dim',
                                     'short_conv_kernel_size')
    gate_lower_bound = _linear_value('linear_lower_bound',
                                     'gate_lower_bound', -5.0)
    missing = [
        name for name, value in (
            ('linear_num_heads/num_heads', num_heads),
            ('linear_head_dim/head_dim', head_dim),
            ('linear_conv_kernel_dim/short_conv_kernel_size',
             conv_kernel_size),
        ) if value is None
    ]
    if missing:
        raise ValueError('GLM-5.3 is missing linear-attention config fields: '
                         + ', '.join(missing) + '.')

    legacy.update({
        'num_heads': int(num_heads),
        'head_dim': int(head_dim),
        'short_conv_kernel_size': int(conv_kernel_size),
        'gate_lower_bound': gate_lower_bound,
        'kda_layers': linear_layer_ids,
        'full_attn_layers': full_attention_layer_ids,
    })
    text_config.linear_attn_config = legacy
    _set_layer_ids_compat(text_config, 'linear_layer_ids', linear_layer_ids)
    _set_layer_ids_compat(text_config, 'full_attention_layer_ids',
                          full_attention_layer_ids)
    if not callable(getattr(text_config, 'is_kda_layer', None)):
        # Install a class method instead of an instance closure so HF's
        # ``to_dict`` remains JSON serializable after model-config building.
        setattr(type(text_config), 'is_kda_layer', _compat_is_kda_layer)
    return legacy, linear_layer_ids, full_attention_layer_ids


def configure_kda_replayssm(config, layer_ids, heads, dim, conv_width,
                           num_spec_tokens, buffer_len):
    """Replace only KDA banks; KPool tails retain their upstream layout.

    Unlike scalar GDN, decay is channelwise and scales the key factor. Keep transformed keys/deltas in FP32 so
    reconstruction does not introduce a BF16 rounding at each retained KDA transition.
    """
    if buffer_len not in (8, 16, 32):
        raise ValueError('replayssm_buffer_len must be one of 8, 16, or 32.')
    if num_spec_tokens and buffer_len < 2 * (num_spec_tokens + 1):
        raise ValueError('MTP ReplaySSM requires L >= 2 * Tmax.')
    config.replayssm_baseline_state_shapes = list(config.states_shapes)
    config.enable_replayssm = True
    config.replayssm_buffer_len = buffer_len
    config.replayssm_mode = 'mtp' if num_spec_tokens else 'standard'
    shapes = [
        (GDN_CONV_STATE_CACHE_NAME, (3 * heads * dim, conv_width + num_spec_tokens), torch.bfloat16),
        (GDN_REPLAY_CHECKPOINT_CACHE_NAME, (heads, dim, dim), torch.float32),
        (GDN_REPLAY_D_CACHE_NAME, (heads, buffer_len, dim), torch.float32),
        (GDN_REPLAY_K_CACHE_NAME, (heads, buffer_len, dim), torch.float32),
        (GDN_REPLAY_G_CACHE_NAME, (heads, buffer_len, dim), torch.float32),
    ]
    if num_spec_tokens:
        shapes.extend([
            (GDN_REPLAY_RAW_V_CACHE_NAME, (heads, buffer_len, dim), torch.bfloat16),
            (GDN_REPLAY_RAW_K_CACHE_NAME, (heads, buffer_len, dim), torch.bfloat16),
            (GDN_REPLAY_BETA_CACHE_NAME, (heads, buffer_len), torch.float32),
        ])
    tails = [spec for spec in config.state_cache_specs
             if spec.name in (GLM5_KPOOL_TAIL_K_STATE, GLM5_KPOOL_TAIL_SCORE_STATE)]
    config.state_cache_specs = [StateCacheSpec(name, shape, dtype, layer_ids=layer_ids)
                                for name, shape, dtype in shapes]
    config.state_cache_specs.extend([
        StateCacheSpec(GDN_REPLAY_CURSOR_CACHE_NAME, (3,), torch.int32), *tails])
    config.states_shapes = [(tuple(spec.shape), spec.dtype) for spec in config.state_cache_specs]


def _check_env_glm5_next(device: str):
    """Validate the reused KDA, dense-MLA, and FA3 dependencies."""
    _check_env_qwen3_next(device)
    if device != 'cuda':
        return

    try:
        import flash_mla
    except ImportError:
        raise ImportError('GLM-5.3 CUDA support requires <flash_mla>.')
    if not hasattr(flash_mla, 'flash_mla_with_kvcache'):
        raise RuntimeError(
            'GLM-5.3 requires FlashMLA dense paged-attention support.')

    # KPool group selection reuses LMDeploy's shared sparse-index Top-K
    # implementation; validate the two model geometries without importing
    # SGLang providers.
    from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import (
        is_sparse_index_topk_supported,
    )
    if not all(is_sparse_index_topk_supported(k) for k in (512, 2048)):
        raise RuntimeError(
            'GLM-5.3 KPool requires LMDeploy sparse-index Top-K support for '
            'K=512 and K=2048.')


class Glm5NextModelConfigBuilder(AutoModelConfigBuilder):
    """Combine sparse-MLA KV cache and KDA recurrent-state resources."""

    @classmethod
    def condition(cls, hf_config):
        return getattr(hf_config, 'model_type', None) == 'glm5_next'

    @classmethod
    def build(cls, hf_config, model_path: str | None = None, **kwargs):
        if not hasattr(hf_config, 'text_config'):
            raise ValueError('GLM-5.3 config must define `text_config`.')

        text_config = hf_config.text_config
        if text_config.index_kpool != 4:
            raise ValueError('GLM-5.3 currently requires index_kpool=4 for its 64-token cache owner.')
        quant_config = getattr(hf_config, 'quantization_config', None)
        if quant_config is not None:
            text_config.quantization_config = quant_config

        linear_config, linear_layer_ids, full_attention_layer_ids = (
            _resolve_glm5_linear_config(text_config))
        is_draft = kwargs.get('is_draft_model', False)
        num_spec_tokens = kwargs.get('num_spec_tokens', 0)
        if is_draft and getattr(text_config, 'num_nextn_predict_layers', 0) != 1:
            raise ValueError('GLM-5.3 MTP requires one checkpoint predictor layer.')
        if (kwargs.get('enable_replayssm', False) and not is_draft
                and kwargs.get('spec_method') not in (None, 'deepseek_mtp')):
            raise ValueError('GLM-5.3 ReplaySSM supports deepseek_mtp only.')
        config = DeepseekV32ModelConfigBuilder.build(
            text_config, model_path=model_path,
            **dict(kwargs, is_draft_model=False))

        tp = kwargs.get('tp', 1)
        num_linear_layers = len(linear_layer_ids)
        num_full_layers = len(full_attention_layer_ids)
        num_heads = linear_config['num_heads']
        head_dim = linear_config['head_dim']
        conv_kernel_size = linear_config['short_conv_kernel_size']
        if num_heads % tp:
            raise ValueError(
                f'GLM-5.3 linear attention has {num_heads} heads, which is '
                f'not divisible by TP={tp}.')

        local_heads = num_heads // tp
        conv_dim = 3 * local_heads * head_dim
        config.num_layers = num_full_layers
        # GLM owns KPool selection in the model adapter, rather than the
        # DeepSeek-V3.2 token indexer selected by mla_index_topk.  Keeping this
        # unset also preserves the BF16 latent MLA cache policy.
        config.mla_index_topk = None
        config.reserve_dsa_score_workspace = True
        config.k_head_dim = text_config.kv_lora_rank
        # Reuse Qwen3.5's token ring for convolution; recurrent/KPool states
        # keep a complete checkpoint after each verified token. Store resumes
        # on 64-token boundaries, where the unfinished KPool tail is empty.
        ring_shape = (num_spec_tokens + 1,) if num_spec_tokens else ()
        config.state_cache_specs = [
            StateCacheSpec(
                GLM5_KDA_CONV_STATE,
                (num_linear_layers, conv_dim, conv_kernel_size + num_spec_tokens),
                torch.bfloat16,
            ),
            StateCacheSpec(
                GLM5_KDA_RECURRENT_STATE,
                (num_linear_layers, *ring_shape, local_heads, head_dim, head_dim),
                torch.float32,
            ),
            StateCacheSpec(
                GLM5_KPOOL_TAIL_K_STATE,
                (num_full_layers, text_config.index_kpool + 1 + num_spec_tokens,
                 text_config.index_head_dim),
                torch.bfloat16,
                store_cacheable=False,
                # Prefix checkpoints are 64-token aligned and index_kpool=4:
                # the incomplete pool is empty at restore. Keep the runtime
                # raw ring for chunk continuation and speculative rejection.
                prefix_checkpoint=False,
            ),
            StateCacheSpec(
                GLM5_KPOOL_TAIL_SCORE_STATE,
                (num_full_layers, text_config.index_kpool + 1 + num_spec_tokens,
                 text_config.index_head_dim),
                torch.bfloat16,
                store_cacheable=False,
                prefix_checkpoint=False,
            ),
        ]
        # Scheduler compatibility: named specs own allocation, while this
        # bridge keeps hybrid state checkpointing enabled in legacy callers.
        config.states_shapes = [
            (tuple(spec.shape), spec.dtype)
            for spec in config.state_cache_specs
        ]
        config.is_gated_delta = True
        if kwargs.get('enable_replayssm', False) and not is_draft:
            configure_kda_replayssm(
                config, linear_layer_ids, local_heads, head_dim, conv_kernel_size,
                num_spec_tokens, kwargs.get('replayssm_buffer_len', 16))
        config.update_cache_config_func = (partial(update_cache_config, token_lookahead=1)
                                           if num_spec_tokens else update_cache_config)
        if kwargs.get('spec_method') not in (None, 'deepseek_mtp'):
            config.prefix_caching_unsupported_reason = (
                'GLM-5.3 prefix caching supports AR and deepseek_mtp only.')
        config.check_env_func = _check_env_glm5_next
        config.hf_config = hf_config
        config.llm_config = text_config
        if is_draft:
            # The predictor has MLA/KPool but no KDA or mHC. Its unfinished
            # KPool tail is reconstructed from its own pageable token cache.
            hf_config.architectures = ['Glm5NextMTPModel']
            if hasattr(hf_config, 'auto_map'):
                del hf_config.auto_map
            config.num_layers = 1
            config.state_cache_specs = []
            config.states_shapes = []
            config.is_gated_delta = False

        text_dtype = getattr(text_config, 'dtype', None)
        if text_dtype is not None:
            hf_config.dtype = text_dtype
        return config
