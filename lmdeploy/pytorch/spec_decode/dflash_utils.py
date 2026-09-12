# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Any


def _parse_layer_ids(value: Any) -> tuple[int, ...] | None:
    """Parse target layer ids."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError('DFlash dflash_config.target_layer_ids must be a list of integers.')
    if len(value) == 0:
        raise ValueError('DFlash dflash_config.target_layer_ids must not be empty.')
    if any(isinstance(layer_id, bool) or not isinstance(layer_id, int) for layer_id in value):
        raise ValueError(f'DFlash target_layer_ids must contain only integers, got {value!r}.')
    layer_ids = tuple(value)
    _validate_target_layer_id_order(layer_ids, 'DFlash dflash_config.target_layer_ids')
    return layer_ids


def _validate_target_layer_id_order(layer_ids: tuple[int, ...], field_name: str) -> None:
    """Validate DFlash target feature ids are in checkpoint feature order."""
    seen: set[int] = set()
    prev_layer_id: int | None = None
    for pos, layer_id in enumerate(layer_ids):
        if layer_id in seen:
            raise ValueError(f'{field_name} must be duplicate-free; duplicate layer id {layer_id} at position {pos}.')
        if prev_layer_id is not None and layer_id <= prev_layer_id:
            raise ValueError(f'{field_name} must be strictly increasing and duplicate-free; got {layer_ids}.')
        seen.add(layer_id)
        prev_layer_id = layer_id


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> tuple[int, ...]:
    """Select evenly spaced DFlash target layer ids.

    DFlash consumes hidden features sampled from the target model. This fallback matches the SGLang/vLLM convention used
    when checkpoint metadata does not explicitly list target layers.
    """
    if num_target_layers < 1:
        raise ValueError(f'DFlash num_target_layers must be positive, got {num_target_layers!r}.')
    if num_draft_layers < 1:
        raise ValueError(f'DFlash num_hidden_layers must be positive, got {num_draft_layers!r}.')
    if num_draft_layers == 1:
        return (num_target_layers // 2,)

    start = 1
    end = num_target_layers - 3
    if end < start:
        raise ValueError(f'DFlash target layer fallback requires at least 4 target layers, got {num_target_layers}.')
    span = end - start
    layer_ids = tuple(int(round(start + i * span / (num_draft_layers - 1))) for i in range(num_draft_layers))
    _validate_target_layer_id_order(layer_ids, 'DFlash fallback target_layer_ids')
    return layer_ids


def _normalize_sliding_window(sliding_window: Any) -> int | None:
    """Normalize the no-window values used by HF and ``ModelConfig``."""
    if sliding_window in (None, 0, -1):
        return None
    if isinstance(sliding_window, bool) or not isinstance(sliding_window, int):
        raise ValueError(f'Invalid DFlash sliding_window: {sliding_window!r}.')
    if sliding_window < 1:
        raise ValueError(f'DFlash sliding_window must be positive, got {sliding_window!r}.')
    return sliding_window


def _validate_dflash_v1_supported(draft_hf_config: Any, dflash_config: Any) -> None:
    """Validate the complete DFlash V1 checkpoint contract once."""
    if not isinstance(dflash_config, dict):
        raise ValueError('DFlash checkpoint requires a dflash_config dictionary.')

    num_hidden_layers = draft_hf_config.num_hidden_layers
    layer_types = getattr(draft_hf_config, 'layer_types', None)
    if not isinstance(layer_types, (list, tuple)):
        raise ValueError('DFlash V1 requires an explicit layer_types list.')
    if len(layer_types) != num_hidden_layers:
        raise ValueError('DFlash layer_types length must equal num_hidden_layers. '
                         f'Got len(layer_types)={len(layer_types)}, num_hidden_layers={num_hidden_layers}.')
    if any(not isinstance(layer_type, str) for layer_type in layer_types):
        raise ValueError('DFlash layer_types must contain only strings.')
    unsupported_layer_types = sorted(
        {layer_type for layer_type in layer_types if layer_type not in ('full_attention', 'sliding_attention')})
    if unsupported_layer_types:
        raise ValueError('DFlash supports only full_attention and sliding_attention draft layers. '
                         f'Got unsupported layer_types={unsupported_layer_types}.')

    use_sliding_window = getattr(draft_hf_config, 'use_sliding_window', True)
    default_sliding_window = None
    if use_sliding_window:
        default_sliding_window = _normalize_sliding_window(getattr(draft_hf_config, 'sliding_window', None))

    has_sliding_layer = 'sliding_attention' in layer_types
    if has_sliding_layer and default_sliding_window is None:
        raise ValueError('DFlash sliding_attention layers require the model-default sliding_window.')

    explicit_sliding_window = dflash_config.get('swa_window_size', dflash_config.get('sliding_window'))
    if explicit_sliding_window is not None:
        explicit_sliding_window = _normalize_sliding_window(explicit_sliding_window)
        if explicit_sliding_window != default_sliding_window:
            raise ValueError('DFlash V1 requires the SWA window to equal ModelConfig.sliding_window. '
                             f'Got SWA window={explicit_sliding_window}, '
                             f'model default={default_sliding_window}.')

    use_swa = bool(dflash_config.get('use_swa', getattr(draft_hf_config, 'use_swa', False)))
    if use_swa and not has_sliding_layer:
        raise ValueError('DFlash V1 does not support use_swa forcing non-causal sliding attention; '
                         'checkpoint layer_types must describe the supported attention pattern explicitly.')

    causal_override = dflash_config.get('causal', getattr(draft_hf_config, 'causal', None))
    if causal_override is not None and not isinstance(causal_override, bool):
        raise ValueError(f'DFlash causal override must be boolean, got {causal_override!r}.')
    allowed_patterns = {(default_sliding_window, True), (None, False)}
    for layer_idx, layer_type in enumerate(layer_types):
        sliding_window = default_sliding_window if layer_type == 'sliding_attention' else None
        default_causal = layer_type == 'sliding_attention'
        causal = default_causal if causal_override is None else bool(causal_override)
        pattern = (sliding_window, causal)
        if pattern not in allowed_patterns:
            raise ValueError('DFlash V1 supports only the model-default causal attention pattern and '
                             'non-causal full attention. '
                             f'Layer {layer_idx} resolves to unsupported pattern={pattern}, '
                             f'allowed={sorted(allowed_patterns, key=str)}.')

    if dflash_config.get('attention_sink_bias', getattr(draft_hf_config, 'add_swa_attention_sink_bias', False)):
        raise ValueError('DFlash attention-sink bias checkpoints are not supported yet.')


def parse_dflash_config(draft_hf_config: Any, num_speculative_tokens: int,
                        target_num_layers: int) -> tuple[tuple[int, ...], int]:
    """Return resolved ``(target_layer_ids, mask_token_id)`` metadata."""
    num_hidden_layers = draft_hf_config.num_hidden_layers
    num_target_layers = draft_hf_config.num_target_layers
    dflash_config = draft_hf_config.dflash_config
    _validate_dflash_v1_supported(draft_hf_config, dflash_config)
    if num_target_layers != target_num_layers:
        raise ValueError('DFlash draft/target depth mismatch: '
                         f'draft declares num_target_layers={num_target_layers}, '
                         f'but target ModelConfig has num_layers={target_num_layers}.')

    max_query_length = dflash_config['block_size']
    query_length = num_speculative_tokens + 1
    if query_length > max_query_length:
        raise ValueError('DFlash query length (1 + speculative_num_draft_tokens) must not exceed checkpoint '
                         'dflash_config.block_size. '
                         f'Got block_size={max_query_length}, query_length={query_length}.')

    mask_token_id = dflash_config.get('mask_token_id')
    if mask_token_id is None:
        raise ValueError('DFlash checkpoint requires dflash_config.mask_token_id.')

    target_layer_ids = _parse_layer_ids(dflash_config.get('target_layer_ids'))
    if target_layer_ids is None:
        target_layer_ids = build_target_layer_ids(num_target_layers, num_hidden_layers)
    for pos, layer_id in enumerate(target_layer_ids):
        if layer_id < 0 or layer_id >= num_target_layers:
            raise ValueError('DFlash target_layer_ids contains an out-of-range value: '
                             f'target_layer_ids[{pos}]={layer_id}, num_target_layers={num_target_layers}.')

    return target_layer_ids, mask_token_id


def validate_dflash_dist_config(dist_config: Any):
    """Validate the initially supported DFlash distribution shape."""
    if dist_config is None:
        return
    if getattr(dist_config, 'dp', 1) != 1:
        raise ValueError('DFlash V1 does not support data parallelism. Set dp=1.')
    if getattr(dist_config, 'ep', 1) != 1:
        raise ValueError('DFlash V1 does not support expert parallel draft execution. Set ep=1.')


def validate_dflash_cache_config(cache_config: Any):
    """Validate cache options that affect DFlash draft KV correctness."""
    if cache_config is None:
        return
    if getattr(cache_config, 'enable_prefix_caching', False):
        raise ValueError('DFlash V1 does not support prefix-cache reuse yet. Disable enable_prefix_caching.')
    quant_policy = getattr(cache_config, 'quant_policy', 0)
    if int(quant_policy) != 0:
        raise ValueError('DFlash V1 does not support KV-cache quantization yet. Set quant_policy=0.')


def validate_dflash_runtime_config(cache_config: Any = None, backend_config: Any = None):
    """Validate the currently supported DFlash runtime envelope."""
    device_type = getattr(backend_config, 'device_type', None)
    if device_type is None:
        device_type = getattr(cache_config, 'device_type', 'cuda')
    if device_type != 'cuda':
        raise ValueError('DFlash V1 requires CUDA because draft context K/V materialization uses CUDA kernels. '
                         f'Got device_type={device_type!r}.')
