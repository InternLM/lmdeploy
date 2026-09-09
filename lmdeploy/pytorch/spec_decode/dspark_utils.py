# Copyright (c) OpenMMLab. All rights reserved.
"""Configuration helpers shared by LMDeploy DSpark implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dflash_utils import (
    validate_dflash_cache_config,
    validate_dflash_dist_config,
    validate_dflash_runtime_config,
)

_SUPPORTED_HEADS = ('vanilla', 'gated', 'rnn')


def _get(config: Any, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _set(config: Any, name: str, value) -> None:
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def _as_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    return dict(vars(value))


def prepare_dspark_hf_config(config: Any) -> Any:
    """Normalize known DSpark checkpoint schemas for LMDeploy model builders.

    Transformers' Speculators config keeps the Qwen draft backbone under
    ``transformer_layer_config``.  LMDeploy model builders consume a flat
    config, so copy that transformer metadata onto the outer config while
    retaining DSpark-specific fields.
    """
    architectures = list(_get(config, 'architectures', None) or [])
    speculators_type = _get(config, 'speculators_model_type', None)
    is_raw = speculators_type == 'dspark' or 'DSparkDraftModel' in architectures
    nested = _get(config, 'transformer_layer_config', None)
    if is_raw and nested is not None:
        for name, value in _as_dict(nested).items():
            if name.startswith('_'):
                continue
            _set(config, name, value)
        _set(config, 'architectures', ['Qwen3DSparkModel'])
        _set(config, 'model_type', _get(nested, 'model_type', 'qwen3'))

        aux_ids = _get(config, 'aux_hidden_state_layer_ids', None)
        if aux_ids is not None:
            _set(config, 'target_layer_ids', [int(idx) - 1 for idx in aux_ids])
        _set(config, 'dflash_config', {
            'mask_token_id': _get(config, 'mask_token_id', None),
            'target_layer_ids': _get(config, 'target_layer_ids', None),
            'causal': not bool(_get(config, 'sliding_window_non_causal', True)),
        })
        # Raw Speculators checkpoints express ``block_size`` as the complete
        # verifier capacity.  The explicit field remains the source of truth
        # for query-layout selection in ``parse_dspark_config``.
        if _get(config, 'sample_from_anchor', None) is None:
            _set(config, 'sample_from_anchor', False)
    if _get(config, 'dflash_config', None) is None:
        # Native dense checkpoints such as deepseek-ai/dspark_qwen3_* expose
        # the draft attention metadata at the top level rather than under the
        # DFlash-compatible dictionary consumed by ``DFlashDraftModel``.
        # Their full-attention proposal block is non-causal by default.
        _set(config, 'dflash_config', {
            'mask_token_id': _get(config, 'mask_token_id', None),
            'target_layer_ids': _get(config, 'target_layer_ids', None),
            'causal': not bool(_get(config, 'sliding_window_non_causal', True)),
        })
    return config


@dataclass(frozen=True)
class DSparkResolvedConfig:
    target_layer_ids: tuple[int, ...]
    mask_token_id: int
    sample_from_anchor: bool
    draft_query_len: int
    verify_block_len: int
    checkpoint_block_capacity: int
    draft_vocab_size: int
    markov_rank: int
    markov_head_type: str
    bundled_draft: bool
    enable_confidence_head: bool
    confidence_head_with_markov: bool
    dynamic_verify_policy: str = 'fixed'


def _validate_layer_ids(layer_ids: Any, target_num_layers: int) -> tuple[int, ...]:
    if not isinstance(layer_ids, (list, tuple)) or not layer_ids:
        raise ValueError('DSpark requires a non-empty target-layer id list.')
    if any(isinstance(idx, bool) or not isinstance(idx, int) for idx in layer_ids):
        raise ValueError(f'DSpark target layer ids must contain integers, got {layer_ids!r}.')
    result = tuple(int(idx) for idx in layer_ids)
    if tuple(sorted(set(result))) != result:
        raise ValueError(f'DSpark target layer ids must be strictly increasing and duplicate-free, got {result}.')
    for idx in result:
        if idx < 0 or idx >= target_num_layers:
            raise ValueError(f'DSpark target layer id {idx} is outside target depth {target_num_layers}.')
    return result


def parse_dspark_config(draft_hf_config: Any, num_speculative_tokens: int,
                        target_num_layers: int) -> DSparkResolvedConfig:
    """Parse external Speculators and native dense Qwen DSpark."""
    if (isinstance(num_speculative_tokens, bool)
            or not isinstance(num_speculative_tokens, int)
            or num_speculative_tokens < 1):
        raise ValueError('DSpark num_speculative_tokens must be a positive integer.')

    bundled = False
    architectures = list(_get(draft_hf_config, 'architectures', None) or [])
    raw_speculators = (
        _get(draft_hf_config, 'speculators_model_type', None) == 'dspark'
        or _get(draft_hf_config, 'transformer_layer_config', None) is not None
        or 'DSparkDraftModel' in architectures)
    nested_dspark = _as_dict(_get(draft_hf_config, 'dspark_config', None))
    block_capacity = _get(draft_hf_config, 'block_size', None)
    mask_token_id = _get(draft_hf_config, 'mask_token_id',
                         nested_dspark.get('mask_token_id'))
    raw_aux_ids = _get(draft_hf_config, 'aux_hidden_state_layer_ids', None)
    if raw_aux_ids is not None:
        layer_ids = [int(idx) - 1 for idx in raw_aux_ids]
    else:
        layer_ids = _get(draft_hf_config, 'target_layer_ids', None)
    markov_rank = _get(draft_hf_config, 'markov_rank',
                       nested_dspark.get('markov_rank'))
    markov_head_type = _get(draft_hf_config, 'markov_head_type',
                            nested_dspark.get('markov_head_type', 'vanilla'))
    sample_from_anchor = bool(_get(
        draft_hf_config, 'sample_from_anchor',
        not bool(_get(draft_hf_config, 'dspark_bonus_anchor', False))))

    if block_capacity is None:
        raise ValueError('DSpark checkpoint requires block_size or dspark_block_size.')
    if (isinstance(block_capacity, bool)
            or not isinstance(block_capacity, int)
            or block_capacity < 1):
        raise ValueError(f'DSpark checkpoint block capacity must be positive, got {block_capacity!r}.')
    block_capacity = int(block_capacity)
    verify_block_len = int(num_speculative_tokens) + 1
    draft_query_len = (int(num_speculative_tokens)
                       if sample_from_anchor else verify_block_len)
    # Bundled/native-dense block size is gamma. Raw Speculators block size is
    # the complete trained verifier window. ``prepare_dspark_hf_config`` may
    # normalize the raw architecture name, so retain the schema markers above.
    required_capacity = (verify_block_len if raw_speculators else
                         int(num_speculative_tokens))
    if required_capacity > block_capacity:
        raise ValueError('DSpark requested block exceeds checkpoint capacity: '
                         f'required={required_capacity}, capacity={block_capacity}.')

    if mask_token_id is None:
        raise ValueError('DSpark checkpoint requires mask_token_id or dspark_noise_token_id.')
    mask_token_id = int(mask_token_id)
    vocab_size = int(_get(draft_hf_config, 'vocab_size', 0) or 0)
    if mask_token_id < 0 or (vocab_size > 0 and mask_token_id >= vocab_size):
        raise ValueError(f'DSpark mask token id {mask_token_id} is outside vocab size {vocab_size}.')

    target_layer_ids = _validate_layer_ids(layer_ids, target_num_layers)
    if markov_rank is None or isinstance(markov_rank, bool) or int(markov_rank) <= 0:
        raise ValueError(f'DSpark markov_rank must be positive, got {markov_rank!r}.')
    markov_rank = int(markov_rank)
    markov_head_type = str(markov_head_type or 'vanilla').lower()
    if markov_head_type not in _SUPPORTED_HEADS:
        raise ValueError(f'Unsupported DSpark markov_head_type={markov_head_type!r}; '
                         f'supported values are {_SUPPORTED_HEADS}.')

    draft_vocab_size = int(
        _get(draft_hf_config, 'draft_vocab_size', None)
        or nested_dspark.get('draft_vocab_size') or vocab_size)
    if draft_vocab_size <= 0:
        raise ValueError(f'DSpark draft_vocab_size must be positive, got {draft_vocab_size}.')
    enable_confidence_head = bool(_get(draft_hf_config,
                                       'enable_confidence_head', False))
    confidence_head_with_markov = bool(
        _get(draft_hf_config, 'confidence_head_with_markov', True))
    return DSparkResolvedConfig(
        target_layer_ids=target_layer_ids,
        mask_token_id=mask_token_id,
        sample_from_anchor=sample_from_anchor,
        draft_query_len=draft_query_len,
        verify_block_len=verify_block_len,
        checkpoint_block_capacity=block_capacity,
        draft_vocab_size=draft_vocab_size,
        markov_rank=markov_rank,
        markov_head_type=markov_head_type,
        bundled_draft=bundled,
        enable_confidence_head=enable_confidence_head,
        confidence_head_with_markov=confidence_head_with_markov,
    )


def validate_dspark_target_config(
    draft_hf_config: Any,
    target_hf_config: Any,
    resolved: DSparkResolvedConfig,
) -> None:
    """Validate the external/bundled draft's declared target contract."""
    target_num_layers = _get(target_hf_config, 'num_hidden_layers', None)
    declared_num_layers = _get(draft_hf_config, 'num_target_layers', None)
    if (declared_num_layers is not None and target_num_layers is not None
            and int(declared_num_layers) != int(target_num_layers)):
        raise ValueError(
            'DSpark target depth mismatch: draft declares '
            f'num_target_layers={declared_num_layers}, but target has '
            f'num_hidden_layers={target_num_layers}.')

    architectures = list(_get(draft_hf_config, 'architectures', None) or [])
    raw_speculators = (
        _get(draft_hf_config, 'speculators_model_type', None) == 'dspark'
        or _get(draft_hf_config, 'transformer_layer_config', None) is not None
        or 'DSparkDraftModel' in architectures)
    native_or_bundled = not raw_speculators

    expected_hidden_size = _get(draft_hf_config, 'target_hidden_size', None)
    if expected_hidden_size is None and native_or_bundled:
        expected_hidden_size = _get(draft_hf_config, 'hidden_size', None)
    target_hidden_size = _get(target_hf_config, 'hidden_size', None)
    if (expected_hidden_size is not None and target_hidden_size is not None
            and int(expected_hidden_size) != int(target_hidden_size)):
        raise ValueError(
            'DSpark target hidden-size mismatch: draft expects '
            f'{expected_hidden_size}, but target has {target_hidden_size}.')

    expected_vocab_size = _get(draft_hf_config, 'target_vocab_size', None)
    if expected_vocab_size is None and native_or_bundled:
        expected_vocab_size = _get(draft_hf_config, 'vocab_size', None)
    target_vocab_size = _get(target_hf_config, 'vocab_size', None)
    if (expected_vocab_size is not None and target_vocab_size is not None
            and int(expected_vocab_size) != int(target_vocab_size)):
        raise ValueError(
            'DSpark target vocabulary mismatch: draft expects '
            f'{expected_vocab_size}, but target has {target_vocab_size}.')
    if (target_vocab_size is not None
            and resolved.mask_token_id >= int(target_vocab_size)):
        raise ValueError(
            'DSpark mask token is outside the target vocabulary: '
            f'mask_token_id={resolved.mask_token_id}, '
            f'target_vocab_size={target_vocab_size}.')


def validate_dspark_dist_config(dist_config: Any):
    """DSpark V1 shares DFlash's CUDA DP1/EP1 envelope."""
    try:
        return validate_dflash_dist_config(dist_config)
    except ValueError as exc:
        raise ValueError(str(exc).replace('DFlash', 'DSpark')) from exc


def validate_dspark_cache_config(cache_config: Any):
    try:
        return validate_dflash_cache_config(cache_config)
    except ValueError as exc:
        raise ValueError(str(exc).replace('DFlash', 'DSpark')) from exc


def validate_dspark_runtime_config(cache_config: Any = None,
                                   backend_config: Any = None):
    try:
        validate_dflash_runtime_config(cache_config, backend_config)
    except ValueError as exc:
        raise ValueError(str(exc).replace('DFlash', 'DSpark')) from exc
