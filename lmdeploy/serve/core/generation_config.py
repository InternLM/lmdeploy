# Copyright (c) OpenMMLab. All rights reserved.
"""Server-side generation config resolution and sampling parameter merge
helpers."""

from __future__ import annotations

from typing import Any

from lmdeploy.messages import GenerationConfig
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

PROTOCOL_FALLBACKS: dict[str, Any] = {
    'temperature': 0.7,
    'top_p': 1.0,
    'top_k': 40,
    'repetition_penalty': 1.0,
    'min_p': 0.0,
    'do_sample': True,
}

REQUEST_SAMPLING_FIELDS = (
    'temperature',
    'top_p',
    'top_k',
    'min_p',
    'repetition_penalty',
)


def _load_hf_generation_config(path: str, trust_remote_code: bool) -> dict[str, Any]:
    from transformers import GenerationConfig

    try:
        cfg = GenerationConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
        return cfg.to_diff_dict()
    except OSError:
        return {}


def resolve_default_gen_config(
    generation_config: str,
    model_path: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    """Resolve server-side default generation config from CLI flags."""
    src = generation_config

    if src == 'lmdeploy':
        config: dict[str, Any] = {}
    elif src == 'auto':
        config = _load_hf_generation_config(model_path, trust_remote_code)
    else:
        config = _load_hf_generation_config(src, trust_remote_code)

    if config and src != 'lmdeploy':
        source = "the model's `generation_config.json`" if src == 'auto' else src
        logger.info(
            f'Using default generation config from {source}: {config}. '
            'Use `--generation-config lmdeploy` to disable.')

    return config


def merge_sampling_params(
    request_values: dict[str, Any],
    default_gen_config: dict[str, Any],
    fallbacks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge sampling params with request > default_gen_config > protocol
    fallback priority."""
    fallbacks = fallbacks or PROTOCOL_FALLBACKS
    merged: dict[str, Any] = {}
    all_keys = set(fallbacks) | set(default_gen_config) | set(request_values)
    for key in all_keys:
        if key in request_values:
            merged[key] = request_values[key]
        elif key in default_gen_config:
            merged[key] = default_gen_config[key]
        elif key in fallbacks:
            merged[key] = fallbacks[key]
    return merged


def extract_request_sampling_values(request: Any) -> dict[str, Any]:
    """Extract explicitly provided sampling fields from a request object."""
    values: dict[str, Any] = {}
    for field in REQUEST_SAMPLING_FIELDS:
        if not hasattr(request, field):
            continue
        value = getattr(request, field)
        if value is not None:
            values[field] = value
    return values


def build_generation_config(
    request_values: dict[str, Any],
    default_gen_config: dict[str, Any],
    *,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    fallbacks: dict[str, Any] | None = None,
    **extra_kwargs: Any,
) -> GenerationConfig:
    """Build ``GenerationConfig`` from merged sampling defaults and request
    values."""
    merged = merge_sampling_params(request_values, default_gen_config, fallbacks)
    max_new_tokens = max_completion_tokens if max_completion_tokens is not None else max_tokens
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=merged.get('do_sample', PROTOCOL_FALLBACKS['do_sample']),
        top_k=merged['top_k'],
        top_p=merged['top_p'],
        temperature=merged['temperature'],
        repetition_penalty=merged['repetition_penalty'],
        min_p=merged['min_p'],
        **extra_kwargs,
    )
