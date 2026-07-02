# Copyright (c) OpenMMLab. All rights reserved.
"""Server-side generation config resolution and sampling parameter merge
helpers."""

from __future__ import annotations

import dataclasses
from typing import Any

from lmdeploy.messages import GenerationConfig
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


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


def merge_gen_config(
    request_values: dict[str, Any],
    default_gen_config: dict[str, Any],
) -> dict[str, Any]:
    """Merge generation config with request > default_gen_config priority."""
    merged: dict[str, Any] = {}
    for key in set(default_gen_config) | set(request_values):
        if key in request_values:
            merged[key] = request_values[key]
        else:
            merged[key] = default_gen_config[key]
    return merged


def extract_request_gen_config(request: Any) -> dict[str, Any]:
    """Extract non-None GenerationConfig fields present on the request."""
    values: dict[str, Any] = {}
    for field in dataclasses.fields(GenerationConfig):
        if not hasattr(request, field.name):
            continue
        value = getattr(request, field.name)
        if value is not None:
            values[field.name] = value
    return values


def build_generation_config(
    request: Any,
    default_gen_config: dict[str, Any],
    *,
    max_new_tokens: int | None = None,
    **extra_kwargs: Any,
) -> GenerationConfig:
    """Build ``GenerationConfig`` from merged sampling defaults and request
    values."""
    request_values = extract_request_gen_config(request)
    merged = merge_gen_config(request_values, default_gen_config)
    merged.pop('max_new_tokens', None)
    merged.pop('do_sample', None)
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        **merged,
        **extra_kwargs,
    )
