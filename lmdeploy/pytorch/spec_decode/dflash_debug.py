# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import torch


def dflash_debug_dir() -> Path | None:
    """Return the optional DFlash debug output directory."""
    path = os.getenv('LMDEPLOY_DFLASH_DEBUG_DIR')
    return None if not path else Path(path)


def dflash_debug_enabled() -> bool:
    """Return whether opt-in DFlash tracing is enabled."""
    return dflash_debug_dir() is not None


def debug_tensor(value: torch.Tensor | None, limit: int = 64):
    """Serialize a small tensor, truncating large tensors."""
    if value is None:
        return None
    tensor = value.detach().cpu()
    if tensor.numel() <= limit:
        return tensor.tolist()
    return {
        'shape': list(tensor.shape),
        'head': tensor.flatten()[:limit].tolist(),
    }


def write_dflash_debug(rank: int, event: str, payload: dict | Callable[[], dict]):
    """Append one DFlash debug event when opt-in tracing is enabled."""
    debug_dir = dflash_debug_dir()
    if debug_dir is None:
        return
    if callable(payload):
        payload = payload()
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f'rank{rank}.jsonl'
    record = {'event': event, **payload}
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, sort_keys=True) + '\n')
