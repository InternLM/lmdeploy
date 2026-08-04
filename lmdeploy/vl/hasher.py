# Copyright (c) OpenMMLab. All rights reserved.
import enum
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor


def _hash_multimodal_value(hasher: 'hashlib._Hash', value: Any):
    """Update a hash with a deterministic multimodal value representation."""
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(f'tensor:{tensor.dtype}:{tuple(tensor.shape)}:'.encode())
        hasher.update(tensor.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(f'ndarray:{array.dtype}:{array.shape}:'.encode())
        hasher.update(array.tobytes())
    elif isinstance(value, Mapping):
        hasher.update(b'dict:{')
        for key in sorted(value, key=lambda x: repr(x)):
            _hash_multimodal_value(hasher, key)
            hasher.update(b':')
            _hash_multimodal_value(hasher, value[key])
            hasher.update(b',')
        hasher.update(b'}')
    elif isinstance(value, (list, tuple)):
        hasher.update(f'{type(value).__name__}:['.encode())
        for item in value:
            _hash_multimodal_value(hasher, item)
            hasher.update(b',')
        hasher.update(b']')
    elif isinstance(value, enum.Enum):
        _hash_multimodal_value(hasher, value.value)
    else:
        hasher.update(f'{type(value).__name__}:{repr(value)}'.encode())


def make_multimodal_content_hash(data: Any,
                                 meta: Mapping[str, Any] | None = None,
                                 mrope_pos_ids: np.ndarray | None = None) -> str:
    """Create a stable content hash for multimodal content identity."""
    hasher = hashlib.sha256()
    _hash_multimodal_value(hasher, data)
    _hash_multimodal_value(hasher, meta)
    _hash_multimodal_value(hasher, mrope_pos_ids)
    return hasher.hexdigest()
