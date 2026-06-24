# Copyright (c) OpenMMLab. All rights reserved.
import enum
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor

_POSITION_KEYS = {
    'content_hash',
    'offset',
    'start',
    'end',
    'token_begin',
    'token_end',
}


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
    """Create a stable content hash for prefix-cache multimodal matching."""
    hasher = hashlib.sha256()
    _hash_multimodal_value(hasher, data)
    _hash_multimodal_value(hasher, meta)
    _hash_multimodal_value(hasher, mrope_pos_ids)
    return hasher.hexdigest()


def ensure_multimodal_content_hashes(input_mms):
    """Populate missing ``content_hash`` values on PyTorch multimodal data."""
    if input_mms is None:
        return input_mms

    for modal_datas in input_mms.values():
        for modal_data in modal_datas:
            if modal_data.content_hash is None:
                modal_data.content_hash = make_multimodal_content_hash(modal_data.data, modal_data.meta,
                                                                       modal_data.mrope_pos_ids)
    return input_mms


def make_multimodal_item_content_hash(item: Mapping[str, Any]) -> str:
    """Create a stable content hash for dict-style multimodal items.

    Prompt positions stay outside the content hash so backends can combine the
    same content identity with block-relative offsets when building cache keys.
    """
    content_view = {key: value for key, value in item.items() if key not in _POSITION_KEYS}
    hasher = hashlib.sha256()
    _hash_multimodal_value(hasher, content_view)
    return hasher.hexdigest()


def ensure_multimodal_item_content_hashes(items: list[dict[str, Any]] | None):
    """Populate missing ``content_hash`` values on dict-style multimodal data."""
    if not items:
        return items

    for item in items:
        if item.get('content_hash') is None:
            item['content_hash'] = make_multimodal_item_content_hash(item)
    return items
