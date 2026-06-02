# Copyright (c) OpenMMLab. All rights reserved.
import enum
import hashlib
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lmdeploy.vl.constants import Modality

NestedTensor = Tensor | list[Tensor]


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
    elif isinstance(value, dict):
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


def make_multimodal_content_hash(data: Any, meta: dict[str, Any] | None,
                                 mrope_pos_ids: np.ndarray | None = None) -> str:
    """Create a stable content hash for prefix-cache multimodal matching."""
    hasher = hashlib.sha256()
    _hash_multimodal_value(hasher, data)
    _hash_multimodal_value(hasher, meta)
    _hash_multimodal_value(hasher, mrope_pos_ids)
    return hasher.hexdigest()


@dataclass
class MultiModalData:
    data: NestedTensor
    start: int
    end: int | None = None
    meta: dict[str, Any] | None = None
    modality: Modality = Modality.IMAGE

    # for qwen-vl
    mrope_pos_ids: np.ndarray | None = None

    content_hash: str | None = None

    def __post_init__(self):
        if self.end is None:
            self.end = self.start

    def to_device(self, device: str, non_blocking: bool = False):
        """To device."""
        out_dict = dict()
        for f in fields(self):
            k = f.name
            if k in ('data', 'meta'):
                continue
            v = getattr(self, k)
            out_dict[k] = v

        if isinstance(self.data, Tensor):
            data = self.data.to(device=device, non_blocking=non_blocking)
        else:
            data = [d.to(device=device, non_blocking=non_blocking) for d in self.data]
        out_dict['data'] = data

        new_meta = None
        if self.meta is not None:
            new_meta = dict()
            for k, v in self.meta.items():
                if isinstance(v, Tensor):
                    v = v.to(device=device, non_blocking=non_blocking)
                elif hasattr(v, 'to_device'):
                    v = v.to_device(device=device, non_blocking=non_blocking)
                new_meta[k] = v

        out_dict['meta'] = new_meta
        return MultiModalData(**out_dict)


MultiModalInputs = dict[str, list[MultiModalData]]


def ensure_multimodal_content_hashes(input_mms: MultiModalInputs | None):
    """Populate missing multimodal content hashes in-place."""
    if input_mms is None:
        return input_mms

    for modal_datas in input_mms.values():
        for modal_data in modal_datas:
            if modal_data.content_hash is None:
                modal_data.content_hash = make_multimodal_content_hash(modal_data.data, modal_data.meta,
                                                                       modal_data.mrope_pos_ids)
    return input_mms
