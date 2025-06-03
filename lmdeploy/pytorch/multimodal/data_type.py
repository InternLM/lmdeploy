# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Union

import torch
from torch import Tensor

import lmdeploy.pytorch.distributed as dist


class MultiModalData:
    pass


MultiModalDataList = List[MultiModalData]

NestedTensor = Union[Tensor, List[Tensor]]


def _broadcast_tensor(value: torch.Tensor, src: int = 0, device: str = 'cuda'):
    """Broadcast tensor."""
    if value.device.type == 'meta':
        value = torch.empty_like(value, device=device)
    dist.broadcast(value, src)
    return value


@dataclass
class MultiModalTensor:
    data: NestedTensor
    start: int
    end: int = None
    encoder_len: int = None
    meta: Dict[str, Any] = None

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
        return MultiModalTensor(**out_dict)

    def broadcast(self):
        """Broadcast inputs tensors."""
        out_dict = dict()
        for f in fields(self):
            k = f.name
            if k in ('data', 'meta'):
                continue
            v = getattr(self, k)
            out_dict[k] = v

        if isinstance(self.data, Tensor):
            data = _broadcast_tensor(self.data)
        else:
            data = [_broadcast_tensor(d) for d in self.data]
        out_dict['data'] = data

        new_meta = None
        if self.meta is not None:
            new_meta = dict()
            for k, v in self.meta.items():
                if isinstance(v, Tensor):
                    v = _broadcast_tensor(v)
                    self.meta[k] = v
                elif hasattr(v, 'to_device'):
                    assert hasattr(v, 'broadcast')
                    v = v.broadcast()
                    self.meta[k] = v
                new_meta[k] = v

        out_dict['meta'] = new_meta
        return MultiModalTensor(**out_dict)


MultiModalInputs = Dict[str, List[MultiModalTensor]]
