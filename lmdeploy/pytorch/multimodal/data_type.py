# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Union

from torch import Tensor


class MultiModalData:
    pass


MultiModalDataList = List[MultiModalData]

NestedTensor = Union[Tensor, List[Tensor]]


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


MultiModalInputs = Dict[str, List[MultiModalTensor]]
