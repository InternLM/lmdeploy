# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
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
        """to device."""
        if isinstance(self.data, Tensor):
            self.data = self.data.to(device=device, non_blocking=non_blocking)
        else:
            data = [
                d.to(device=device, non_blocking=non_blocking)
                for d in self.data
            ]
            self.data = data

        if self.meta is not None:
            for k, v in self.meta.items():
                if isinstance(v, Tensor):
                    v = v.to(device=device, non_blocking=non_blocking)
                    self.meta[k] = v
                elif hasattr(v, 'to_device'):
                    v = v.to_device(device=device, non_blocking=non_blocking)
                    self.meta[k] = v
        return self


MultiModalInputs = Dict[str, List[MultiModalTensor]]
