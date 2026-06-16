# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch


class BaseHcSplitSinkhorn(ABC):

    @abstractmethod
    def forward(
        self,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class BaseHcSplitSinkhornBuilder:

    @staticmethod
    @abstractmethod
    def build(hc_mult: int, sinkhorn_iters: int, eps: float) -> BaseHcSplitSinkhorn:
        raise NotImplementedError
