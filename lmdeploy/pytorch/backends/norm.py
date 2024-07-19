# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch


class RMSNormImpl(ABC):

    @abstractmethod
    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        raise NotImplementedError


class RMSNormBuilder(ABC):

    @staticmethod
    @abstractmethod
    def build(weight: torch.Tensor, eps: float = 1e-6):
        raise NotImplementedError
