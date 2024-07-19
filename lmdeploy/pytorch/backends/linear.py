# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

from torch import nn


class LinearImpl(ABC, nn.Module):

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


class LinearBuilder(ABC):

    @staticmethod
    @abstractmethod
    def build(mod: nn.Module, all_reduce: bool = False):
        raise NotImplementedError
