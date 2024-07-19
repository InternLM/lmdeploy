# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod


class ApplyRotaryEmbImpl(ABC):

    @abstractmethod
    def forward(self, query, key, cos, sin, inplace: bool = True):
        raise NotImplementedError


class ApplyRotaryEmbBuilder(ABC):

    @staticmethod
    @abstractmethod
    def build():
        raise NotImplementedError
