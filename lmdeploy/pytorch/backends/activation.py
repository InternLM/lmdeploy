# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod


class SiluAndMulImpl(ABC):

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


class SiluAndMulBuilder(ABC):

    @staticmethod
    @abstractmethod
    def build(inplace: bool = False):
        raise NotImplementedError
