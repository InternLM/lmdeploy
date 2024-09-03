# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod


class SiluAndMulImpl(ABC):
    """silu + multiple residual fused implementation."""

    @abstractmethod
    def forward(self, x):
        """forward."""
        raise NotImplementedError


class SiluAndMulBuilder(ABC):
    """silu and mul implementation builder."""

    @staticmethod
    @abstractmethod
    def build(inplace: bool = False):
        """build."""
        raise NotImplementedError


class GeluAndMulImpl(ABC):
    """gelu + multiple residual fused implementation."""

    @abstractmethod
    def forward(self, x):
        """forward."""
        raise NotImplementedError


class GeluAndMulBuilder(ABC):
    """gelu and mul implementation builder."""

    @staticmethod
    @abstractmethod
    def build(approximate: str = 'none'):
        """build."""
        raise NotImplementedError
