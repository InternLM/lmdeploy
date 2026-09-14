# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .base import BuildSpec


class SiluAndMulImpl(ABC):
    """Silu + multiple residual fused implementation."""

    @abstractmethod
    def forward(self, x):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class SiluAndMulBuildSpec(BuildSpec[SiluAndMulImpl]):
    """Immutable requirements for constructing a SiLU-and-multiply operator."""

    inplace: bool


class GeluAndMulImpl(ABC):
    """Gelu + multiple residual fused implementation."""

    @abstractmethod
    def forward(self, x):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class GeluAndMulBuildSpec(BuildSpec[GeluAndMulImpl]):
    """Immutable requirements for constructing a GELU-and-multiply operator."""

    approximate: str
