# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class RopeType(Enum):
    """rotary embedding type."""
    Default = auto()
    LinearScaling = auto()
    DynamicNTKScaling = auto()
    Llama3 = auto()
    Yarn = auto()
    LongRoPEScaling = auto()


@dataclass
class YarnParameters:
    """Yarn parameters."""
    beta_fast: int = 32
    beta_slow: float = 1
    mscale: int = 1
    mscale_all_dim: int = 0


@dataclass
class LongRoPEScalingParameters:
    """Long Ropescaling parameters."""
    short_factor: List[int]
    long_factor: List[int]
    original_max_position_embeddings: int
    long_mscale: float = None
    short_mscale: float = None


@dataclass
class Llama3Parameters:
    """llama3 rope parameters."""
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0


class RotaryEmbeddingImpl(ABC):
    """rotary embedding implementation api."""

    @abstractmethod
    def forward(self, x, position_ids):
        """forward."""
        raise NotImplementedError


class RotaryEmbeddingBuilder(ABC):
    """rotary embedding implementation builder."""

    @staticmethod
    @abstractmethod
    def build(
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
        yarn_params: YarnParameters = None,
        longrope_params: LongRoPEScalingParameters = None,
        llama3_params: Llama3Parameters = None,
        emb_type: RopeType = RopeType.Default,
    ):
        """build."""
        raise NotImplementedError
