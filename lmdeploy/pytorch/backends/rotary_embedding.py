# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from enum import Enum, auto


class EmbeddingType(Enum):
    """rotary embedding type."""
    Default = auto()
    LinearScaling = auto()
    DynamicNTKScaling = auto()


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
        emb_type: EmbeddingType = EmbeddingType.Default,
    ):
        """build."""
        raise NotImplementedError
