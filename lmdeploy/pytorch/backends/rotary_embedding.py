# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from enum import Enum, auto


class EmbeddingType(Enum):
    Default = auto()
    LinearScaling = auto()
    DynamicNTKScaling = auto()


class RotaryEmbeddingImpl(ABC):

    @abstractmethod
    def forward(self, x, position_ids):
        raise NotImplementedError


class RotaryEmbeddingBuilder(ABC):

    @staticmethod
    @abstractmethod
    def build(
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
        emb_type: EmbeddingType = EmbeddingType.Default,
    ):
        raise NotImplementedError
