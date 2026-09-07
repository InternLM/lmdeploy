# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .base import BuildSpec


class EmbeddingImpl(ABC):
    """Embedding implementation api."""

    @abstractmethod
    def forward(self, x, weight: torch.Tensor, all_reduce: bool = False, group: dist.ProcessGroup = None):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class EmbeddingBuildSpec(BuildSpec[EmbeddingImpl]):
    """Immutable requirements for constructing an embedding operator."""

    start_index: int
    end_index: int
