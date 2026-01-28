# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist


class EmbeddingImpl(ABC):
    """Embedding implementation api."""

    @abstractmethod
    def forward(self, x, weight: torch.Tensor, all_reduce: bool = False, group: dist.ProcessGroup = None):
        """forward."""
        raise NotImplementedError


class EmbeddingBuilder(ABC):
    """Embedding implementation builder."""

    @staticmethod
    @abstractmethod
    def build(start_index: int, end_index: int):
        """build."""
        raise NotImplementedError
