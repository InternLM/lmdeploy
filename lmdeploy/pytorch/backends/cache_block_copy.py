# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class CacheBlockCopyImpl(ABC):
    """Logical-block copy implementation owning its packed cache pools."""

    @abstractmethod
    def forward(self, src_block_offsets: torch.Tensor, dst_block_offsets: torch.Tensor) -> None:
        """Copy complete logical blocks on the current stream.

        Both offset tensors are one-dimensional ``torch.long`` device tensors
        at scheduler block granularity. The caller owns plan semantic
        validation and lifetime.
        """
        raise NotImplementedError


class CacheBlockCopyBuilder(ABC):
    """Logical-block copy implementation builder."""

    @staticmethod
    @abstractmethod
    def build(packed_caches: Sequence[torch.Tensor], num_logical_blocks: int,
              pages_per_block: int) -> CacheBlockCopyImpl:
        """Build an implementation for stable logical/cache-page geometry."""
        raise NotImplementedError
