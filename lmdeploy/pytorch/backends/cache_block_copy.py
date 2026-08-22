# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .base import BuildSpec


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


@dataclass(frozen=True)
class CacheBlockCopyBuildSpec(BuildSpec[CacheBlockCopyImpl]):
    """Stable packed pools and geometry for logical-block copying."""

    packed_caches: tuple[torch.Tensor, ...]
    num_logical_blocks: int
    pages_per_block: int
