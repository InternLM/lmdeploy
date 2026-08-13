# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Sequence

import torch

from lmdeploy.pytorch.kernels.cuda.copy_packed_cache import copy_packed_cache

from ..cache_block_copy import CacheBlockCopyBuilder, CacheBlockCopyImpl


class CudaCacheBlockCopyImpl(CacheBlockCopyImpl):
    """Triton packed logical-block copy for CUDA cache pools."""

    def __init__(self, packed_caches: Sequence[torch.Tensor], pages_per_block: int):
        self.pages_per_block = pages_per_block
        self._packed_caches = tuple(packed_caches)

    def forward(self, src_block_offsets: torch.Tensor, dst_block_offsets: torch.Tensor) -> None:
        if src_block_offsets.numel() == 0:
            return

        for packed_cache in self._packed_caches:
            copy_packed_cache(packed_cache, src_block_offsets, dst_block_offsets, self.pages_per_block)


class CudaCacheBlockCopyBuilder(CacheBlockCopyBuilder):
    """Build the CUDA packed logical-block copy implementation."""

    @staticmethod
    def build(packed_caches: Sequence[torch.Tensor], num_logical_blocks: int,
              pages_per_block: int) -> CacheBlockCopyImpl:
        return CudaCacheBlockCopyImpl(packed_caches=packed_caches,
                                      pages_per_block=pages_per_block)
