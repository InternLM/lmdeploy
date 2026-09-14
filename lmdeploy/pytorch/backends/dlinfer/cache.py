# Copyright (c) OpenMMLab. All rights reserved.
"""Contiguous cache layout selection for dlinfer device backends."""

from ...engine.cache_engine.layout import ContiguousBlockCacheLayout, ContiguousStateCacheLayout
from ..default.cache import DefaultCacheBackend


class DlinferCacheBackend(DefaultCacheBackend):
    """Select dlinfer-compatible contiguous cache layouts.

    These layouts make dlinfer's external CacheEngine and StateCacheEngine allocator patches unnecessary.
    """

    @classmethod
    def build_block_layout(cls, tensor_specs, num_layers: int):
        """Select independent contiguous block-cache tensors."""
        return ContiguousBlockCacheLayout(tuple(tensor_specs), num_layers=num_layers)

    @classmethod
    def build_state_layout(cls, tensor_specs):
        """Select independent contiguous state-cache tensors."""
        return ContiguousStateCacheLayout(tuple(tensor_specs))
