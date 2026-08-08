# Copyright (c) OpenMMLab. All rights reserved.

from ...engine.cache_engine.layout import LayerRowBlockCacheLayout, PackedBlockCacheLayout, PackedStateCacheLayout
from ..cache import CacheBackend


class DefaultCacheBackend(CacheBackend):
    """Build the default cache storage layouts."""

    @classmethod
    def build_block_layout(cls, resources, num_layers: int):
        """Select the default block-cache packing."""
        resources = tuple(resources)
        layered = [resource.layer_rows is not None for resource in resources]
        if any(resource.tensor_contract.per_layer_contiguous for resource in resources):
            raise ValueError('The default cache backend does not yet support per-layer contiguous cache contracts.')
        if len(resources) > 0 and all(layered):
            return LayerRowBlockCacheLayout(resources)
        return PackedBlockCacheLayout(resources, num_layers=num_layers)

    @classmethod
    def build_state_layout(cls, resources):
        """Select the default packed state-cache layout."""
        return PackedStateCacheLayout(tuple(resources))
