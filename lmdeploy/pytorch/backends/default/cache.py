# Copyright (c) OpenMMLab. All rights reserved.

from itertools import groupby

from ...engine.cache_engine.layout import (
    CompositeBlockCacheLayout,
    ContiguousBlockCacheLayout,
    PackedBlockCacheLayout,
    PackedStateCacheLayout,
    RowBlockCacheLayout,
)
from ..cache import CacheBackend


class DefaultCacheBackend(CacheBackend):
    """Build the default cache storage layouts."""

    @classmethod
    def build_block_layout(cls, resources, num_layers: int):
        """Select the default block-cache packing."""
        resources = tuple(resources)
        if not resources:
            return PackedBlockCacheLayout(resources, num_layers=num_layers)

        def layout_kind(resource):
            if resource.per_row_contiguous:
                return 'contiguous'
            if resource.has_rows:
                return 'rows'
            return 'packed'

        layouts = []
        for kind, group in groupby(resources, key=layout_kind):
            group = tuple(group)
            if kind == 'contiguous':
                layout = ContiguousBlockCacheLayout(group, num_layers=num_layers)
            elif kind == 'rows':
                layout = RowBlockCacheLayout(group)
            else:
                layout = PackedBlockCacheLayout(group, num_layers=num_layers)
            layouts.append(layout)

        if len(layouts) == 1:
            return layouts[0]
        return CompositeBlockCacheLayout(tuple(layouts))

    @classmethod
    def build_state_layout(cls, resources):
        """Select the default packed state-cache layout."""
        return PackedStateCacheLayout(tuple(resources))
