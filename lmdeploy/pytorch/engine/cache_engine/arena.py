# Copyright (c) OpenMMLab. All rights reserved.
"""Shared GPU arena ownership and narrow native cache projections."""

from dataclasses import dataclass

import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.config import CacheConfig, ModelConfig, SharedCacheArenaGeometry

from .layout import CacheAllocation, CachePool, PackedBlockCacheLayout, PackedStateCacheLayout
from .plan import BlockCachePlan
from .schema import build_state_cache_tensor_specs


@dataclass
class SharedCacheArena:
    """Own one group-strided byte arena for the native shared path.

    The first projection deliberately supports the packed default layout with one kernel page per scheduler block.  Its
    views are strided, so neither projection is an independent storage owner.
    """

    root: torch.Tensor
    num_groups: int
    group_size: int
    kv_block_nbytes: int

    @classmethod
    def validate_plan(cls,
                      cache_config: CacheConfig,
                      model_config: ModelConfig,
                      block_cache_plan: BlockCachePlan) -> None:
        """Validate the layout contract shared by sizing and projection."""
        if not cache_config.enable_kv_state_cache_sharing:
            return
        layout = block_cache_plan.layout
        if not isinstance(layout, PackedBlockCacheLayout):
            raise ValueError('Shared KV/state cache requires the packed default block layout.')
        if block_cache_plan.kernel_blocks_per_logical_block != 1:
            raise ValueError('Shared KV/state cache requires kernel_block_size == block_size.')
        if not layout.tensor_specs:
            raise ValueError('Shared KV/state cache requires a non-empty target KV cache.')
        if any(spec.consumer_rows is not None or spec.per_row_contiguous for spec in layout.tensor_specs):
            raise ValueError('Shared KV/state cache does not support row-scoped or contiguous operator caches.')

        state_specs = build_state_cache_tensor_specs(
            cache_config.states_shapes,
            state_specs=getattr(model_config, 'state_cache_specs', None),
        )
        if state_specs:
            state_layout = get_backend().get_cache_backend().build_state_layout(state_specs)
            if not isinstance(state_layout, PackedStateCacheLayout):
                raise ValueError('Shared KV/state cache requires the packed default state layout.')

    @classmethod
    def allocate(cls,
                 cache_config: CacheConfig,
                 model_config: ModelConfig,
                 block_cache_plan: BlockCachePlan,
                 device: torch.device | str = 'cuda') -> 'SharedCacheArena':
        """Allocate the root for finalized shared-mode capacity."""
        cls.validate_plan(cache_config, model_config, block_cache_plan)

        geometry = SharedCacheArenaGeometry.from_cache_config(cache_config, require_capacity=True)

        layout = block_cache_plan.layout
        pool_size = sum(spec.desc.aligned_size for spec in layout.tensor_specs)
        kv_block_nbytes = block_cache_plan.logical_block_nbytes
        expected_nbytes = layout.num_layers * pool_size
        if kv_block_nbytes != expected_nbytes:
            raise ValueError(
                'Shared KV/state cache requires one packed pool with a stable per-block byte footprint: '
                f'expected {expected_nbytes}, got {kv_block_nbytes}.')

        root = torch.zeros((geometry.num_groups, geometry.units_per_group * kv_block_nbytes),
                           dtype=torch.uint8,
                           device=device)
        return cls(root=root,
                   num_groups=geometry.num_groups,
                   group_size=geometry.units_per_group,
                   kv_block_nbytes=kv_block_nbytes)

    @property
    def address_span(self) -> int:
        """Return the physical KV-block span represented by the root."""
        return self.num_groups * self.group_size

    @property
    def group_stride_nbytes(self) -> int:
        """Return the byte stride between state groups."""
        return self.group_size * self.kv_block_nbytes

    def project_kv(self, block_cache_plan: BlockCachePlan) -> CacheAllocation:
        """Project packed KV pages from the root without allocating storage."""
        layout = block_cache_plan.layout
        if not isinstance(layout, PackedBlockCacheLayout):
            raise ValueError('Shared KV/state cache requires the packed default block layout.')
        if block_cache_plan.kernel_blocks_per_logical_block != 1:
            raise ValueError('Shared KV/state cache requires one kernel page per logical block.')

        pool_size = sum(spec.desc.aligned_size for spec in layout.tensor_specs)
        if layout.num_layers * pool_size != self.kv_block_nbytes:
            raise ValueError('Block-cache plan does not match the shared arena root.')

        num_blocks = self.address_span
        flat_root = self.root.reshape(num_blocks, self.kv_block_nbytes)
        pool = torch.as_strided(
            flat_root,
            size=(layout.num_layers, num_blocks, pool_size),
            stride=(pool_size, self.kv_block_nbytes, 1),
        )

        tensor_views = []
        offset = 0
        for spec in layout.tensor_specs:
            desc = spec.desc
            cache = pool[:, :, offset:offset + desc.size].view(desc.dtype)
            cache = cache.view((layout.num_layers, num_blocks, *desc.shape))
            tensor_views.append(cache)
            offset += desc.aligned_size

        return CacheAllocation(
            pools=(CachePool(pool, entry_axis=1), ),
            tensor_views=tuple(tensor_views),
            owns_storage=False,
        )

    def project_state(self, cache_config: CacheConfig, model_config: ModelConfig) -> CacheAllocation:
        """Project physical state groups from the root without allocation."""
        tensor_specs = build_state_cache_tensor_specs(
            cache_config.states_shapes,
            state_specs=getattr(model_config, 'state_cache_specs', None),
        )
        if not tensor_specs:
            return CacheAllocation(pools=(), tensor_views=(), owns_storage=False)

        backend_layout = get_backend().get_cache_backend().build_state_layout(tensor_specs)
        if not isinstance(backend_layout, PackedStateCacheLayout):
            raise ValueError('Shared KV/state cache requires the packed default state layout.')

        state_pool_size = sum(spec.desc.aligned_size for spec in tensor_specs)
        if state_pool_size > self.group_stride_nbytes:
            raise ValueError(
                'State slot does not fit in one shared arena group: '
                f'state_slot_nbytes={state_pool_size}, group_stride_nbytes={self.group_stride_nbytes}.')

        state_pool = torch.as_strided(
            self.root,
            size=(self.num_groups, state_pool_size),
            stride=(self.group_stride_nbytes, 1),
        )
        tensor_views = []
        offset = 0
        for spec in tensor_specs:
            desc = spec.desc
            cache = state_pool[:, offset:offset + desc.size].view(desc.dtype)
            cache = cache.view((self.num_groups, *desc.shape))
            if spec.layer_rows is not None:
                dims = list(range(cache.dim()))
                cache = cache.permute(1, 0, *dims[2:])
            tensor_views.append(cache)
            offset += desc.aligned_size

        return CacheAllocation(
            pools=(CachePool(state_pool, entry_axis=0), ),
            tensor_views=tuple(tensor_views),
            owns_storage=False,
        )
