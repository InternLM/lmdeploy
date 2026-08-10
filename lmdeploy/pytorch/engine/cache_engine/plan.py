# Copyright (c) OpenMMLab. All rights reserved.
"""Block-cache plan construction and its finalized allocation recipe."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from lmdeploy.pytorch.backends import get_backend

from ...config import CacheConfig, ModelConfig
from .layout import BlockCacheLayout, CacheAllocation
from .schema import (
    BlockCacheGeometry,
    BlockCacheRequest,
    CacheTensorSpec,
    build_model_block_cache_tensor_specs,
)


@dataclass(frozen=True)
class BlockCachePlan:
    """Reuse one finalized block-cache layout across sizing and allocation."""

    tensor_specs: tuple[CacheTensorSpec, ...]
    layout: BlockCacheLayout
    kernel_blocks_per_logical_block: int

    def __post_init__(self):
        if self.kernel_blocks_per_logical_block <= 0:
            raise ValueError('kernel blocks per logical block must be positive.')

        row_kind_by_name = {}
        row_ids_by_name = {}
        for spec in self.tensor_specs:
            if spec.consumer_rows is not None:
                row_kind = 'consumer'
                row_ids = spec.consumer_rows
            elif spec.layer_rows is not None:
                row_kind = 'layer'
                row_ids = spec.layer_rows.layer_ids
            else:
                row_kind = 'plain'
                row_ids = ()

            existing_kind = row_kind_by_name.get(spec.name)
            if existing_kind is not None:
                if row_kind == 'plain' or existing_kind != row_kind:
                    raise ValueError(
                        f'Block cache {spec.name} cannot mix {existing_kind} and {row_kind} tensor specs.')
            else:
                row_kind_by_name[spec.name] = row_kind

            seen_rows = row_ids_by_name.setdefault(spec.name, set())
            overlap = seen_rows.intersection(row_ids)
            if overlap:
                row = min(overlap)
                raise ValueError(f'Block cache {spec.name} row {row} belongs to multiple tensor specs.')
            seen_rows.update(row_ids)

        for name, row_kind in row_kind_by_name.items():
            if row_kind != 'consumer':
                continue
            rows = row_ids_by_name[name]
            if rows != set(range(len(rows))):
                raise ValueError(f'Block cache {name} consumer rows must be contiguous from zero.')

    @property
    def cache_names(self) -> tuple[str, ...]:
        """Return tensor names in model-facing cache order."""
        return tuple(spec.name for spec in self.tensor_specs)

    @property
    def model_cache_indices(self) -> tuple[int, ...]:
        """Return tensors exposed through the per-layer model cache."""
        return tuple(index for index, spec in enumerate(self.tensor_specs) if not spec.has_rows)

    def allocate(self, num_logical_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Realize a logical block count through the selected physical
        layout."""
        num_kernel_blocks = num_logical_blocks * self.kernel_blocks_per_logical_block
        return self.layout.allocate(num_blocks=num_kernel_blocks, device=device)

    @property
    def logical_block_nbytes(self) -> int:
        """Return owning storage bytes required by one logical block."""
        return self.allocate(num_logical_blocks=1, device='meta').nbytes


def build_block_cache_plan(
    model_config: ModelConfig,
    cache_config: CacheConfig,
    world_size: int,
    geometry: BlockCacheGeometry,
    block_requests: Sequence[BlockCacheRequest] | None = None,
) -> BlockCachePlan:
    """Build one plan from finalized model, cache, and request inputs."""
    tensor_specs = build_model_block_cache_tensor_specs(model_config,
                                                        cache_config,
                                                        world_size,
                                                        block_requests=block_requests)
    layout = get_backend().get_cache_backend().build_block_layout(tensor_specs,
                                                                  num_layers=model_config.num_layers)
    return BlockCachePlan(
        tensor_specs=tensor_specs,
        layout=layout,
        kernel_blocks_per_logical_block=geometry.kernel_blocks_per_logical_block,
    )
