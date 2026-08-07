# Copyright (c) OpenMMLab. All rights reserved.
"""Validated cache resource descriptions and layer membership."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from operator import index as as_index

import torch

from ...config import BlockCacheSpec, StateCacheSpec


def round_up(x: int, alignment: int) -> int:
    """Round up x to the nearest multiple of alignment."""
    return ((x + alignment - 1) // alignment) * alignment


@dataclass
class CacheDesc:
    """Describe one cache payload and its aligned storage size."""
    shape: list[int]
    dtype: torch.dtype
    alignment: int = 256

    def __post_init__(self):
        self.numel = math.prod(self.shape)
        self.size = self.numel * self.dtype.itemsize
        self.aligned_size = round_up(self.size, self.alignment)


def _normalize_cache_layer_ids(cache_name: str, layer_ids: Sequence[int]) -> list[int]:
    """Validate structural layer-id invariants and return host integers."""
    normalized = []
    seen = set()
    for layer_id in layer_ids:
        layer_id = as_index(layer_id)
        if layer_id < 0:
            raise ValueError(f'{cache_name} layer id {layer_id} must be non-negative.')
        if layer_id in seen:
            raise ValueError(f'{cache_name} layer id {layer_id} is duplicated.')
        seen.add(layer_id)
        normalized.append(layer_id)
    if len(normalized) == 0:
        raise ValueError(f'{cache_name} layer_ids must not be empty.')
    return normalized


@dataclass(frozen=True)
class LayerRowMap:
    """Map global layer ids to compact resource rows."""

    layer_ids: tuple[int, ...]
    row_by_layer: dict[int, int]

    @classmethod
    def build(cls, cache_name: str, layer_ids: Sequence[int]):
        layer_ids = tuple(_normalize_cache_layer_ids(cache_name, layer_ids))
        row_by_layer = {
            layer_id: cache_row
            for cache_row, layer_id in enumerate(layer_ids)
        }
        return cls(layer_ids=layer_ids, row_by_layer=row_by_layer)

    @property
    def num_rows(self) -> int:
        """Number of compact cache rows required by this layout."""
        return len(self.layer_ids)


@dataclass(frozen=True)
class CacheResource:
    """Pair one named cache payload with its optional layer rows."""

    name: str
    desc: CacheDesc
    layer_rows: LayerRowMap | None = None

    @property
    def layer_map(self) -> dict[int, int] | None:
        """Return the global-layer-id to compact-row map if layered."""
        if self.layer_rows is None:
            return None
        return self.layer_rows.row_by_layer

    @property
    def num_rows(self) -> int:
        """Return compact layer rows for layered resources."""
        assert self.layer_rows is not None
        return self.layer_rows.num_rows


def layer_maps_from_resources(resources: Sequence[CacheResource]) -> dict[str, dict[int, int]]:
    """Collect layer maps from named resources."""
    return {
        resource.name: resource.layer_map
        for resource in resources if resource.layer_map is not None
    }


def build_block_cache_resources(block_specs: Sequence[BlockCacheSpec]) -> tuple[CacheResource, ...]:
    """Normalize named block-cache specs into validated resources."""
    resources = []
    for spec in block_specs:
        layer_rows = LayerRowMap.build(spec.name, spec.layer_ids)
        desc = CacheDesc(shape=spec.shape, dtype=spec.dtype, alignment=spec.alignment)
        resources.append(CacheResource(name=spec.name, desc=desc, layer_rows=layer_rows))
    return tuple(resources)


def build_state_cache_resources(state_shapes: Sequence[tuple[tuple[int, ...], torch.dtype]],
                                state_specs: Sequence[StateCacheSpec] | None = None) -> tuple[CacheResource, ...]:
    """Normalize named or legacy state-cache declarations."""
    state_specs = state_specs or ()
    if len(state_specs) > 0:
        resources = []
        for spec in state_specs:
            layer_rows = None
            shape = spec.shape
            if spec.layer_ids is not None:
                layer_rows = LayerRowMap.build(spec.name, spec.layer_ids)
                shape = (layer_rows.num_rows, *shape)
            desc = CacheDesc(shape=shape, dtype=spec.dtype, alignment=spec.alignment)
            resources.append(CacheResource(name=spec.name, desc=desc, layer_rows=layer_rows))
        return tuple(resources)

    return tuple(
        CacheResource(name=f'state_{idx}', desc=CacheDesc(shape=shape, dtype=dtype))
        for idx, (shape, dtype) in enumerate(state_shapes))
