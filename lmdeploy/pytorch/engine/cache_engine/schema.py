# Copyright (c) OpenMMLab. All rights reserved.
"""Validated cache-tensor specifications and row membership."""

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


@dataclass(frozen=True)
class BlockCacheGeometry:
    """Finalized logical and physical block sizes passed to request
    collection."""

    block_size: int
    kernel_block_size: int

    def __post_init__(self):
        if self.block_size <= 0:
            raise ValueError('block_size must be positive.')
        if self.kernel_block_size <= 0:
            raise ValueError('kernel_block_size must be positive.')
        if self.block_size < self.kernel_block_size:
            raise ValueError(
                f'block_size {self.block_size} must be greater than or equal to '
                f'kernel_block_size {self.kernel_block_size}.')
        if self.block_size % self.kernel_block_size != 0:
            raise ValueError(
                f'block_size {self.block_size} must be divisible by '
                f'kernel_block_size {self.kernel_block_size}.')

    @property
    def kernel_blocks_per_logical_block(self) -> int:
        """Return physical kernel blocks in one scheduler block."""
        return self.block_size // self.kernel_block_size


@dataclass(frozen=True)
class BlockCacheRequest:
    """Describe one block-cache payload requested by a built operator."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    alignment: int = 256
    per_row_contiguous: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Block cache request name must not be empty.')
        alignment = as_index(self.alignment)
        shape = tuple(as_index(dim) for dim in self.shape)
        if alignment <= 0:
            raise ValueError(f'{self.name} alignment must be positive.')
        if any(dim < 0 for dim in shape):
            raise ValueError(f'{self.name} shape dimensions must be non-negative.')
        object.__setattr__(self, 'alignment', alignment)
        object.__setattr__(self, 'shape', shape)


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
    """Map global layer ids to compact cache-tensor rows."""

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
class CacheTensorSpec:
    """Describe one named cache tensor without owning its storage."""

    name: str
    desc: CacheDesc
    layer_rows: LayerRowMap | None = None
    consumer_rows: tuple[int, ...] | None = None
    per_row_contiguous: bool = False

    def __post_init__(self):
        if self.consumer_rows is None:
            return
        if self.layer_rows is not None:
            raise ValueError(f'{self.name} cannot define both consumer_rows and layer_rows.')

        consumer_rows = tuple(as_index(row) for row in self.consumer_rows)
        if not consumer_rows:
            raise ValueError(f'{self.name} consumer_rows must not be empty.')
        if any(row < 0 for row in consumer_rows):
            raise ValueError(f'{self.name} consumer rows must be non-negative.')
        if len(consumer_rows) != len(set(consumer_rows)):
            raise ValueError(f'{self.name} consumer rows must be unique within one tensor spec.')
        object.__setattr__(self, 'consumer_rows', consumer_rows)

    @property
    def has_rows(self) -> bool:
        """Whether the tensor has an explicit compact-row axis."""
        return self.consumer_rows is not None or self.layer_rows is not None

    @property
    def layer_map(self) -> dict[int, int] | None:
        """Return the global-layer-id to compact-row map if layered."""
        if self.layer_rows is None:
            return None
        return self.layer_rows.row_by_layer

    @property
    def num_rows(self) -> int:
        """Return compact rows for an operator- or layer-scoped tensor."""
        if self.consumer_rows is not None:
            return len(self.consumer_rows)
        assert self.layer_rows is not None
        return self.layer_rows.num_rows


def layer_maps_from_specs(tensor_specs: Sequence[CacheTensorSpec]) -> dict[str, dict[int, int]]:
    """Collect layer maps from named cache-tensor specs."""
    return {
        spec.name: spec.layer_map
        for spec in tensor_specs if spec.layer_map is not None
    }


def build_block_cache_tensor_specs(block_specs: Sequence[BlockCacheSpec]) -> tuple[CacheTensorSpec, ...]:
    """Normalize configured block caches into internal tensor specs."""
    tensor_specs = []
    for spec in block_specs:
        layer_rows = LayerRowMap.build(spec.name, spec.layer_ids)
        desc = CacheDesc(shape=spec.shape, dtype=spec.dtype, alignment=spec.alignment)
        tensor_specs.append(CacheTensorSpec(name=spec.name, desc=desc, layer_rows=layer_rows))
    return tuple(tensor_specs)


def build_block_cache_tensor_specs_from_requests(
        requests: Sequence[BlockCacheRequest]) -> tuple[CacheTensorSpec, ...]:
    """Group equal contracts while retaining each consumer's logical row."""
    rows_by_request: dict[BlockCacheRequest, list[int]] = {}
    next_row_by_name: dict[str, int] = {}

    for request in requests:
        row = next_row_by_name.get(request.name, 0)
        next_row_by_name[request.name] = row + 1
        rows_by_request.setdefault(request, []).append(row)

    tensor_specs = []
    for request, consumer_rows in rows_by_request.items():
        desc = CacheDesc(shape=list(request.shape), dtype=request.dtype, alignment=request.alignment)
        tensor_specs.append(
            CacheTensorSpec(name=request.name,
                            desc=desc,
                            consumer_rows=tuple(consumer_rows),
                            per_row_contiguous=request.per_row_contiguous))
    return tuple(tensor_specs)


def build_state_cache_tensor_specs(
        state_shapes: Sequence[tuple[tuple[int, ...], torch.dtype]],
        state_specs: Sequence[StateCacheSpec] | None = None) -> tuple[CacheTensorSpec, ...]:
    """Normalize named or legacy state-cache declarations."""
    state_specs = state_specs or ()
    if len(state_specs) > 0:
        tensor_specs = []
        for spec in state_specs:
            layer_rows = None
            shape = spec.shape
            if spec.layer_ids is not None:
                layer_rows = LayerRowMap.build(spec.name, spec.layer_ids)
                shape = (layer_rows.num_rows, *shape)
            desc = CacheDesc(shape=shape, dtype=spec.dtype, alignment=spec.alignment)
            tensor_specs.append(CacheTensorSpec(name=spec.name, desc=desc, layer_rows=layer_rows))
        return tuple(tensor_specs)

    return tuple(
        CacheTensorSpec(name=f'state_{idx}', desc=CacheDesc(shape=shape, dtype=dtype))
        for idx, (shape, dtype) in enumerate(state_shapes))
