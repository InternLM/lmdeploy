# Copyright (c) OpenMMLab. All rights reserved.
"""Validated cache resource descriptions and layer membership."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
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
    """Finalized logical and physical block sizes passed to providers."""

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
class CacheTensorContract:
    """Kernel-visible tensor requirements declared by a built operator."""

    per_layer_contiguous: bool = False


@dataclass(frozen=True)
class BlockCacheRequest:
    """Describe one block-cache payload requested by a built operator."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    alignment: int = 256
    tensor_contract: CacheTensorContract = field(default_factory=CacheTensorContract)

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


@dataclass(frozen=True)
class ScopedBlockCacheRequest:
    """Bind one operator request to a model-owned layer scope.

    ``None`` is reserved for a future whole-model scope.
    """

    request: BlockCacheRequest
    layer_id: int | None


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
    tensor_contract: CacheTensorContract = field(default_factory=CacheTensorContract)

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


def build_block_cache_resources_from_requests(
        scoped_requests: Sequence[ScopedBlockCacheRequest]) -> tuple[CacheResource, ...]:
    """Aggregate built-operator requests into ordered cache resources."""
    requests_by_scope: dict[tuple[str, int], BlockCacheRequest] = {}
    request_by_name: dict[str, BlockCacheRequest] = {}
    layer_ids_by_name: dict[str, list[int]] = {}

    for scoped_request in scoped_requests:
        request = scoped_request.request
        layer_id = scoped_request.layer_id
        if layer_id is None:
            raise ValueError('Global block cache requests are not supported yet.')
        layer_id = as_index(layer_id)
        if layer_id < 0:
            raise ValueError(f'{request.name} layer id {layer_id} must be non-negative.')

        scope = (request.name, layer_id)
        existing = requests_by_scope.get(scope)
        if existing is not None:
            if existing != request:
                raise ValueError(f'Conflicting block cache requests for {request.name} at layer {layer_id}.')
            continue

        existing = request_by_name.get(request.name)
        if existing is not None and existing != request:
            raise ValueError(f'Heterogeneous block cache request segments for {request.name} are not supported yet.')

        if request.name not in layer_ids_by_name:
            layer_ids_by_name[request.name] = [layer_id]
            request_by_name[request.name] = request
        else:
            layer_ids_by_name[request.name].append(layer_id)
        requests_by_scope[scope] = request

    resources = []
    for name, request in request_by_name.items():
        layer_ids = layer_ids_by_name[name]
        layer_rows = LayerRowMap.build(name, layer_ids)
        desc = CacheDesc(shape=list(request.shape), dtype=request.dtype, alignment=request.alignment)
        resources.append(
            CacheResource(name=name,
                          desc=desc,
                          layer_rows=layer_rows,
                          tensor_contract=request.tensor_contract))
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
