# Copyright (c) OpenMMLab. All rights reserved.
"""Model-facing access to named cache tensors."""

from collections.abc import Mapping, Sequence
from operator import index as as_index
from typing import TypeAlias

import torch

from .schema import CacheTensorSpec

_CacheRowLocation: TypeAlias = tuple[int, int]
"""Physical cache tensor index and row within that tensor."""


class NamedCacheView(Mapping[str, torch.Tensor]):
    """Resolve named cache tensors and their consumer/layer rows."""

    def __init__(self, caches: dict[str, torch.Tensor], rows_by_layer: dict[str, dict[int, int]] | None = None):
        self._cache_tensors: dict[str, tuple[torch.Tensor, ...]] = {
            name: (cache, )
            for name, cache in caches.items()
        }
        self._consumer_row_locations: dict[str, tuple[_CacheRowLocation, ...]] = {}
        self._layer_row_locations: dict[str, dict[int, _CacheRowLocation]] = {
            name: {
                layer_id: (0, row)
                for layer_id, row in layer_map.items()
            }
            for name, layer_map in (rows_by_layer or {}).items()
        }

    @classmethod
    def from_specs(cls, tensor_specs: Sequence[CacheTensorSpec], caches: Sequence[torch.Tensor]):
        """Build row lookup across planned cache tensors."""
        if len(tensor_specs) != len(caches):
            raise ValueError('Cache tensor specs and tensors must have the same length.')

        view = cls({})
        cache_tensors: dict[str, list[torch.Tensor]] = {}
        consumer_row_locations: dict[str, dict[int, _CacheRowLocation]] = {}
        layer_row_locations: dict[str, dict[int, _CacheRowLocation]] = {}
        for spec, cache in zip(tensor_specs, caches):
            tensors = cache_tensors.setdefault(spec.name, [])
            tensor_idx = len(tensors)
            tensors.append(cache)

            if spec.consumer_rows is not None:
                locations = consumer_row_locations.setdefault(spec.name, {})
                for local_row, consumer_row in enumerate(spec.consumer_rows):
                    locations[consumer_row] = (tensor_idx, local_row)
            elif spec.layer_rows is not None:
                locations = layer_row_locations.setdefault(spec.name, {})
                for layer_id, local_row in spec.layer_rows.row_by_layer.items():
                    locations[layer_id] = (tensor_idx, local_row)

        view._cache_tensors = {
            name: tuple(tensors)
            for name, tensors in cache_tensors.items()
        }
        view._consumer_row_locations = {
            name: tuple(locations[row] for row in range(len(locations)))
            for name, locations in consumer_row_locations.items()
        }
        view._layer_row_locations = layer_row_locations
        return view

    def __getitem__(self, name: str):
        tensors = self._cache_tensors[name]
        if len(tensors) != 1:
            accessor = 'row' if name in self._consumer_row_locations else 'layer'
            raise RuntimeError(
                f'Cache {name} has multiple physical tensors; use block_caches.{accessor}(...) to select one row.')
        return tensors[0]

    def __contains__(self, name: str):
        return name in self._cache_tensors

    def __iter__(self):
        return iter(self._cache_tensors)

    def __len__(self):
        return len(self._cache_tensors)

    def row(self, name: str, consumer_row: int):
        """Return the physical cache row bound to one built consumer."""
        try:
            locations = self._consumer_row_locations[name]
        except KeyError as e:
            raise RuntimeError(f'Cache {name} is not bound to consumer rows.') from e
        consumer_row = as_index(consumer_row)
        if consumer_row < 0 or consumer_row >= len(locations):
            raise RuntimeError(f'Consumer row {consumer_row} does not own cache {name}.')
        tensor_idx, local_row = locations[consumer_row]
        return self._cache_tensors[name][tensor_idx][local_row]

    def layer(self, name: str, layer_id: int):
        """Return a named cache row for a global layer id."""
        try:
            locations = self._layer_row_locations[name]
        except KeyError as e:
            raise RuntimeError(f'Cache {name} is not bound to model layers.') from e
        try:
            tensor_idx, local_row = locations[layer_id]
        except KeyError as e:
            raise RuntimeError(f'Layer {layer_id} does not own cache {name}.') from e
        return self._cache_tensors[name][tensor_idx][local_row]
