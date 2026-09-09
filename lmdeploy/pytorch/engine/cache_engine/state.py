# Copyright (c) OpenMMLab. All rights reserved.
"""State-cache allocation and slot lifecycle."""

from collections.abc import Iterator, Mapping, Sequence
from operator import index as as_index

import torch

from lmdeploy.pytorch.backends import get_backend

from ...config import CacheConfig, ModelConfig, StateCacheSpec
from .layout import CacheAllocation
from .schema import CacheTensorSpec, build_state_cache_tensor_specs
from .view import NamedCacheView


def _allocate_state_caches(tensor_specs: Sequence[CacheTensorSpec], num_caches: int,
                           device: torch.device | str) -> CacheAllocation:
    """Realize state-cache tensor specs through the selected backend layout."""
    layout = get_backend().get_cache_backend().build_state_layout(tensor_specs)
    return layout.allocate(num_caches=num_caches, device=device)


class StateCacheEngine:
    """Own state-cache allocation and state-slot transitions."""

    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig):
        self.cache_config = cache_config
        tensor_specs = build_state_cache_tensor_specs(cache_config.states_shapes,
                                                      state_specs=model_config.state_cache_specs)

        # Non-CUDA device integrations patch the canonical "cuda" device path
        # before reaching this layer, so keep using it here.
        self.allocation = _allocate_state_caches(tensor_specs,
                                                 num_caches=cache_config.num_state_caches,
                                                 device='cuda')
        self._cache_tensors = list(self.allocation.tensor_views)
        # Each pool declares the axis that indexes independently movable slots.
        self._slot_tensors = tuple((pool.tensor, pool.entry_axis) for pool in self.allocation.pools)
        if any(spec.layer_rows is not None for spec in tensor_specs):
            self._named_state_caches = NamedCacheView(tensor_specs, self._cache_tensors)
        else:
            self._named_state_caches = {
                spec.name: cache
                for spec, cache in zip(tensor_specs, self._cache_tensors)
            }

    @staticmethod
    def get_state_slot_nbytes(state_shapes: Sequence[tuple[tuple[int, ...], torch.dtype]],
                              state_specs: Sequence[StateCacheSpec] | None = None) -> int:
        """Return owning storage bytes required by one state slot."""
        tensor_specs = build_state_cache_tensor_specs(state_shapes, state_specs=state_specs)
        return _allocate_state_caches(tensor_specs, num_caches=1, device='meta').nbytes

    @property
    def state_caches(self) -> Sequence[torch.Tensor]:
        """Return state-cache tensors in model-facing order."""
        return self._cache_tensors

    @property
    def named_state_caches(self) -> Mapping[str, torch.Tensor]:
        """Return model-facing state-cache tensors keyed by semantic name."""
        return self._named_state_caches

    def zero_slots(self, slot_ids: torch.Tensor | None, zero_mask: torch.Tensor) -> None:
        """Zero the selected state slots in every physical tensor."""
        if slot_ids is None or not self._cache_tensors:
            return

        num_slots = self.cache_config.num_state_caches
        slot_mask = torch.zeros((num_slots, ), dtype=torch.bool, device=slot_ids.device)
        slot_mask.index_copy_(0, slot_ids, zero_mask)
        for tensor, slot_axis in self._slot_tensors:
            mask_shape = [1] * tensor.dim()
            mask_shape[slot_axis] = num_slots
            tensor.masked_fill_(slot_mask.view(mask_shape), 0)

    @staticmethod
    def _normalize_slot_ids(slot_ids: int | Sequence[int]) -> list[int]:
        """Normalize one or more host-side state-slot ids."""
        if isinstance(slot_ids, torch.Tensor):
            raise TypeError('State slot ids must be host integers, not torch.Tensor.')
        if isinstance(slot_ids, (str, bytes)):
            raise TypeError('State slot ids must be an int or a sequence of ints.')
        try:
            return [as_index(slot_ids)]
        except TypeError:
            pass
        if not isinstance(slot_ids, Sequence):
            raise TypeError('State slot ids must be an int or a sequence of ints.')
        if any(isinstance(slot_id, torch.Tensor) for slot_id in slot_ids):
            raise TypeError('State slot ids must be host integers, not torch.Tensor.')
        return [as_index(slot_id) for slot_id in slot_ids]

    @staticmethod
    def _validate_slot_ids(slot_ids: Sequence[int], num_slots: int) -> None:
        """Check that normalized state-slot ids index allocated storage."""
        for slot_id in slot_ids:
            if slot_id < 0 or slot_id >= num_slots:
                raise ValueError(f'State slot {slot_id} is out of range [0, {num_slots}).')

    @staticmethod
    def _coalesce_copy_ranges(src_slots: list[int], dst_slots: list[int]) -> Iterator[tuple[int, int, int]]:
        """Yield contiguous copy ranges as ``(src_start, dst_start,
        length)``."""
        pairs = sorted(zip(src_slots, dst_slots))
        if len(pairs) == 0:
            return
        start_src = prev_src = pairs[0][0]
        start_dst = prev_dst = pairs[0][1]
        length = 1
        for src, dst in pairs[1:]:
            if src == prev_src + 1 and dst == prev_dst + 1:
                prev_src = src
                prev_dst = dst
                length += 1
                continue
            yield start_src, start_dst, length
            start_src = prev_src = src
            start_dst = prev_dst = dst
            length = 1
        yield start_src, start_dst, length

    def copy_slots(self, src_slots: int | Sequence[int], dst_slots: int | Sequence[int]) -> None:
        """Copy non-overlapping state slots across every physical tensor."""
        if not self._cache_tensors:
            return

        src_slots = self._normalize_slot_ids(src_slots)
        dst_slots = self._normalize_slot_ids(dst_slots)
        if len(src_slots) != len(dst_slots):
            raise ValueError('src_slots and dst_slots must have the same number of elements.')
        if len(src_slots) == 0:
            return

        num_slots = self.cache_config.num_state_caches
        self._validate_slot_ids(src_slots, num_slots)
        self._validate_slot_ids(dst_slots, num_slots)
        if len(set(dst_slots)) != len(dst_slots):
            raise ValueError('dst_slots must not contain duplicate entries.')
        if not set(src_slots).isdisjoint(dst_slots):
            raise ValueError('src_slots and dst_slots must not overlap for stream-ordered state copies.')

        copy_ranges = tuple(self._coalesce_copy_ranges(src_slots, dst_slots))
        for tensor, slot_axis in self._slot_tensors:
            for src, dst, length in copy_ranges:
                src_tensor = tensor.narrow(slot_axis, src, length)
                dst_tensor = tensor.narrow(slot_axis, dst, length)
                dst_tensor.copy_(src_tensor, non_blocking=True)
