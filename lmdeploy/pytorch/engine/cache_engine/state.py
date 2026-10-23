# Copyright (c) OpenMMLab. All rights reserved.
"""State-cache allocation and slot lifecycle."""

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from operator import index as as_index

import torch

from lmdeploy.pytorch.backends import get_backend

from ...config import CacheConfig, CompactStateCapacity, ModelConfig, StateCacheSpec
from ...kv_connector.base import KVCachePool
from .layout import CacheAllocation, CachePool, PackedStateCacheLayout
from .schema import CacheTensorSpec, build_state_cache_tensor_specs
from .view import NamedCacheView


def _allocate_state_caches(tensor_specs: Sequence[CacheTensorSpec], num_caches: int,
                           device: torch.device | str) -> CacheAllocation:
    """Realize state-cache tensor specs through the selected backend layout."""
    layout = get_backend().get_cache_backend().build_state_layout(tensor_specs)
    return layout.allocate(num_caches=num_caches, device=device)


def _allocate_state_cache_groups(tensor_specs: Sequence[CacheTensorSpec],
                                 state_specs: Sequence[StateCacheSpec] | None,
                                 num_caches: int,
                                 device: torch.device | str) -> tuple[CacheAllocation, tuple[CachePool, ...]]:
    """Keep runtime-only state out of physical checkpoint-copy pools."""
    checkpoint_flags = [spec.prefix_checkpoint for spec in state_specs] if state_specs else [True] * len(tensor_specs)
    if all(checkpoint_flags):
        allocation = _allocate_state_caches(tensor_specs, num_caches, device)
        return allocation, allocation.pools

    # Packed layouts copy owning pools, not named views. Partition before
    # layout construction, then restore the model-facing declaration order.
    views = [None] * len(tensor_specs)
    pools = []
    checkpoint_pools = ()
    for checkpoint in (True, False):
        indices = [i for i, flag in enumerate(checkpoint_flags) if flag == checkpoint]
        if not indices:
            continue
        allocation = _allocate_state_caches([tensor_specs[i] for i in indices], num_caches, device)
        pools.extend(allocation.pools)
        if checkpoint:
            checkpoint_pools = allocation.pools
        for i, view in zip(indices, allocation.tensor_views):
            views[i] = view
    return CacheAllocation(pools=tuple(pools), tensor_views=tuple(views)), checkpoint_pools


class CompactStatePlan:
    """Compile compact-unit and request-private auxiliary layouts once.

    Callers must supply explicit *single-boundary* declarations, never legacy ring shapes. Model/backend capability
    selection remains outside this plan. Both allocations use the existing backend layout and owning-byte accounting;
    auxiliary entries are indexed by request slots, not compact unit IDs. Explicit internal selection wires this plan to
    indexed execution; omitted plans retain the legacy allocation and execution path.
    """

    @classmethod
    def from_model_config(cls, model_config: ModelConfig, capacity: CompactStateCapacity) -> 'CompactStatePlan':
        """Select declared single-boundary shapes, never guess legacy ring
        axes.

        Stateless target/draft models retain zero resources on every backend. Stateful unknown models/backends keep
        their legacy path unless a caller explicitly requests this plan, in which case unsupported input fails.
        """
        specs = model_config.state_cache_specs
        if not specs and model_config.states_shapes:
            raise ValueError('Model does not declare compact state payloads.')
        if specs and get_backend().get_name() != 'cuda':
            raise ValueError('Compact state execution currently requires the CUDA backend.')
        compact_specs = []
        for spec in specs:
            if spec.prefix_checkpoint:
                if spec.compact_shape is None:
                    raise ValueError(f'State {spec.name} does not declare a compact payload.')
                if not spec.compact_shape or any(dim <= 0 for dim in spec.compact_shape):
                    raise ValueError(f'State {spec.name} has invalid compact geometry.')
                compact_specs.append(replace(spec, shape=spec.compact_shape))
            else:
                if spec.compact_shape is not None:
                    raise ValueError(f'Auxiliary state {spec.name} must retain its request-private geometry.')
                compact_specs.append(spec)
        plan = cls(compact_specs, capacity)
        plan.model_declarations = tuple(deepcopy(specs))
        return plan

    def __init__(self, state_specs: Sequence[StateCacheSpec], capacity: CompactStateCapacity):
        self.capacity = capacity
        self.model_declarations = None  # only model-compiled plans may bind StateCacheEngine
        if any(spec.prefix_checkpoint != spec.store_cacheable for spec in state_specs):
            raise ValueError('Compact state requires matching local and Store recovery fields.')
        if len({spec.name for spec in state_specs}) != len(state_specs):
            raise ValueError('Compact state declarations must have unique names.')
        unit_specs = [spec for spec in state_specs if spec.prefix_checkpoint]
        auxiliary_specs = [spec for spec in state_specs if not spec.prefix_checkpoint]
        if bool(unit_specs) != bool(capacity.num_units):
            raise ValueError('Compact capacity and recoverable state declarations must agree.')
        if auxiliary_specs and not capacity.request_slots:
            raise ValueError('Auxiliary state requires request capacity.')
        self.tensor_specs = build_state_cache_tensor_specs((), state_specs=state_specs)
        self.unit_specs = build_state_cache_tensor_specs((), state_specs=unit_specs)
        self.auxiliary_specs = build_state_cache_tensor_specs((), state_specs=auxiliary_specs)
        backend = get_backend().get_cache_backend() if self.tensor_specs else None
        self._unit_layout = backend.build_state_layout(self.unit_specs) if self.unit_specs else None
        self._auxiliary_layout = backend.build_state_layout(self.auxiliary_specs) if self.auxiliary_specs else None

    def allocate(self, device: torch.device | str) -> tuple[CacheAllocation, CacheAllocation]:
        """Return unified-unit storage and independent request auxiliary
        storage."""
        empty = CacheAllocation(pools=(), tensor_views=())
        units = (self._unit_layout.allocate(self.capacity.num_units, device)
                 if self._unit_layout is not None else empty)
        if isinstance(self._unit_layout, PackedStateCacheLayout):
            offset, ranges = 0, []
            for spec in self.unit_specs:
                if ranges and ranges[-1][0] + ranges[-1][1] == offset:
                    start, size = ranges[-1]
                    ranges[-1] = (start, size + spec.desc.size)
                else:
                    ranges.append((offset, spec.desc.size))
                offset += spec.desc.aligned_size
            units = CacheAllocation(pools=(replace(units.pools[0], transfer_ranges=tuple(ranges)),),
                                    tensor_views=units.tensor_views)
        auxiliary = (self._auxiliary_layout.allocate(self.capacity.request_slots, device)
                     if self._auxiliary_layout is not None else empty)
        return units, auxiliary

    def get_nbytes(self) -> tuple[int, int]:
        """Exact owning bytes, excluding future ID metadata and kernel
        scratch."""
        units, auxiliary = self.allocate('meta')
        return units.nbytes, auxiliary.nbytes


class StateCacheEngine:
    """Own state-cache allocation and state-slot transitions."""

    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig,
                 *, compact_plan: CompactStatePlan | None = None):
        self.cache_config = cache_config
        self.compact_plan = compact_plan
        tensor_specs = build_state_cache_tensor_specs(cache_config.states_shapes,
                                                      state_specs=model_config.state_cache_specs)

        # Non-CUDA integrations patch the canonical device in the legacy path.
        # An explicitly selected compact plan replaces that allocation entirely;
        # sizing and real allocation retain the same backend-compiled plan.
        if compact_plan is None:
            self.allocation, checkpoint_pools = _allocate_state_cache_groups(
                tensor_specs, model_config.state_cache_specs, cache_config.num_state_caches, device='cuda')
            self.unit_allocation = self.auxiliary_allocation = None
        else:
            if tuple(model_config.state_cache_specs) != compact_plan.model_declarations:
                raise ValueError('Compact plan does not match model state bindings.')
            tensor_specs = compact_plan.tensor_specs
            self.unit_allocation, self.auxiliary_allocation = compact_plan.allocate('cuda')
            by_name = {
                spec.name: view
                for specs, allocation in ((compact_plan.unit_specs, self.unit_allocation),
                                          (compact_plan.auxiliary_specs, self.auxiliary_allocation))
                for spec, view in zip(specs, allocation.tensor_views)
            }
            checkpoint_pools = self.unit_allocation.pools
            self.allocation = CacheAllocation(
                pools=self.unit_allocation.pools + self.auxiliary_allocation.pools,
                tensor_views=tuple(by_name[spec.name] for spec in tensor_specs))
        self._cache_tensors = list(self.allocation.tensor_views)
        self._slot_tensors = tuple((pool.tensor, pool.entry_axis) for pool in self.allocation.pools)
        self._checkpoint_slot_tensors = tuple((pool.tensor, pool.entry_axis) for pool in checkpoint_pools)
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
        allocation, _ = _allocate_state_cache_groups(tensor_specs, state_specs, num_caches=1, device='meta')
        return allocation.nbytes

    @property
    def state_caches(self) -> Sequence[torch.Tensor]:
        """Return state-cache tensors in model-facing order."""
        return self._cache_tensors

    @property
    def named_state_caches(self) -> Mapping[str, torch.Tensor]:
        """Return model-facing state-cache tensors keyed by semantic name."""
        return self._named_state_caches

    @property
    def connector_state_cache_pools(self) -> tuple[KVCachePool, ...]:
        """Return owning pools and the slot bytes persisted by the Store."""
        pools = self.unit_allocation.pools if self.compact_plan is not None else self.allocation.pools
        return tuple(KVCachePool(pool.tensor, pool.entry_axis, pool.transfer_ranges) for pool in pools)

    def zero_slots(self, slot_ids: torch.Tensor | None, zero_mask: torch.Tensor) -> None:
        """Zero the selected state slots in every physical tensor."""
        if self.compact_plan is not None:
            raise RuntimeError('Compact inputs are read-only; legacy slot zeroing is not allowed.')
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

    def copy_slots(self, src_slots: int | Sequence[int], dst_slots: int | Sequence[int],
                   *, checkpoint_only: bool = False) -> None:
        """Copy non-overlapping slots, optionally excluding runtime-only state.

        Excluded state is left untouched on both save and restore. Its model must not read stale contents when resuming
        at a prefix boundary.
        """
        if self.compact_plan is not None:
            raise RuntimeError('Compact checkpoints share units; legacy state copying is not allowed.')
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
        slot_tensors = self._checkpoint_slot_tensors if checkpoint_only else self._slot_tensors
        for tensor, slot_axis in slot_tensors:
            for src, dst, length in copy_ranges:
                src_tensor = tensor.narrow(slot_axis, src, length)
                dst_tensor = tensor.narrow(slot_axis, dst, length)
                dst_tensor.copy_(src_tensor, non_blocking=True)
