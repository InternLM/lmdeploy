# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.messages import SchedulerSequence

from .block_manager.group_allocator import GroupAllocator, GroupHandle


class StateAllocator:
    """State allocator."""

    def __init__(self, num_states: int, offset: int = 0):
        self.num_states = num_states
        self._free_states = np.arange(offset, offset + num_states, dtype=np.int64)
        self._free_count = num_states

    def allocate(self, max_id: int | None = None):
        """Allocate one free state, optionally below ``max_id``."""
        if self.get_num_free() == 0:
            raise RuntimeError('No free states.')
        start = len(self._free_states) - self._free_count
        if max_id is None:
            free_index = start
        else:
            candidates = np.flatnonzero(self._free_states[start:] < max_id)
            if len(candidates) == 0:
                raise RuntimeError('No free states.')
            free_index = start + int(candidates[0])
            self._free_states[start], self._free_states[free_index] = (
                self._free_states[free_index], self._free_states[start])
            # The candidate is now at the head of the free segment.  Reading
            # ``free_index`` here would return the value displaced by the
            # swap when the candidate was not already at ``start``.
            free_index = start
        alloc_id = self._free_states[free_index]
        self._free_count -= 1
        return alloc_id

    def free(self, state_id: int):
        """free."""
        if self._free_count >= self.num_states:
            raise RuntimeError('All states are free.')
        self._free_count += 1
        self._free_states[-self._free_count] = state_id

    def get_num_free(self):
        return self._free_count


class StateManager:
    """Manage runtime and checkpoint ownership over one elastic state pool.

    Runtime sequence states have a configurable capacity cap so a large prefix
    checkpoint budget cannot starve active requests.  Checkpoint states borrow
    from the same allocator and are evicted by ``BlockTrie`` when runtime slots
    need to be recovered.
    """

    def __init__(self,
                 num_states: int,
                 num_reserved: int = 0,
                 num_runtime_states: int = None,
                 group_allocator: GroupAllocator | None = None):
        if num_states is None:
            num_states = 1
        self.num_states = num_states
        self.num_reserved = num_reserved
        num_available = max(0, num_states - num_reserved)

        if num_runtime_states is None:
            num_runtime_states = num_available
        num_runtime_states = max(0, min(num_runtime_states, num_available))

        self.num_runtime_states = num_runtime_states
        self.allocator = StateAllocator(num_available, offset=num_reserved)
        self.group_allocator = group_allocator
        self._state_groups: dict[int, GroupHandle] = {}
        if group_allocator is not None:
            num_protected_state_groups = num_reserved + num_runtime_states
            if group_allocator.num_protected_groups < num_protected_state_groups + int(
                    group_allocator.padding_group_id is not None):
                raise ValueError('Shared allocator does not contain enough protected state groups.')
            state_group_offset = int(group_allocator.padding_group_id is not None)
            for state_id in range(num_reserved + num_runtime_states):
                group_id = state_group_offset + state_id
                self._state_groups[state_id] = group_allocator.protected_group_handle(group_id)
        self._runtime_states: set[int] = set()
        self._checkpoint_states: set[int] = set()

    @property
    def _protected_state_limit(self) -> int:
        """Return the first state id that is not a runtime-protected slot."""
        return self.num_reserved + self.num_runtime_states

    def _num_free_protected_states(self) -> int:
        """Count protected state groups not currently borrowed or running."""
        used = self._runtime_states | {
            state_id for state_id in self._checkpoint_states if state_id < self._protected_state_limit
        }
        return max(0, self.num_runtime_states - len(used))

    def is_allocated(self, seq: SchedulerSequence):
        """Check if a sequence is allocated."""
        return seq.logical_state >= 0

    def allocate_state(self):
        """Allocate one state-cache slot for an active sequence."""
        if self.get_num_free_runtime() <= 0:
            raise RuntimeError('No free states.')
        max_id = self._protected_state_limit if self.group_allocator is not None else None
        state_id = int(self.allocator.allocate(max_id=max_id))
        self._runtime_states.add(state_id)
        return state_id

    def free_state(self, state_id: int):
        """Free one state-cache slot."""
        state_id = int(state_id)
        if state_id not in self._runtime_states:
            raise RuntimeError(f'State {state_id} is not a runtime state.')
        self._runtime_states.remove(state_id)
        self.allocator.free(state_id)

    def allocate_checkpoint_state(self):
        """Allocate one frozen prefix-cache checkpoint state slot."""
        if self.group_allocator is None:
            state_id = int(self.allocator.allocate())
        else:
            # Prefer a free protected runtime slot. It already has a stable
            # protected group mapping; only checkpoint-only slots need a new
            # flexible state group.
            if self._num_free_protected_states() > 0:
                state_id = int(self.allocator.allocate(max_id=self._protected_state_limit))
            else:
                state_id = int(self.allocator.allocate())
            if state_id >= self._protected_state_limit:
                try:
                    self._state_groups[state_id] = self.group_allocator.acquire_group(role='state')
                except Exception:
                    self.allocator.free(state_id)
                    raise
        self._checkpoint_states.add(state_id)
        return state_id

    def free_checkpoint_state(self, state_id: int):
        """Free one frozen prefix-cache checkpoint state slot."""
        state_id = int(state_id)
        if state_id not in self._checkpoint_states:
            raise RuntimeError(f'State {state_id} is not a checkpoint state.')
        self._checkpoint_states.remove(state_id)
        if self.group_allocator is not None and state_id >= self._protected_state_limit:
            handle = self._state_groups.pop(state_id)
            self.group_allocator.release_group(handle)
        self.allocator.free(state_id)

    def allocate(self, seq: SchedulerSequence):
        """Allocate states for a sequence."""
        if self.is_allocated(seq):
            return None
        seq.logical_state = self.allocate_state()

    def free(self, seq: SchedulerSequence):
        """Free states for a sequence."""
        if not self.is_allocated(seq):
            return None
        self.free_state(seq.logical_state)
        seq.logical_state = -1

    def get_num_free(self):
        """Get num free."""
        return self.allocator.get_num_free()

    def get_num_free_runtime(self):
        """Get slots still available under the runtime-state cap."""
        free_runtime_capacity = self.num_runtime_states - len(self._runtime_states)
        if self.group_allocator is not None:
            return max(0, min(free_runtime_capacity, self._num_free_protected_states()))
        return max(0, min(free_runtime_capacity, self.allocator.get_num_free()))

    def get_num_free_checkpoint(self):
        """Get raw free slots that checkpoint saves may reserve."""
        free_slots = self.allocator.get_num_free()
        if self.group_allocator is not None:
            free_protected = self._num_free_protected_states()
            free_flexible_slots = max(0, free_slots - free_protected)
            free_slots = free_protected + min(free_flexible_slots, self.group_allocator.num_empty_groups)
        return free_slots

    def get_num_runtime_states(self):
        """Get num allocated runtime states."""
        return len(self._runtime_states)

    def get_num_allocated_checkpoint_states(self):
        """Get num allocated checkpoint states."""
        return len(self._checkpoint_states)

    def get_physical_state_id(self, state_id: int) -> int:
        """Resolve a logical state ID to its physical shared-group ID."""
        state_id = int(state_id)
        if self.group_allocator is None:
            return state_id
        try:
            return self._state_groups[state_id].group_id
        except KeyError as exc:
            raise ValueError(f'State {state_id} has no physical shared-group mapping.') from exc


def build_state_manager(cache_config: CacheConfig,
                        group_allocator: GroupAllocator | None = None) -> StateManager:
    """Build state manager."""
    # state is different from block, we always reserve one state for system use
    num_reserved = 1
    num_state_caches = cache_config.num_state_caches
    if num_state_caches is None:
        num_state_caches = num_reserved

    # `num_state_caches` is the number of allocated cache rows, including
    # reserved rows. StateManager subtracts reserved rows internally, so pass
    # the total row count to keep allocatable state ids below num_state_caches.
    # Rows left after reserved rows and explicit checkpoint budget are runtime
    # rows. With ExecutorBase's default sizing this gives max_batches plus one
    # spare runtime row.
    num_runtime_states = num_state_caches - num_reserved - cache_config.prefix_cache_state_budget
    return StateManager(num_state_caches,
                        num_reserved,
                        num_runtime_states=num_runtime_states,
                        group_allocator=group_allocator)
