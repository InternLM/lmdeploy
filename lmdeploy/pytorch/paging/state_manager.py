# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.messages import SchedulerSequence


class StateAllocator:
    """State allocator."""

    def __init__(self, num_states: int, offset: int = 0):
        self.num_states = num_states
        self._free_states = np.arange(offset, offset + num_states, dtype=np.int64)
        self._free_count = num_states

    def allocate(self):
        """allocate."""
        if self.get_num_free() == 0:
            raise RuntimeError('No free states.')
        alloc_id = self._free_states[-self._free_count]
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
                 num_runtime_states: int = None):
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
        self._runtime_states: set[int] = set()
        self._checkpoint_states: set[int] = set()

    def is_allocated(self, seq: SchedulerSequence):
        """Check if a sequence is allocated."""
        return seq.logical_state >= 0

    def allocate_state(self):
        """Allocate one state-cache slot for an active sequence."""
        if self.get_num_free_runtime() <= 0:
            raise RuntimeError('No free states.')
        state_id = int(self.allocator.allocate())
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
        state_id = int(self.allocator.allocate())
        self._checkpoint_states.add(state_id)
        return state_id

    def free_checkpoint_state(self, state_id: int):
        """Free one frozen prefix-cache checkpoint state slot."""
        state_id = int(state_id)
        if state_id not in self._checkpoint_states:
            raise RuntimeError(f'State {state_id} is not a checkpoint state.')
        self._checkpoint_states.remove(state_id)
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
        return max(0, min(free_runtime_capacity, self.allocator.get_num_free()))

    def get_num_free_checkpoint(self):
        """Get raw free slots that checkpoint saves may reserve."""
        return self.allocator.get_num_free()

    def get_num_runtime_states(self):
        """Get num allocated runtime states."""
        return len(self._runtime_states)

    def get_num_allocated_checkpoint_states(self):
        """Get num allocated checkpoint states."""
        return len(self._checkpoint_states)


def build_state_manager(cache_config: CacheConfig) -> StateManager:
    """Build state manager."""
    num_states = cache_config.num_state_caches
    # state is different from block, we always reserve one state for system use
    num_reserved = 1
    return StateManager(num_states,
                        num_reserved,
                        num_runtime_states=cache_config.max_batches)
