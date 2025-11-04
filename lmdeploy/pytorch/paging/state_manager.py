# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from lmdeploy.pytorch.messages import SchedulerSequence


class StateAllocator:
    """State allocator."""

    def __init__(self, num_states: int):
        self.num_states = num_states
        self._free_states = np.arange(num_states, dtype=np.int64)
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

    def __init__(self, num_states: int):
        if num_states is None:
            num_states = 1
        self.allocator = StateAllocator(num_states)

    def is_allocated(self, seq: SchedulerSequence):
        """Check if a sequence is allocated."""
        return seq.logical_state >= 0

    def allocate(self, seq: SchedulerSequence):
        """Allocate states for a sequence."""
        if self.is_allocated(seq):
            return None
        seq.logical_state = self.allocator.allocate()

    def free(self, seq: SchedulerSequence):
        """Free states for a sequence."""
        if not self.is_allocated(seq):
            return None
        self.allocator.free(seq.logical_state)
        seq.logical_state = -1

    def get_num_free(self):
        """Get num free."""
        return self.allocator.get_num_free()
