# Copyright (c) OpenMMLab. All rights reserved.
"""GPU-only block manager for the shared KV/state arena."""

from dataclasses import dataclass

import numpy as np

from ...messages import SchedulerSequence
from .base_block_manager import BaseBlockManager, _div_up, _num_required_blocks
from .group_allocator import GroupAllocator, GroupHandle, GroupRole


@dataclass
class _AppendGroup:
    """Sequence-owned append cursor for one shared KV group."""

    handle: GroupHandle
    next_slot: int = 0


class SharedBlockManager(BaseBlockManager):
    """Manage sequence-private KV groups in the shared GPU arena.

    Shared allocation reserves complete groups for each request. Protected state and padding groups are owned by their
    respective lifecycles and are never exposed through this manager's append cursors.
    """

    def __init__(self,
                 num_gpu_blocks: int,
                 num_cpu_blocks: int,
                 *,
                 group_size: int = 1,
                 num_protected_groups: int = 0,
                 reserve_padding_group: bool = True) -> None:
        if num_cpu_blocks != 0:
            raise ValueError('Shared KV/state allocation requires num_cpu_blocks=0.')
        group_allocator = GroupAllocator(
            num_gpu_blocks,
            group_size,
            num_protected_groups=num_protected_groups,
            reserve_padding_group=reserve_padding_group,
        )
        super().__init__(num_gpu_blocks,
                         num_cpu_blocks,
                         group_allocator=group_allocator)
        self._append_groups: dict[int, list[_AppendGroup]] = {}

    @classmethod
    def num_required_blocks(cls, obj: SchedulerSequence, prealloc_size: int = 0):
        """Get num required blocks."""
        return _num_required_blocks(obj, prealloc_size)

    def can_allocate(self, msg: SchedulerSequence, prealloc_size: int = 0):
        """Return whether enough complete groups remain for this request."""
        return self.num_required_groups(msg, prealloc_size) <= self.group_allocator.num_empty_groups

    def num_required_groups(self, msg: SchedulerSequence, prealloc_size: int = 0) -> int:
        """Return new groups needed after consuming the open append group."""
        num_required_blocks = self.num_required_blocks(msg, prealloc_size)
        if num_required_blocks == 0:
            return 0
        available_slots = self._available_append_slots(msg)
        remaining = max(0, num_required_blocks - available_slots)
        return _div_up(remaining, self.group_allocator.group_size)

    def num_required_capacity(self, msg: SchedulerSequence, prealloc_size: int = 0) -> int:
        """Return admission capacity in complete shared groups."""
        return self.num_required_groups(msg, prealloc_size)

    def num_free_capacity(self) -> int:
        """Return the number of empty groups available for admission."""
        return self.group_allocator.num_empty_groups

    def allocate_msg(self, msg: SchedulerSequence, prealloc_size: int = 0):
        """Allocate a fresh suffix from sequence-private KV groups."""
        num_required_blocks = self.num_required_blocks(msg, prealloc_size)
        if num_required_blocks > 0:
            self._allocate_shared(msg, num_required_blocks)

    def _available_append_slots(self, msg: SchedulerSequence) -> int:
        """Return slots left in the sequence's open append group."""
        groups = self._prune_append_groups(msg)
        if not groups:
            return 0
        return max(0, self.group_allocator.group_size - groups[-1].next_slot)

    def _prune_append_groups(self, msg: SchedulerSequence) -> list[_AppendGroup]:
        """Drop cursors whose groups were released or recycled."""
        groups = self._append_groups.get(msg.seq_id)
        if not groups:
            return []
        while groups:
            group = groups[-1]
            try:
                is_current = (self.group_allocator.group_role(group.handle.group_id) == GroupRole.KV
                               and self.group_allocator.group_handle(group.handle.group_id) == group.handle)
            except ValueError:
                is_current = False
            if is_current:
                break
            groups.pop()
        if not groups:
            self._append_groups.pop(msg.seq_id, None)
        return groups

    def _allocate_shared(self, msg: SchedulerSequence, num_required_blocks: int) -> None:
        """Allocate a fresh suffix using sequence-owned complete groups."""
        groups = self._prune_append_groups(msg)
        if not groups:
            groups = self._append_groups.setdefault(msg.seq_id, [])
        physical_blocks = []
        previous_next_slot = groups[-1].next_slot if groups else None
        if groups and groups[-1].next_slot < self.group_allocator.group_size:
            group = groups[-1]
            take = min(num_required_blocks, self.group_allocator.group_size - group.next_slot)
            start = group.handle.group_id * self.group_allocator.group_size + group.next_slot
            physical_blocks.extend(range(start, start + take))
            group.next_slot += take
            num_required_blocks -= take

        groups_needed = _div_up(num_required_blocks, self.group_allocator.group_size)
        new_groups = ()
        try:
            new_groups = self.group_allocator.acquire_groups(groups_needed, role=GroupRole.KV)
            groups.extend(_AppendGroup(handle) for handle in new_groups)
            if new_groups:
                group_ids = np.fromiter((handle.group_id for handle in new_groups), dtype=np.int64)
                slot_offsets = np.arange(self.group_allocator.group_size, dtype=np.int64)
                new_physical_blocks = (group_ids[:, None] * self.group_allocator.group_size + slot_offsets).reshape(-1)
                new_physical_blocks = new_physical_blocks[:num_required_blocks]
                physical_blocks = np.concatenate((np.asarray(physical_blocks, dtype=np.int64), new_physical_blocks))
                for group_index, group in enumerate(groups[-len(new_groups):]):
                    group.next_slot = min(self.group_allocator.group_size,
                                          num_required_blocks - group_index * self.group_allocator.group_size)
            blocks = self.allocator.allocate_at(np.asarray(physical_blocks, dtype=np.int64))
        except Exception:
            for handle in reversed(new_groups):
                self.group_allocator.release_group(handle)
            if new_groups:
                del groups[-len(new_groups):]
            if previous_next_slot is not None and groups:
                groups[-1].next_slot = previous_next_slot
            if not groups:
                self._append_groups.pop(msg.seq_id, None)
            raise
        msg.logical_blocks.append(blocks)

    def free(self, msg: SchedulerSequence):
        """Free all physical blocks allocated for the sequence."""
        self.allocator.free(msg.logical_blocks.get_real_blocks())
        self._append_groups.pop(msg.seq_id, None)
        msg.logical_blocks.reset()

    def truncate(self, msg: SchedulerSequence, target_num_blocks: int) -> np.ndarray:
        """Release a logical suffix and rewind the sequence append cursor."""
        released = super().truncate(msg, target_num_blocks)
        groups = self._append_groups.get(msg.seq_id)
        if not groups:
            return released

        remaining_physical = self.allocator.get_physical_blocks(msg.logical_blocks.get_real_blocks())
        remaining_groups = set(int(offset // self.group_allocator.group_size) for offset in remaining_physical)
        while groups:
            group = groups[-1]
            group_id = group.handle.group_id
            if group_id in remaining_groups:
                slots = [int(offset % self.group_allocator.group_size) for offset in remaining_physical
                         if int(offset // self.group_allocator.group_size) == group_id]
                group.next_slot = max(slots, default=-1) + 1
                break
            groups.pop()
        if not groups:
            self._append_groups.pop(msg.seq_id, None)
        return released

    def try_swap_out(self, msg: SchedulerSequence):
        """Shared allocation has no CPU swap tier."""
        raise RuntimeError('CPU swap is not supported in shared allocation mode.')

    def try_swap_in(self, msg: SchedulerSequence):
        """Shared allocation has no CPU swap tier."""
        raise RuntimeError('CPU swap is not supported in shared allocation mode.')
