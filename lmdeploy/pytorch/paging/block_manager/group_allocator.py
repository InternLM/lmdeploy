# Copyright (c) OpenMMLab. All rights reserved.
"""Ownership of physical groups used by shared KV/state paging."""

from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class GroupRole(IntEnum):
    """Role of one physical shared-cache group."""

    PROTECTED = 0
    EMPTY = 1
    KV = 2
    STATE = 3


@dataclass(frozen=True)
class GroupHandle:
    """Generation-safe handle for one physical group."""

    group_id: int
    generation: int


class GroupAllocator:
    """Reserve complete physical groups for KV or state owners.

    This class does not know about logical block ids or ref counts.  It only decides which flexible groups are available
    and keeps protected groups stable for the state manager.
    """

    def __init__(self,
                 num_gpu_blocks: int,
                 group_size: int,
                 num_protected_groups: int = 0,
                 reserve_padding_group: bool = False) -> None:
        if group_size <= 0:
            raise ValueError('group_size must be positive.')
        if num_gpu_blocks < 0 or num_gpu_blocks % group_size != 0:
            raise ValueError('num_gpu_blocks must be a non-negative multiple of group_size.')
        if num_protected_groups < 0:
            raise ValueError('num_protected_groups must be non-negative.')

        self.group_size = group_size
        self.num_groups = num_gpu_blocks // group_size + num_protected_groups + int(reserve_padding_group)
        self.num_protected_groups = num_protected_groups + int(reserve_padding_group)
        self._padding_group_id = 0 if reserve_padding_group else None
        self._group_roles = np.full((self.num_groups, ), GroupRole.EMPTY, dtype=np.int8)
        self._group_roles[:self.num_protected_groups] = GroupRole.PROTECTED
        # Fresh groups are consumed in order. Recycled groups stay in a FIFO
        # queue, preserving the old deque allocation order without making the
        # common fresh-allocation path pop one item at a time.
        self._next_flexible_group = self.num_protected_groups
        self._recycled_flexible_groups = deque()
        self._group_generation = np.zeros((self.num_groups, ), dtype=np.int64)

    @property
    def address_span(self) -> int:
        """Return the first physical offset after all groups."""
        return self.num_groups * self.group_size

    @property
    def num_empty_groups(self) -> int:
        """Return flexible groups available for a new owner."""
        return (self.num_groups - self._next_flexible_group + len(self._recycled_flexible_groups))

    @staticmethod
    def _validate_role(role: GroupRole) -> GroupRole:
        try:
            role = GroupRole(role)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Unsupported shared group role: {role}.') from exc
        if role not in (GroupRole.KV, GroupRole.STATE):
            raise ValueError(f'Unsupported shared group role: {role}.')
        return role

    def acquire_group(self, role: GroupRole = GroupRole.KV) -> GroupHandle:
        """Reserve one empty flexible group for ``role``."""
        role = self._validate_role(role)
        if self.num_empty_groups == 0:
            raise MemoryError('No empty shared cache group is available.')
        if self._next_flexible_group < self.num_groups:
            group_id = self._next_flexible_group
            self._next_flexible_group += 1
        else:
            group_id = self._recycled_flexible_groups.popleft()
        self._group_roles[group_id] = role
        self._group_generation[group_id] += 1
        return GroupHandle(group_id, int(self._group_generation[group_id]))

    def acquire_groups(self, num_groups: int, role: GroupRole = GroupRole.KV) -> tuple[GroupHandle, ...]:
        """Atomically reserve complete empty groups."""
        if num_groups < 0:
            raise ValueError('num_groups must be non-negative.')
        role = self._validate_role(role)
        if num_groups > self.num_empty_groups:
            raise MemoryError('No enough empty shared cache groups.')
        fresh_groups = min(num_groups, self.num_groups - self._next_flexible_group)
        group_ids = list(range(self._next_flexible_group, self._next_flexible_group + fresh_groups))
        self._next_flexible_group += fresh_groups
        for _ in range(num_groups - fresh_groups):
            group_ids.append(self._recycled_flexible_groups.popleft())

        group_ids = np.asarray(group_ids, dtype=np.int64)
        self._group_roles[group_ids] = role
        self._group_generation[group_ids] += 1
        return tuple(
            GroupHandle(int(group_id), int(generation))
            for group_id, generation in zip(group_ids, self._group_generation[group_ids]))

    def release_group(self, group_id: int | GroupHandle, generation: int | None = None) -> None:
        """Release an owned group after validating its generation."""
        if isinstance(group_id, GroupHandle):
            group_id, generation = group_id.group_id, group_id.generation
        if generation is None:
            raise TypeError('generation is required when releasing a shared group.')
        if not 0 <= group_id < self.num_groups:
            raise RuntimeError('Cannot release an out-of-range shared group.')
        role = GroupRole(self._group_roles[group_id])
        if role not in (GroupRole.KV, GroupRole.STATE):
            raise RuntimeError('Cannot release a non-owned shared group.')
        if int(self._group_generation[group_id]) != int(generation):
            raise RuntimeError('Cannot release a stale shared group handle.')
        self._group_roles[group_id] = GroupRole.EMPTY
        self._recycled_flexible_groups.append(group_id)

    def group_handle(self, group_id: int) -> GroupHandle:
        """Return the current handle for one owned group."""
        if not 0 <= group_id < self.num_groups:
            raise ValueError('group_id is out of range.')
        role = GroupRole(self._group_roles[group_id])
        if role not in (GroupRole.KV, GroupRole.STATE):
            raise ValueError('group_id is not currently owned.')
        return GroupHandle(group_id, int(self._group_generation[group_id]))

    def protected_group_handle(self, group_id: int) -> GroupHandle:
        """Return the stable handle for one protected group."""
        if not 0 <= group_id < self.num_groups:
            raise ValueError('group_id is out of range.')
        if GroupRole(self._group_roles[group_id]) != GroupRole.PROTECTED:
            raise ValueError('group_id is not protected.')
        return GroupHandle(group_id, int(self._group_generation[group_id]))

    def protected_group_handles(self, num_groups: int) -> tuple[GroupHandle, ...]:
        """Return stable handles for protected data groups in allocator
        order."""
        if num_groups < 0:
            raise ValueError('num_groups must be non-negative.')
        protected_ids = np.flatnonzero(self._group_roles == GroupRole.PROTECTED)
        if self._padding_group_id is not None:
            protected_ids = protected_ids[protected_ids != self._padding_group_id]
        if num_groups > len(protected_ids):
            raise ValueError('Shared allocator does not contain enough protected data groups.')
        return tuple(self.protected_group_handle(int(group_id)) for group_id in protected_ids[:num_groups])

    def group_role(self, group_id: int) -> GroupRole:
        """Return the current role of one physical group."""
        if not 0 <= group_id < self.num_groups:
            raise ValueError('group_id is out of range.')
        return GroupRole(self._group_roles[group_id])

    def group_roles(self, group_ids: np.ndarray) -> np.ndarray:
        """Return integer role tags for several physical groups."""
        group_ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if np.any(group_ids < 0) or np.any(group_ids >= self.num_groups):
            raise ValueError('group_id is out of range.')
        return self._group_roles[group_ids].copy()

    def group_offsets(self, group_id: int) -> np.ndarray:
        """Return all physical offsets belonging to one group."""
        if not 0 <= group_id < self.num_groups:
            raise ValueError('group_id is out of range.')
        start = group_id * self.group_size
        return np.arange(start, start + self.group_size, dtype=np.int64)
