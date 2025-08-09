# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
from typing import Dict, Sequence, Union

from triton.runtime.cache import FileCacheManager

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

TypeHintType = Union[Dict[str, type], Sequence[type], None]


class MPLockCacheManager(FileCacheManager):
    """A cache manager that uses a lock to ensure thread safety."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(f'Create MPLockCacheManager with key={self.key}')
        self._lock_map = dict()

    def _acquire_lock(self, lock_path, timeout=5):
        """Acquire an exclusive lock on the file."""
        import filelock
        logger.debug(f'Acquiring lock for {lock_path}')
        full_lock_path = osp.join(self.cache_dir, f'{lock_path}.lock')
        lock = filelock.FileLock(full_lock_path)

        lock.acquire(timeout=timeout)
        self._lock_map[lock_path] = lock

    def _release_lock(self, lock_path):
        """Release the lock."""
        if lock_path not in self._lock_map:
            return
        logger.debug(f'Release lock for {lock_path}')
        lock_file = self._lock_map.pop(lock_path)
        lock_file.release()

    def _group_is_ready(self, filename: str, group: dict) -> bool:
        """Check if the group is ready."""
        if not isinstance(group, dict):
            return False
        return filename in group

    def get_group(self, filename: str) -> Dict[str, str]:
        out = super().get_group(filename)
        if self._group_is_ready(filename, out):
            return out

        # lock if group is not ready
        self._acquire_lock(filename)
        out = super().get_group(filename)

        if self._group_is_ready(filename, out):
            self._release_lock(filename)
        return out

    def get_file(self, filename) -> str:
        out = super().get_file(filename)
        if out is not None:
            return out

        # lock if file is not ready
        self._acquire_lock(filename)
        # try get file again if other process has put the file
        out = super().get_file(filename)

        # release lock if file exists
        if out is not None:
            self._release_lock(filename)
        return out

    def put(self, data, filename, binary=True) -> str:
        out = super().put(data, filename, binary)
        logger.debug(f'Put file {filename}.')
        if filename.startswith('__grp__'):
            # release group
            self._release_lock(filename[7:])
        else:
            self._release_lock(filename)
        return out
