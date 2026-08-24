# Copyright (c) OpenMMLab. All rights reserved.
"""Paging ownership for asynchronous external KV-cache saves."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lmdeploy.pytorch.kv_connector import KVConnectorMetadata, KVOperationId

if TYPE_CHECKING:
    from .scheduler import Scheduler


class KVSaveCoordinator:
    """Keep immutable GPU block snapshots alive until all TP ranks finish.

    Connectors own save planning and remote I/O. This coordinator owns only paging references, indexed by the
    connector's unique save operation ID.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self._leases: dict[KVOperationId, np.ndarray] = {}

    def acquire(self, metadata: KVConnectorMetadata) -> None:
        """Acquire every save lease before metadata reaches model workers."""
        block_manager = self.scheduler.block_manager
        for lease in metadata.get_save_block_leases():
            if lease.operation_id in self._leases:
                raise RuntimeError(f'save operation {lease.operation_id} already owns a block lease')
            logical_block_ids = np.asarray(lease.logical_block_ids, dtype=np.int64)
            block_manager.pin_logical_blocks(logical_block_ids)
            self._leases[lease.operation_id] = logical_block_ids

    def update(self, completed_save_ids: frozenset[KVOperationId]) -> None:
        """Release operations that reached a terminal state on every TP
        rank."""
        block_manager = self.scheduler.block_manager
        for operation_id in completed_save_ids:
            logical_block_ids = self._leases.pop(operation_id, None)
            if logical_block_ids is not None:
                block_manager.release_logical_blocks(logical_block_ids)

    def has_pending(self) -> bool:
        return bool(self._leases)

    def clear(self) -> None:
        """Release all leases after worker-side transfer queues have
        drained."""
        block_manager = self.scheduler.block_manager
        for logical_block_ids in self._leases.values():
            block_manager.release_logical_blocks(logical_block_ids)
        self._leases.clear()
