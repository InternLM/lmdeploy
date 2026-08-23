# Copyright (c) OpenMMLab. All rights reserved.
"""Asynchronous Mooncake Store KV-cache load worker."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lmdeploy.pytorch.kv_connector.base import RequestId
from lmdeploy.utils import get_logger

from .data import (
    MooncakeStoreKeyMetadata,
    MooncakeStoreLoadRequest,
    MooncakeStoreRegistration,
    build_store_key,
)

logger = get_logger('lmdeploy')


@dataclass(frozen=True)
class _LoadTask:
    request: MooncakeStoreLoadRequest
    enqueue_time: float


class KVCacheStoreRecvingThread(threading.Thread):
    """Read Mooncake values directly into scheduler-owned GPU blocks."""

    _STOP = object()

    def __init__(
        self,
        *,
        store: Any,
        registrations: tuple[MooncakeStoreRegistration, ...],
        row_block_sizes: tuple[int, ...],
        num_gpu_blocks: int,
        key_metadata: MooncakeStoreKeyMetadata,
        global_rank: int,
        tp_rank: int,
        tp_size: int,
        completion_callback: Callable[[RequestId, set[int]], None],
    ) -> None:
        super().__init__(name='MooncakeKVCacheStoreReceiver', daemon=True)
        if not registrations or len(registrations) != len(row_block_sizes):
            raise ValueError('Mooncake receiver requires one block size per registered region')
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(f'tp_rank must be in [0, {tp_size}), got {tp_rank}')
        if key_metadata.tp_size != tp_size:
            raise ValueError('receiver tp_size must match Mooncake key metadata')

        self.store = store
        self.registrations = registrations
        self.row_block_sizes = row_block_sizes
        self.num_gpu_blocks = num_gpu_blocks
        self.key_metadata = key_metadata
        self.global_rank = global_rank
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.key_rank = tp_rank // key_metadata.kv_head_replica_num
        self.completion_callback = completion_callback
        self.request_queue: queue.Queue[_LoadTask | object] = queue.Queue()
        self._state_lock = threading.Lock()
        self._closed = False

    def add_request(self, request: MooncakeStoreLoadRequest) -> None:
        """Enqueue a load without waiting for Store I/O."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError('Mooncake KV-cache receiver is closed')
            self.request_queue.put(_LoadTask(request, time.perf_counter()))
        logger.info(
            'Mooncake KV load enqueued: global_rank=%d tp_rank=%d tp_size=%d '
            'request_id=%s blocks=%d',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.request_id,
            len(request.block_ids),
        )

    def _validate_block_id(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self.num_gpu_blocks:
            raise ValueError(
                f'physical block ID {block_id} is outside [0, {self.num_gpu_blocks})')

    def _scatter_block(self, block_id: int) -> tuple[list[int], list[int]]:
        self._validate_block_id(block_id)
        addresses = [
            registration.address + block_id * block_size
            for registration, block_size in zip(
                self.registrations,
                self.row_block_sizes,
                strict=True,
            )
        ]
        return addresses, list(self.row_block_sizes)

    def _load(self, request: MooncakeStoreLoadRequest) -> set[int]:
        keys = [
            build_store_key(self.key_metadata, self.key_rank, block_hash)
            for block_hash in request.block_hashes
        ]
        addresses = []
        sizes = []
        for block_id in request.block_ids:
            block_addresses, block_sizes = self._scatter_block(block_id)
            addresses.append(block_addresses)
            sizes.append(block_sizes)

        total_bytes = sum(sum(block_sizes) for block_sizes in sizes)
        logger.info(
            'Mooncake Store interaction before: operation=load_batch_get_into_multi_buffers '
            'global_rank=%d tp_rank=%d tp_size=%d request_id=%s keys=%d '
            'fragments_per_key=%d bytes=%d',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.request_id,
            len(keys),
            len(self.registrations),
            total_bytes,
        )
        start = time.perf_counter()
        try:
            results = self.store.batch_get_into_multi_buffers(keys, addresses, sizes)
            results = list(results)
            if len(results) != len(keys):
                raise ValueError(
                    f'batch_get_into_multi_buffers returned {len(results)} results '
                    f'for {len(keys)} keys')
            if any(isinstance(result, bool) or not isinstance(result, int) for result in results):
                raise TypeError('batch_get_into_multi_buffers returned a non-integer result')
        except Exception as error:
            logger.error(
                'Mooncake Store interaction after: operation=load_batch_get_into_multi_buffers '
                'global_rank=%d tp_rank=%d tp_size=%d request_id=%s status=error '
                'keys=%d elapsed_ms=%.3f error=%s',
                self.global_rank,
                self.tp_rank,
                self.tp_size,
                request.request_id,
                len(keys),
                (time.perf_counter() - start) * 1000,
                error,
                exc_info=True,
            )
            return set(request.block_ids)

        failed_indices = [index for index, result in enumerate(results) if result < 0]
        failed_blocks = {request.block_ids[index] for index in failed_indices}
        log = logger.info if not failed_blocks else logger.error
        log(
            'Mooncake Store interaction after: operation=load_batch_get_into_multi_buffers '
            'global_rank=%d tp_rank=%d tp_size=%d request_id=%s status=%s '
            'keys=%d failed=%d bytes=%d elapsed_ms=%.3f',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.request_id,
            'ok' if not failed_blocks else 'partial_failure',
            len(keys),
            len(failed_blocks),
            total_bytes,
            (time.perf_counter() - start) * 1000,
        )
        return failed_blocks

    def run(self) -> None:
        while True:
            item = self.request_queue.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, _LoadTask)
                request = item.request
                logger.info(
                    'Mooncake KV load dequeued: global_rank=%d tp_rank=%d tp_size=%d '
                    'request_id=%s queue_wait_ms=%.3f',
                    self.global_rank,
                    self.tp_rank,
                    self.tp_size,
                    request.request_id,
                    (time.perf_counter() - item.enqueue_time) * 1000,
                )
                try:
                    failed_blocks = self._load(request)
                except Exception:
                    failed_blocks = set(request.block_ids)
                    logger.exception(
                        'Mooncake KV load failed before Store completion: '
                        'global_rank=%d tp_rank=%d request_id=%s',
                        self.global_rank,
                        self.tp_rank,
                        request.request_id,
                    )
                self.completion_callback(request.request_id, failed_blocks)
                logger.info(
                    'Mooncake KV load completed: global_rank=%d tp_rank=%d tp_size=%d '
                    'request_id=%s status=%s failed_blocks=%d',
                    self.global_rank,
                    self.tp_rank,
                    self.tp_size,
                    request.request_id,
                    'ok' if not failed_blocks else 'error',
                    len(failed_blocks),
                )
            finally:
                self.request_queue.task_done()

    def close(self) -> None:
        """Drain accepted loads and stop the receiver exactly once."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self.request_queue.join()
        self.request_queue.put(self._STOP)
        self.request_queue.join()
        self.join()
