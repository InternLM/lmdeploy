# Copyright (c) OpenMMLab. All rights reserved.
"""Asynchronous Mooncake Store KV-cache transfer workers."""

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
    MooncakeStoreSaveRequest,
    build_store_key,
)

logger = get_logger('lmdeploy')


@dataclass(frozen=True)
class _LoadTask:
    request: MooncakeStoreLoadRequest
    enqueue_time: float


@dataclass(frozen=True)
class _SaveTask:
    request: MooncakeStoreSaveRequest
    ready_event: Any
    enqueue_time: float


def _scatter_block(
    registrations: tuple[MooncakeStoreRegistration, ...],
    row_block_sizes: tuple[int, ...],
    num_gpu_blocks: int,
    block_id: int,
) -> tuple[list[int], list[int]]:
    """Resolve one physical block to its registered row fragments."""
    if block_id < 0 or block_id >= num_gpu_blocks:
        raise ValueError(
            f'physical block ID {block_id} is outside [0, {num_gpu_blocks})')
    addresses = [
        registration.address + block_id * block_size
        for registration, block_size in zip(
            registrations,
            row_block_sizes,
            strict=True,
        )
    ]
    return addresses, list(row_block_sizes)


def _new_replicate_config() -> Any:
    """Create Mooncake's optional put policy only when a save is issued."""
    try:
        from mooncake.store import ReplicateConfig
    except ImportError as e:
        raise ImportError(
            'Mooncake KV-cache save requires ReplicateConfig from the '
            'mooncake-transfer-engine package.') from e
    return ReplicateConfig()


class KVCacheStoreSendingThread(threading.Thread):
    """Store immutable scheduler-pinned GPU blocks in the background."""

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
        completion_callback: Callable[[int], None],
        replicate_config: Any = None,
    ) -> None:
        super().__init__(name='MooncakeKVCacheStoreSender', daemon=True)
        if not registrations or len(registrations) != len(row_block_sizes):
            raise ValueError('Mooncake sender requires one block size per registered region')
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(f'tp_rank must be in [0, {tp_size}), got {tp_rank}')
        if key_metadata.tp_size != tp_size:
            raise ValueError('sender tp_size must match Mooncake key metadata')

        self.store = store
        self.registrations = registrations
        self.row_block_sizes = row_block_sizes
        self.num_gpu_blocks = num_gpu_blocks
        self.key_metadata = key_metadata
        self.global_rank = global_rank
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.key_rank = tp_rank // key_metadata.kv_head_replica_num
        self.replica_rank = tp_rank % key_metadata.kv_head_replica_num
        self.completion_callback = completion_callback
        self.replicate_config = replicate_config
        self.request_queue: queue.Queue[_SaveTask | object] = queue.Queue()
        self._state_lock = threading.Lock()
        self._closed = False

    def add_request(
        self,
        request: MooncakeStoreSaveRequest,
        ready_event: Any,
    ) -> None:
        """Enqueue a save without synchronizing the model's compute stream."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError('Mooncake KV-cache sender is closed')
            self.request_queue.put(
                _SaveTask(
                    request=request,
                    ready_event=ready_event,
                    enqueue_time=time.perf_counter(),
                ))
        logger.debug(
            'Mooncake KV save enqueued: global_rank=%d tp_rank=%d tp_size=%d '
            'save_id=%d request_id=%s blocks=%d',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.save_id,
            request.request_id,
            len(request.block_ids),
        )

    def _owned_entries(
        self,
        request: MooncakeStoreSaveRequest,
    ) -> tuple[list[str], list[int]]:
        if not (len(request.block_ids) == len(request.block_hashes)
                == len(request.logical_block_ids)):
            raise ValueError('Mooncake save request block fields must have equal lengths')

        replica_num = self.key_metadata.kv_head_replica_num
        keys = []
        block_ids = []
        for suffix_index, (block_id, block_hash) in enumerate(
                zip(request.block_ids, request.block_hashes, strict=True)):
            absolute_block = request.start_block + suffix_index
            if absolute_block % replica_num != self.replica_rank:
                continue
            keys.append(build_store_key(self.key_metadata, self.key_rank, block_hash))
            block_ids.append(block_id)
        return keys, block_ids

    def _find_missing(self, request: MooncakeStoreSaveRequest, keys: list[str]) -> list[int]:
        logger.debug(
            'Mooncake Store interaction before: operation=save_batch_is_exist '
            'global_rank=%d tp_rank=%d tp_size=%d save_id=%d request_id=%s keys=%d',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.save_id,
            request.request_id,
            len(keys),
        )
        start = time.perf_counter()
        try:
            states = list(self.store.batch_is_exist(keys))
            if len(states) != len(keys):
                raise ValueError(
                    f'batch_is_exist returned {len(states)} states for {len(keys)} keys')
            if any(isinstance(state, bool) or not isinstance(state, int)
                   or state not in (0, 1) for state in states):
                raise TypeError('batch_is_exist returned a state other than integer 0 or 1')
        except Exception as error:
            logger.error(
                'Mooncake Store interaction after: operation=save_batch_is_exist '
                'global_rank=%d tp_rank=%d tp_size=%d save_id=%d request_id=%s '
                'status=error keys=%d elapsed_ms=%.3f error=%s',
                self.global_rank,
                self.tp_rank,
                self.tp_size,
                request.save_id,
                request.request_id,
                len(keys),
                (time.perf_counter() - start) * 1000,
                error,
                exc_info=True,
            )
            raise
        missing = [index for index, state in enumerate(states) if state == 0]
        logger.debug(
            'Mooncake Store interaction after: operation=save_batch_is_exist '
            'global_rank=%d tp_rank=%d tp_size=%d save_id=%d request_id=%s '
            'status=ok keys=%d missing=%d elapsed_ms=%.3f',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.save_id,
            request.request_id,
            len(keys),
            len(missing),
            (time.perf_counter() - start) * 1000,
        )
        return missing

    def _put_missing(
        self,
        request: MooncakeStoreSaveRequest,
        keys: list[str],
        block_ids: list[int],
        missing: list[int],
    ) -> bool:
        missing_keys = [keys[index] for index in missing]
        addresses = []
        sizes = []
        for index in missing:
            block_addresses, block_sizes = _scatter_block(
                self.registrations,
                self.row_block_sizes,
                self.num_gpu_blocks,
                block_ids[index],
            )
            addresses.append(block_addresses)
            sizes.append(block_sizes)

        total_bytes = sum(sum(block_sizes) for block_sizes in sizes)
        logger.debug(
            'Mooncake Store interaction before: operation=save_batch_put_from_multi_buffers '
            'global_rank=%d tp_rank=%d tp_size=%d save_id=%d request_id=%s '
            'keys=%d fragments_per_key=%d bytes=%d',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.save_id,
            request.request_id,
            len(missing_keys),
            len(self.registrations),
            total_bytes,
        )
        start = time.perf_counter()
        try:
            if self.replicate_config is None:
                self.replicate_config = _new_replicate_config()
            results = list(
                self.store.batch_put_from_multi_buffers(
                    missing_keys,
                    addresses,
                    sizes,
                    self.replicate_config,
                ))
            if len(results) != len(missing_keys):
                raise ValueError(
                    f'batch_put_from_multi_buffers returned {len(results)} results '
                    f'for {len(missing_keys)} keys')
            if any(isinstance(result, bool) or not isinstance(result, int)
                   for result in results):
                raise TypeError('batch_put_from_multi_buffers returned a non-integer result')
        except Exception as error:
            logger.error(
                'Mooncake Store interaction after: '
                'operation=save_batch_put_from_multi_buffers '
                'global_rank=%d tp_rank=%d tp_size=%d save_id=%d request_id=%s '
                'status=error keys=%d bytes=%d elapsed_ms=%.3f error=%s',
                self.global_rank,
                self.tp_rank,
                self.tp_size,
                request.save_id,
                request.request_id,
                len(missing_keys),
                total_bytes,
                (time.perf_counter() - start) * 1000,
                error,
                exc_info=True,
            )
            raise
        failed = [result for result in results if result < 0]
        log = logger.debug if not failed else logger.error
        log(
            'Mooncake Store interaction after: operation=save_batch_put_from_multi_buffers '
            'global_rank=%d tp_rank=%d tp_size=%d save_id=%d request_id=%s '
            'status=%s keys=%d failed=%d bytes=%d elapsed_ms=%.3f',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.save_id,
            request.request_id,
            'ok' if not failed else 'partial_failure',
            len(missing_keys),
            len(failed),
            total_bytes,
            (time.perf_counter() - start) * 1000,
        )
        return not failed

    def _save(self, task: _SaveTask) -> bool:
        request = task.request
        keys, block_ids = self._owned_entries(request)
        missing = None
        try:
            missing = self._find_missing(request, keys) if keys else []
        finally:
            # Waiting also on lookup failure keeps completion ordering tied to
            # this model step, which makes lease release deterministic.
            task.ready_event.synchronize()

        # The query does not touch KV memory and can overlap the forward. The
        # direct GPU read must wait until all preceding compute-stream writes
        # are visible.
        if missing:
            return self._put_missing(request, keys, block_ids, missing)
        return True

    def run(self) -> None:
        while True:
            item = self.request_queue.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, _SaveTask)
                request = item.request
                logger.debug(
                    'Mooncake KV save dequeued: global_rank=%d tp_rank=%d tp_size=%d '
                    'save_id=%d request_id=%s queue_wait_ms=%.3f',
                    self.global_rank,
                    self.tp_rank,
                    self.tp_size,
                    request.save_id,
                    request.request_id,
                    (time.perf_counter() - item.enqueue_time) * 1000,
                )
                success = False
                try:
                    success = self._save(item)
                except Exception:
                    logger.exception(
                        'Mooncake KV save reached terminal failure: '
                        'global_rank=%d tp_rank=%d save_id=%d request_id=%s',
                        self.global_rank,
                        self.tp_rank,
                        request.save_id,
                        request.request_id,
                    )
                self.completion_callback(request.save_id)
                logger.debug(
                    'Mooncake KV save completed: global_rank=%d tp_rank=%d tp_size=%d '
                    'save_id=%d request_id=%s status=%s',
                    self.global_rank,
                    self.tp_rank,
                    self.tp_size,
                    request.save_id,
                    request.request_id,
                    'ok' if success else 'error',
                )
            finally:
                self.request_queue.task_done()

    def close(self) -> None:
        """Drain accepted saves and stop the sender exactly once."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self.request_queue.join()
        self.request_queue.put(self._STOP)
        self.request_queue.join()
        self.join()


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
        logger.debug(
            'Mooncake KV load enqueued: global_rank=%d tp_rank=%d tp_size=%d '
            'request_id=%s blocks=%d',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.request_id,
            len(request.block_ids),
        )

    def _scatter_block(self, block_id: int) -> tuple[list[int], list[int]]:
        return _scatter_block(
            self.registrations,
            self.row_block_sizes,
            self.num_gpu_blocks,
            block_id,
        )

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
        logger.debug(
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
        log = logger.debug if not failed_blocks else logger.error
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
                logger.debug(
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
                logger.debug(
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
