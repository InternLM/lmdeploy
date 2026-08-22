# Copyright (c) OpenMMLab. All rights reserved.
"""Asynchronous Mooncake Store KV-cache transfer threads."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from lmdeploy.utils import get_logger

from .data import (
    MooncakeStoreKeyMetadata,
    MooncakeStoreLoadRequest,
    MooncakeStoreRegistration,
    MooncakeStoreSaveRequest,
    build_store_key,
)
from .utils import _load_mooncake_replicate_config, _result_histogram

logger = get_logger('lmdeploy')


@dataclass
class _StoreTask:
    request: MooncakeStoreSaveRequest
    ready_event: Any | None
    enqueue_time: float
    ready_waited: bool = False


@dataclass
class _LoadTask:
    request: MooncakeStoreLoadRequest
    enqueue_time: float


class KVCacheStoreSendingThread(threading.Thread):
    """Single-queue background writer for one Mooncake worker rank."""

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
        replicate_config: Any | None = None,
    ) -> None:
        super().__init__(name='MooncakeKVCacheStoreSender', daemon=True)
        if len(registrations) != len(row_block_sizes) or not registrations:
            raise ValueError('Mooncake sender requires one block size per registered region')
        if key_metadata.tp_size != tp_size:
            raise ValueError(
                'Mooncake sender tp_size must match key metadata: '
                f'{tp_size} != {key_metadata.tp_size}')
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(f'tp_rank must be in [0, {tp_size}), got {tp_rank}')
        self.store = store
        self.registrations = registrations
        self.row_block_sizes = row_block_sizes
        self.num_gpu_blocks = num_gpu_blocks
        self.key_metadata = key_metadata
        self.global_rank = global_rank
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.kv_head_replica_num = key_metadata.kv_head_replica_num
        self.key_rank = tp_rank // self.kv_head_replica_num
        self.replica_rank = tp_rank % self.kv_head_replica_num
        self.completion_callback = completion_callback
        self.replicate_config = replicate_config
        self.request_queue: queue.Queue[_StoreTask | object] = queue.Queue()
        self._state_lock = threading.Lock()
        self._closed = False

    def add_request(self, request: MooncakeStoreSaveRequest, ready_event: Any | None) -> None:
        """Enqueue without waiting for GPU readiness or Store I/O."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError('Mooncake KV-cache sender is closed')
            self.request_queue.put(_StoreTask(request, ready_event, time.perf_counter()))
        logger.info(
            'Mooncake KV save enqueued: global_rank=%d tp_rank=%d tp_size=%d '
            'req_id=%s save_id=%d generation=%d token_len=%d blocks=%d',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.req_id,
            request.save_id,
            request.generation,
            request.token_len,
            len(request.block_ids),
        )

    def _log_query_before(self, request: MooncakeStoreSaveRequest, keys: list[str]) -> float:
        logger.info(
            'Mooncake Store interaction before: operation=save_batch_is_exist '
            'global_rank=%d tp_rank=%d tp_size=%d req_id=%s save_id=%d generation=%d '
            'candidate_keys=%d first_key=%s last_key=%s',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.req_id,
            request.save_id,
            request.generation,
            len(keys),
            keys[0],
            keys[-1],
        )
        return time.perf_counter()

    def _query_missing(self, request: MooncakeStoreSaveRequest, keys: list[str]) -> list[int] | None:
        start = self._log_query_before(request, keys)
        try:
            exists_states = self.store.batch_is_exist(keys)
            if not isinstance(exists_states, Sequence) or isinstance(exists_states, (str, bytes)):
                raise TypeError('batch_is_exist must return a sequence')
            exists_states = list(exists_states)
            if len(exists_states) != len(keys):
                raise ValueError(
                    f'batch_is_exist returned {len(exists_states)} states for {len(keys)} keys')
            if any(isinstance(state, bool) or not isinstance(state, int) or state not in (0, 1)
                   for state in exists_states):
                raise ValueError(f'batch_is_exist returned invalid states: {_result_histogram(exists_states)}')
        except Exception as e:
            logger.error(
                'Mooncake Store interaction after: operation=save_batch_is_exist '
                'global_rank=%d tp_rank=%d tp_size=%d req_id=%s save_id=%d generation=%d '
                'status=error candidate_keys=%d elapsed_ms=%.3f error=%s',
                self.global_rank,
                self.tp_rank,
                self.tp_size,
                request.req_id,
                request.save_id,
                request.generation,
                len(keys),
                (time.perf_counter() - start) * 1000,
                e,
                exc_info=True,
            )
            return None

        missing = [index for index, state in enumerate(exists_states) if state == 0]
        logger.info(
            'Mooncake Store interaction after: operation=save_batch_is_exist '
            'global_rank=%d tp_rank=%d tp_size=%d req_id=%s save_id=%d generation=%d '
            'status=ok candidate_keys=%d existing=%d missing=%d elapsed_ms=%.3f result_codes=%s',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.req_id,
            request.save_id,
            request.generation,
            len(keys),
            len(keys) - len(missing),
            len(missing),
            (time.perf_counter() - start) * 1000,
            _result_histogram(exists_states),
        )
        return missing

    def _validate_block_id(self, block_id: int) -> None:
        if isinstance(block_id, bool) or not isinstance(block_id, int):
            raise TypeError('physical block IDs must be integers')
        if block_id < 0 or block_id >= self.num_gpu_blocks:
            raise ValueError(
                f'physical block ID {block_id} is outside [0, {self.num_gpu_blocks})')

    def _scatter_block(self, block_id: int) -> tuple[list[int], list[int]]:
        self._validate_block_id(block_id)
        addresses = [
            registration.address + block_id * block_bytes
            for registration, block_bytes in zip(
                self.registrations,
                self.row_block_sizes,
                strict=True,
            )
        ]
        return addresses, list(self.row_block_sizes)

    def _wait_ready(self, task: _StoreTask) -> None:
        if task.ready_waited:
            return
        task.ready_waited = True
        if task.ready_event is None:
            return
        request = task.request
        logger.info(
            'Mooncake KV save wait before: global_rank=%d tp_rank=%d tp_size=%d '
            'req_id=%s save_id=%d',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.req_id,
            request.save_id,
        )
        wait_start = time.perf_counter()
        try:
            task.ready_event.synchronize()
        except Exception as e:
            logger.error(
                'Mooncake KV save wait after: global_rank=%d tp_rank=%d tp_size=%d '
                'req_id=%s save_id=%d status=error elapsed_ms=%.3f error=%s',
                self.global_rank,
                self.tp_rank,
                self.tp_size,
                request.req_id,
                request.save_id,
                (time.perf_counter() - wait_start) * 1000,
                e,
                exc_info=True,
            )
            raise
        logger.info(
            'Mooncake KV save wait after: global_rank=%d tp_rank=%d tp_size=%d '
            'req_id=%s save_id=%d status=ready elapsed_ms=%.3f',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.req_id,
            request.save_id,
            (time.perf_counter() - wait_start) * 1000,
        )

    def _put_missing(
        self,
        task: _StoreTask,
        keys: list[str],
        missing_indices: list[int],
        physical_block_ids: Sequence[int],
    ) -> bool:
        request = task.request
        missing_keys = [keys[index] for index in missing_indices]
        addresses = []
        sizes = []
        for index in missing_indices:
            block_addresses, block_sizes = self._scatter_block(physical_block_ids[index])
            addresses.append(block_addresses)
            sizes.append(block_sizes)

        total_bytes = sum(sum(block_sizes) for block_sizes in sizes)
        logger.info(
            'Mooncake Store interaction before: operation=save_batch_put_from_multi_buffers '
            'global_rank=%d tp_rank=%d tp_size=%d req_id=%s save_id=%d generation=%d '
            'keys=%d fragments_per_key=%d bytes=%d first_key=%s last_key=%s',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.req_id,
            request.save_id,
            request.generation,
            len(missing_keys),
            len(self.registrations),
            total_bytes,
            missing_keys[0],
            missing_keys[-1],
        )
        start = time.perf_counter()
        try:
            if self.replicate_config is None:
                self.replicate_config = _load_mooncake_replicate_config()
            results = self.store.batch_put_from_multi_buffers(
                missing_keys,
                addresses,
                sizes,
                self.replicate_config,
            )
            if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
                raise TypeError('batch_put_from_multi_buffers must return a sequence')
            results = list(results)
            if len(results) != len(missing_keys):
                raise ValueError(
                    f'batch_put_from_multi_buffers returned {len(results)} results '
                    f'for {len(missing_keys)} keys')
            if any(isinstance(result, bool) or not isinstance(result, int) for result in results):
                raise TypeError('batch_put_from_multi_buffers returned a non-integer result')
        except Exception as e:
            logger.error(
                'Mooncake Store interaction after: operation=save_batch_put_from_multi_buffers '
                'global_rank=%d tp_rank=%d tp_size=%d req_id=%s save_id=%d generation=%d '
                'status=error keys=%d fragments_per_key=%d bytes=%d elapsed_ms=%.3f error=%s',
                self.global_rank,
                self.tp_rank,
                self.tp_size,
                request.req_id,
                request.save_id,
                request.generation,
                len(missing_keys),
                len(self.registrations),
                total_bytes,
                (time.perf_counter() - start) * 1000,
                e,
                exc_info=True,
            )
            return False

        failed = sum(result < 0 for result in results)
        status = 'ok' if failed == 0 else 'partial_failure'
        log = logger.info if failed == 0 else logger.error
        log(
            'Mooncake Store interaction after: operation=save_batch_put_from_multi_buffers '
            'global_rank=%d tp_rank=%d tp_size=%d req_id=%s save_id=%d generation=%d '
            'status=%s keys=%d failed=%d fragments_per_key=%d bytes=%d elapsed_ms=%.3f '
            'result_codes=%s',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.req_id,
            request.save_id,
            request.generation,
            status,
            len(missing_keys),
            failed,
            len(self.registrations),
            total_bytes,
            (time.perf_counter() - start) * 1000,
            _result_histogram(results),
        )
        return failed == 0

    def _handle_request(self, task: _StoreTask) -> bool:
        request = task.request
        full_blocks = min(
            request.token_len // self.key_metadata.block_size,
            len(request.block_ids),
            len(request.block_hashes),
        )
        if full_blocks == 0:
            self._wait_ready(task)
            return True

        # Ranks holding an identical KV shard share one key namespace and
        # stripe logical blocks across the replica group. Use the absolute
        # request-local ordinal, not the physical cache block ID: physical IDs
        # are allocator slots and can change when blocks are reused.
        owned_ordinals = range(
            self.replica_rank,
            full_blocks,
            self.kv_head_replica_num,
        )
        block_hashes = [request.block_hashes[index] for index in owned_ordinals]
        block_ids = [request.block_ids[index] for index in owned_ordinals]
        if not block_hashes:
            # This save wave still owns the scheduler's block lease. Preserve
            # the same forward-readiness fence before reporting completion even
            # though this rank has no Store query or GPU read to perform.
            self._wait_ready(task)
            return True
        for block_id in block_ids:
            self._validate_block_id(block_id)
        keys = [build_store_key(self.key_metadata, self.key_rank, block_hash) for block_hash in block_hashes]
        missing_indices = self._query_missing(request, keys)
        # Query does not touch GPU memory and can overlap the model write. The
        # job still owns a read lease until this event, even when every key was
        # already present or the query failed.
        self._wait_ready(task)
        if missing_indices is None:
            return False
        if not missing_indices:
            return True
        return self._put_missing(task, keys, missing_indices, block_ids)

    def run(self) -> None:
        while True:
            item = self.request_queue.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, _StoreTask)
                request = item.request
                logger.info(
                    'Mooncake KV save dequeued: global_rank=%d tp_rank=%d tp_size=%d '
                    'req_id=%s save_id=%d generation=%d queue_wait_ms=%.3f',
                    self.global_rank,
                    self.tp_rank,
                    self.tp_size,
                    request.req_id,
                    request.save_id,
                    request.generation,
                    (time.perf_counter() - item.enqueue_time) * 1000,
                )
                success = False
                try:
                    success = self._handle_request(item)
                except Exception:
                    logger.exception(
                        'Mooncake KV save failed unexpectedly: global_rank=%d tp_rank=%d '
                        'req_id=%s save_id=%d',
                        self.global_rank,
                        self.tp_rank,
                        request.req_id,
                        request.save_id,
                    )
                finally:
                    if not item.ready_waited:
                        try:
                            self._wait_ready(item)
                        except Exception:
                            pass
                    self.completion_callback(request.save_id)
                    logger.info(
                        'Mooncake KV save completed: global_rank=%d tp_rank=%d tp_size=%d '
                        'req_id=%s save_id=%d generation=%d status=%s',
                        self.global_rank,
                        self.tp_rank,
                        self.tp_size,
                        request.req_id,
                        request.save_id,
                        request.generation,
                        'ok' if success else 'error',
                    )
            finally:
                self.request_queue.task_done()

    def close(self) -> None:
        """Stop accepting work, drain queued saves, and join exactly once."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self.request_queue.join()
        self.request_queue.put(self._STOP)
        self.request_queue.join()
        self.join()


class KVCacheStoreRecvingThread(threading.Thread):
    """Single-queue background reader for one Mooncake worker rank.

    The scheduler allocates and owns every destination block before a request
    reaches this thread.  ``batch_get_into_multi_buffers`` therefore writes
    directly into registered GPU cache rows without blocking model execution;
    the request may resume only after the resulting load ID is completed by
    every tensor-parallel worker.
    """

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
        completion_callback: Callable[[int, set[int]], None],
    ) -> None:
        super().__init__(name='MooncakeKVCacheStoreReceiver', daemon=True)
        if len(registrations) != len(row_block_sizes) or not registrations:
            raise ValueError('Mooncake receiver requires one block size per registered region')
        if key_metadata.tp_size != tp_size:
            raise ValueError(
                'Mooncake receiver tp_size must match key metadata: '
                f'{tp_size} != {key_metadata.tp_size}')
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(f'tp_rank must be in [0, {tp_size}), got {tp_rank}')
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
        """Enqueue a load without waiting for Store or GPU transfer work."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError('Mooncake KV-cache receiver is closed')
            self.request_queue.put(_LoadTask(request, time.perf_counter()))
        logger.info(
            'Mooncake KV load enqueued: global_rank=%d tp_rank=%d tp_size=%d '
            'req_id=%s load_id=%d generation=%d local_token_len=%d '
            'remote_token_len=%d blocks=%d',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.req_id,
            request.load_id,
            request.generation,
            request.local_token_len,
            request.remote_token_len,
            len(request.block_ids),
        )

    def _validate_request(self, request: MooncakeStoreLoadRequest) -> None:
        block_size = self.key_metadata.block_size
        if request.local_token_len % block_size != 0:
            raise ValueError('load local_token_len must be block aligned')
        if request.remote_token_len % block_size != 0:
            raise ValueError('load remote_token_len must be block aligned')
        expected_blocks = (
            request.remote_token_len - request.local_token_len
        ) // block_size
        if expected_blocks <= 0:
            raise ValueError('load request must contain a non-empty external suffix')
        if len(request.block_ids) != expected_blocks:
            raise ValueError(
                f'load request has {len(request.block_ids)} suffix blocks, '
                f'expected {expected_blocks}')
        if len(request.block_hashes) != expected_blocks:
            raise ValueError(
                f'load request has {len(request.block_hashes)} suffix hashes, '
                f'expected {expected_blocks}')
        for block_id in request.block_ids:
            self._validate_block_id(block_id)

    def _validate_block_id(self, block_id: int) -> None:
        if isinstance(block_id, bool) or not isinstance(block_id, int):
            raise TypeError('physical block IDs must be integers')
        if block_id < 0 or block_id >= self.num_gpu_blocks:
            raise ValueError(
                f'physical block ID {block_id} is outside [0, {self.num_gpu_blocks})')

    def _scatter_block(self, block_id: int) -> tuple[list[int], list[int]]:
        self._validate_block_id(block_id)
        addresses = [
            registration.address + block_id * block_bytes
            for registration, block_bytes in zip(
                self.registrations,
                self.row_block_sizes,
                strict=True,
            )
        ]
        return addresses, list(self.row_block_sizes)

    def _load(self, request: MooncakeStoreLoadRequest) -> set[int]:
        """Load the whole external suffix and return failed physical blocks."""
        self._validate_request(request)

        # KV-head replicas share one key namespace, but every TP rank needs a
        # complete local GPU copy.  Unlike the save path, loads must never use
        # replica-rank striding or block_id % replica_num filtering.
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
            'global_rank=%d tp_rank=%d tp_size=%d req_id=%s load_id=%d generation=%d '
            'local_token_len=%d remote_token_len=%d keys=%d fragments_per_key=%d '
            'bytes=%d first_key=%s last_key=%s',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.req_id,
            request.load_id,
            request.generation,
            request.local_token_len,
            request.remote_token_len,
            len(keys),
            len(self.registrations),
            total_bytes,
            keys[0],
            keys[-1],
        )
        start = time.perf_counter()
        try:
            results = self.store.batch_get_into_multi_buffers(keys, addresses, sizes)
            if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
                raise TypeError('batch_get_into_multi_buffers must return a sequence')
            results = list(results)
            if len(results) != len(keys):
                raise ValueError(
                    f'batch_get_into_multi_buffers returned {len(results)} results '
                    f'for {len(keys)} keys')
            if any(isinstance(result, bool) or not isinstance(result, int) for result in results):
                raise TypeError('batch_get_into_multi_buffers returned a non-integer result')
        except Exception as e:
            logger.error(
                'Mooncake Store interaction after: operation=load_batch_get_into_multi_buffers '
                'global_rank=%d tp_rank=%d tp_size=%d req_id=%s load_id=%d generation=%d '
                'status=error keys=%d fragments_per_key=%d bytes=%d elapsed_ms=%.3f error=%s',
                self.global_rank,
                self.tp_rank,
                self.tp_size,
                request.req_id,
                request.load_id,
                request.generation,
                len(keys),
                len(self.registrations),
                total_bytes,
                (time.perf_counter() - start) * 1000,
                e,
                exc_info=True,
            )
            return set(request.block_ids)

        failed_indices = [index for index, result in enumerate(results) if result < 0]
        failed_block_ids = {request.block_ids[index] for index in failed_indices}
        status = 'ok' if not failed_indices else 'partial_failure'
        log = logger.info if not failed_indices else logger.error
        log(
            'Mooncake Store interaction after: operation=load_batch_get_into_multi_buffers '
            'global_rank=%d tp_rank=%d tp_size=%d req_id=%s load_id=%d generation=%d '
            'status=%s keys=%d failed=%d fragments_per_key=%d bytes=%d elapsed_ms=%.3f '
            'result_codes=%s',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.req_id,
            request.load_id,
            request.generation,
            status,
            len(keys),
            len(failed_indices),
            len(self.registrations),
            total_bytes,
            (time.perf_counter() - start) * 1000,
            _result_histogram(results),
        )
        return failed_block_ids

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
                    'req_id=%s load_id=%d generation=%d queue_wait_ms=%.3f',
                    self.global_rank,
                    self.tp_rank,
                    self.tp_size,
                    request.req_id,
                    request.load_id,
                    request.generation,
                    (time.perf_counter() - item.enqueue_time) * 1000,
                )
                failed_block_ids: set[int]
                try:
                    failed_block_ids = self._load(request)
                except Exception:
                    # Validation/address construction failures must still
                    # complete the load so the scheduler can safely fall back
                    # instead of leaving the request parked forever.
                    failed_block_ids = set(request.block_ids)
                    logger.exception(
                        'Mooncake KV load failed unexpectedly: global_rank=%d tp_rank=%d '
                        'req_id=%s load_id=%d',
                        self.global_rank,
                        self.tp_rank,
                        request.req_id,
                        request.load_id,
                    )
                try:
                    self.completion_callback(request.load_id, failed_block_ids)
                except Exception:
                    # A bookkeeping callback must not kill the sole receiver
                    # and strand all subsequently queued load waves.
                    logger.exception(
                        'Mooncake KV load completion callback failed: '
                        'global_rank=%d tp_rank=%d req_id=%s load_id=%d',
                        self.global_rank,
                        self.tp_rank,
                        request.req_id,
                        request.load_id,
                    )
                logger.info(
                    'Mooncake KV load completed: global_rank=%d tp_rank=%d tp_size=%d '
                    'req_id=%s load_id=%d generation=%d status=%s failed_blocks=%d',
                    self.global_rank,
                    self.tp_rank,
                    self.tp_size,
                    request.req_id,
                    request.load_id,
                    request.generation,
                    'ok' if not failed_block_ids else 'error',
                    len(failed_block_ids),
                )
            finally:
                self.request_queue.task_done()

    def close(self) -> None:
        """Stop accepting work, drain queued loads, and join exactly once."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self.request_queue.join()
        self.request_queue.put(self._STOP)
        self.request_queue.join()
        self.join()
