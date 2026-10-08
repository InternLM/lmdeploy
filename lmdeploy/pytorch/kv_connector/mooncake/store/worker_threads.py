# Copyright (c) OpenMMLab. All rights reserved.
"""Asynchronous Mooncake Store KV-cache transfer workers."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from lmdeploy.pytorch.kv_connector.base import RequestId
from lmdeploy.utils import get_logger

from .data import (
    MooncakeStoreKeyMetadata,
    MooncakeStoreLoadRequest,
    MooncakeStoreRegistration,
    MooncakeStoreSaveRequest,
    MooncakeStoreStateRegistration,
    build_store_key,
)

logger = get_logger('lmdeploy')


@dataclass(frozen=True)
class _LoadTask:
    request: MooncakeStoreLoadRequest
    enqueue_time: float
    ready_event: Any = None


@dataclass(frozen=True)
class _SaveTask:
    request: MooncakeStoreSaveRequest
    ready_event: Any
    enqueue_time: float


@dataclass(frozen=True)
class _SaveEntry:
    """A Store key bound to a physical FA block or linear-state snapshot."""

    key: str
    cache_kind: Literal['full_attention', 'linear_attention']
    source_index: int


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


def _state_row_layout(
    registrations: tuple[MooncakeStoreStateRegistration, ...],
) -> tuple[tuple[MooncakeStoreRegistration, ...], tuple[int, ...]]:
    """Expand owning pools in the same payload order for save and load.

    Packed pools have one row of slots; layer-major pools have one per layer. Each row remains inside the already
    registered owning storage.
    """
    rows = []
    slot_sizes = []
    for region in registrations:
        row_size = region.slot_count * region.slot_size
        for row in range(region.size // row_size):
            rows.append(MooncakeStoreRegistration(region.name, region.address + row * row_size, row_size))
            slot_sizes.append(region.slot_size)
    return tuple(rows), tuple(slot_sizes)


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
        state_registrations: tuple[MooncakeStoreStateRegistration, ...] = (),
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
        self.state_rows, self.state_slot_sizes = _state_row_layout(state_registrations)
        self.num_state_slots = state_registrations[0].slot_count if state_registrations else 0
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
        logger.info(
            'Mooncake KV save enqueued: global_rank=%d tp_rank=%d tp_size=%d '
            'save_id=%d request_id=%s blocks=%d state_boundary=%s state_bytes=%d',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.save_id,
            request.request_id,
            len(request.block_ids),
            request.state.boundary_tokens if request.state is not None else None,
            sum(self.state_slot_sizes) if request.state is not None else 0,
        )

    def _owned_entries(
        self,
        request: MooncakeStoreSaveRequest,
    ) -> list[_SaveEntry]:
        if not (len(request.block_ids) == len(request.block_hashes)
                == len(request.logical_block_ids)):
            raise ValueError('Mooncake save request block fields must have equal lengths')

        replica_num = self.key_metadata.kv_head_replica_num
        entries = []
        for suffix_index, (block_id, block_hash) in enumerate(
                zip(request.block_ids, request.block_hashes, strict=True)):
            absolute_block = request.start_block + suffix_index
            if absolute_block % replica_num != self.replica_rank:
                continue
            entries.append(_SaveEntry(
                key=build_store_key(self.key_metadata, self.key_rank, block_hash),
                cache_kind='full_attention',
                source_index=block_id,
            ))
        if request.state is not None:
            # State uses every attention TP rank. It must not inherit FA's
            # replicated-KV-head ownership rule.
            entries.append(_SaveEntry(
                key=build_store_key(self.key_metadata, self.tp_rank, request.block_hashes[-1], group_id=1),
                cache_kind='linear_attention',
                source_index=request.state.snapshot_slot,
            ))
        return entries

    def _find_missing(self, request: MooncakeStoreSaveRequest, keys: list[str]) -> list[int]:
        logger.info(
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
        logger.info(
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
        entries: list[_SaveEntry],
        missing: list[int],
    ) -> bool:
        missing_entries = [entries[index] for index in missing]
        missing_keys = [entry.key for entry in missing_entries]
        addresses = []
        sizes = []
        for entry in missing_entries:
            if entry.cache_kind == 'full_attention':
                block_addresses, block_sizes = _scatter_block(
                    self.registrations, self.row_block_sizes, self.num_gpu_blocks, entry.source_index)
            else:
                block_addresses, block_sizes = _scatter_block(
                    self.state_rows, self.state_slot_sizes, self.num_state_slots, entry.source_index)
            addresses.append(block_addresses)
            sizes.append(block_sizes)

        total_bytes = sum(sum(block_sizes) for block_sizes in sizes)
        logger.info(
            'Mooncake Store interaction before: operation=save_batch_put_from_multi_buffers '
            'global_rank=%d tp_rank=%d tp_size=%d save_id=%d request_id=%s '
            'keys=%d fragments=%d bytes=%d',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.save_id,
            request.request_id,
            len(missing_keys),
            sum(len(parts) for parts in addresses),
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
        log = logger.info if not failed else logger.error
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
        entries = self._owned_entries(request)
        keys = [entry.key for entry in entries]
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
            return self._put_missing(request, entries, missing)
        return True

    def run(self) -> None:
        while True:
            item = self.request_queue.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, _SaveTask)
                request = item.request
                logger.info(
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
                logger.info(
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
    """Read Mooncake values into private FA blocks and runtime state."""

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
        completion_callback: Callable[[RequestId, bool], None],
        state_registrations: tuple[MooncakeStoreStateRegistration, ...] = (),
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
        self.state_rows, self.state_slot_sizes = _state_row_layout(state_registrations)
        self.num_state_slots = state_registrations[0].slot_count if state_registrations else 0
        self.request_queue: queue.Queue[_LoadTask | object] = queue.Queue()
        self._state_lock = threading.Lock()
        self._closed = False

    def add_request(self, request: MooncakeStoreLoadRequest, ready_event: Any = None) -> None:
        """Enqueue a load without waiting for Store I/O."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError('Mooncake KV-cache receiver is closed')
            self.request_queue.put(_LoadTask(request, time.perf_counter(), ready_event))
        logger.info(
            'Mooncake KV load enqueued: global_rank=%d tp_rank=%d tp_size=%d '
            'request_id=%s blocks=%d state_slot=%s state_boundary=%s',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.request_id,
            len(request.block_ids),
            request.state_slot,
            request.remote_block_count * self.key_metadata.block_size if request.state_slot is not None else None,
        )

    def _scatter_block(self, block_id: int) -> tuple[list[int], list[int]]:
        return _scatter_block(
            self.registrations,
            self.row_block_sizes,
            self.num_gpu_blocks,
            block_id,
        )

    def _load(self, request: MooncakeStoreLoadRequest, ready_event: Any = None) -> bool:
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

        if request.state_slot is not None:
            if request.state_slot <= 0:
                raise ValueError('Mooncake load cannot write the reserved state slot')
            state_addresses, state_sizes = _scatter_block(
                self.state_rows, self.state_slot_sizes, self.num_state_slots, request.state_slot)
            keys.append(build_store_key(self.key_metadata, self.tp_rank, request.block_hashes[-1], group_id=1))
            addresses.append(state_addresses)
            sizes.append(state_sizes)

        total_bytes = sum(sum(block_sizes) for block_sizes in sizes)
        logger.info(
            'Mooncake Store interaction before: operation=load_batch_get_into_multi_buffers '
            'global_rank=%d tp_rank=%d tp_size=%d request_id=%s keys=%d '
            'fragments=%d bytes=%d state_slot=%s',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.request_id,
            len(keys),
            sum(len(parts) for parts in addresses),
            total_bytes,
            request.state_slot,
        )
        # Key/address preparation only touches CPU metadata. Wait after it,
        # before Store can overwrite bytes still in use by earlier GPU work.
        ready_wait_ms = 0.0
        if ready_event is not None:
            ready_wait_start = time.perf_counter()
            ready_event.synchronize()
            ready_wait_ms = (time.perf_counter() - ready_wait_start) * 1000
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
                'keys=%d elapsed_ms=%.3f ready_wait_ms=%.3f error=%s',
                self.global_rank,
                self.tp_rank,
                self.tp_size,
                request.request_id,
                len(keys),
                (time.perf_counter() - start) * 1000,
                ready_wait_ms,
                error,
                exc_info=True,
            )
            return False

        num_failed = sum(result < 0 for result in results)
        state_failed = request.state_slot is not None and results[-1] < 0
        log = logger.info if not num_failed else logger.error
        log(
            'Mooncake Store interaction after: operation=load_batch_get_into_multi_buffers '
            'global_rank=%d tp_rank=%d tp_size=%d request_id=%s status=%s '
            'keys=%d failed=%d state_failed=%s bytes=%d elapsed_ms=%.3f ready_wait_ms=%.3f',
            self.global_rank,
            self.tp_rank,
            self.tp_size,
            request.request_id,
            'ok' if not num_failed else 'partial_failure',
            len(keys),
            num_failed,
            state_failed,
            total_bytes,
            (time.perf_counter() - start) * 1000,
            ready_wait_ms,
        )
        return num_failed == 0

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
                    success = self._load(request, item.ready_event)
                except Exception:
                    success = False
                    logger.exception(
                        'Mooncake KV load failed before Store completion: '
                        'global_rank=%d tp_rank=%d request_id=%s',
                        self.global_rank,
                        self.tp_rank,
                        request.request_id,
                    )
                self.completion_callback(request.request_id, success)
                logger.info(
                    'Mooncake KV load completed: global_rank=%d tp_rank=%d tp_size=%d '
                    'request_id=%s status=%s',
                    self.global_rank,
                    self.tp_rank,
                    self.tp_size,
                    request.request_id,
                    'ok' if success else 'error',
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
