# Copyright (c) OpenMMLab. All rights reserved.
"""Worker-side implementation for the Mooncake Store connector."""

from __future__ import annotations

import fcntl
import os
import queue
import socket
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import torch
import zmq

from lmdeploy.pytorch.kv_connector.base import KVCacheValue, KVConnectorOutput, RequestId
from lmdeploy.utils import get_logger

from .data import (
    BlobBlockHashes,
    MooncakeStoreConfig,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreKeyMetadata,
    MooncakeStoreLoadRequest,
    MooncakeStoreRegistration,
    MooncakeStoreSaveRequest,
    build_store_key,
    validate_kv_head_replica_num,
)
from .protocol import LOOKUP_MSG, RESET_MSG, RESP_ERR, RESP_OK

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig

logger = get_logger('lmdeploy')

StoreFactory = Callable[[], Any]

_LOOKUP_POLL_INTERVAL_MS = 100
_LOOKUP_RPC_TIMEOUT_MS = 5000


def prepare_lookup_rpc_path(cache_config: CacheConfig) -> str:
    """Create one endpoint before ``CacheConfig`` is copied to workers."""
    transfer_config = cache_config.kv_transfer_config
    if transfer_config is None or not transfer_config.is_kv_transfer_instance:
        raise ValueError('lookup RPC requires an enabled kv_transfer_config')

    extra_config = transfer_config.kv_connector_extra_config
    configured_path = extra_config.get('lookup_rpc_path')
    if configured_path is not None:
        if not isinstance(configured_path, str) or not configured_path.startswith('ipc://'):
            raise ValueError("lookup_rpc_path must be a non-empty 'ipc://' URI")
        if configured_path == 'ipc://':
            raise ValueError("lookup_rpc_path must be a non-empty 'ipc://' URI")
        return configured_path

    rpc_port = extra_config.get('lookup_rpc_port')
    if rpc_port is None:
        socket_path = f'ipc:///tmp/lmd-mc-lookup-{uuid4().hex}.sock'
    else:
        if isinstance(rpc_port, bool) or not isinstance(rpc_port, int) or rpc_port < 0:
            raise ValueError('lookup_rpc_port must be a non-negative integer')
        socket_path = f'ipc:///tmp/lmd-mc-lookup-{rpc_port}-{socket.gethostname()}.sock'
    extra_config['lookup_rpc_path'] = socket_path
    return socket_path


def get_lookup_rpc_path(cache_config: CacheConfig) -> str:
    """Return the endpoint shared by the scheduler and worker rank 0."""
    return prepare_lookup_rpc_path(cache_config)


def _get_lookup_rpc_timeout_ms(cache_config: CacheConfig) -> int:
    transfer_config = cache_config.kv_transfer_config
    assert transfer_config is not None
    timeout_ms = transfer_config.kv_connector_extra_config.get('lookup_rpc_timeout_ms', _LOOKUP_RPC_TIMEOUT_MS)
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
        raise ValueError('lookup_rpc_timeout_ms must be a positive integer')
    return timeout_ms


def _make_zmq_socket(
    context: zmq.Context,
    socket_path: str,
    socket_type: int,
    *,
    bind: bool,
    timeout_ms: int | None = None,
) -> zmq.Socket:
    """Create one zero-linger lookup socket."""
    rpc_socket = context.socket(socket_type)
    rpc_socket.setsockopt(zmq.LINGER, 0)
    if timeout_ms is not None:
        rpc_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        rpc_socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        if bind:
            rpc_socket.bind(socket_path)
        else:
            rpc_socket.connect(socket_path)
    except Exception:
        rpc_socket.close(linger=0)
        raise
    return rpc_socket


def _load_mooncake_store_factory() -> StoreFactory:
    """Import Mooncake only in a worker that actually enables the connector."""
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as e:
        raise ImportError(
            'MooncakeStoreConnector requires the mooncake-transfer-engine package. '
            'Install it before enabling the connector.') from e
    return MooncakeDistributedStore


def _load_mooncake_replicate_config() -> Any:
    """Construct the default replication policy only when the first put
    runs."""
    try:
        from mooncake.store import ReplicateConfig
    except ImportError as e:
        raise ImportError(
            'Mooncake KV-cache saving requires ReplicateConfig from the '
            'mooncake-transfer-engine package.') from e
    return ReplicateConfig()


def _get_local_hostname() -> str:
    """Resolve the local address selected by the host routing table."""
    candidates = (
        (socket.AF_INET, ('8.8.8.8', 80)),
        (socket.AF_INET6, ('2001:4860:4860::8888', 80)),
    )
    for family, remote_address in candidates:
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as sock:
                sock.connect(remote_address)
                return str(sock.getsockname()[0])
        except OSError:
            continue
    raise RuntimeError('cannot determine the local hostname for Mooncake Store')


def _is_tensor(value: object) -> bool:
    """Keep the production tensor check strict while allowing test patching."""
    return isinstance(value, torch.Tensor)


def _result_histogram(results: Sequence[int]) -> dict[int, int]:
    return dict(sorted(Counter(results).items()))


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


class MooncakeStoreWorker:
    """Worker-side component of the Mooncake Store connector."""

    def __init__(
        self,
        cache_config: CacheConfig,
        *,
        global_rank: int = 0,
        tp_rank: int = 0,
        tp_size: int = 1,
        kv_head_replica_num: int = 1,
        store_factory: StoreFactory | None = None,
        replicate_config: Any | None = None,
    ) -> None:
        kv_transfer_config = cache_config.kv_transfer_config
        if kv_transfer_config is None or not kv_transfer_config.is_kv_transfer_instance:
            raise ValueError('MooncakeStoreWorker requires an enabled kv_transfer_config')
        if kv_transfer_config.kv_connector != 'MooncakeStoreConnector':
            raise ValueError(
                f'MooncakeStoreWorker cannot use kv_connector={kv_transfer_config.kv_connector!r}')
        if global_rank < 0:
            raise ValueError('global_rank must be non-negative')
        if tp_size <= 0:
            raise ValueError('tp_size must be greater than 0')
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(f'tp_rank must be in [0, {tp_size}), got {tp_rank}')
        validate_kv_head_replica_num(kv_head_replica_num, tp_size)
        if cache_config.window_size > 0:
            raise ValueError('Mooncake Store saving does not support sliding-window attention')
        if cache_config.states_shapes:
            raise ValueError('Mooncake Store saving does not support linear-attention state caches')

        self._cache_config = cache_config
        self._kv_transfer_config = kv_transfer_config
        self.kv_role = kv_transfer_config.kv_role
        self.global_rank = global_rank
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.kv_head_replica_num = kv_head_replica_num
        self.store: Any | None = None
        self.lookup_server: LookupKeyServer | None = None
        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        self.kv_recv_thread: KVCacheStoreRecvingThread | None = None
        self._registered_regions: tuple[MooncakeStoreRegistration, ...] | None = None
        self._row_block_sizes: tuple[int, ...] | None = None
        self._key_metadata: MooncakeStoreKeyMetadata | None = None
        self._replicate_config = replicate_config
        self._completion_lock = threading.Lock()
        self._inflight_save_ids: set[int] = set()
        self._completed_save_ids: set[int] = set()
        # Save/load IDs are allocated from independent, monotonically
        # increasing scheduler counters.  A watermark rejects delayed
        # duplicate metadata without retaining one tombstone per operation.
        # An operation already in ``_inflight_*_ids`` remains an explicit
        # exception when a higher ID is acknowledged out of order.
        self._acknowledged_save_watermark = -1
        self._inflight_load_ids: set[int] = set()
        self._completed_load_ids: set[int] = set()
        self._failed_load_ids: set[int] = set()
        self._acknowledged_load_watermark = -1
        self._load_error_block_ids: set[int] = set()

        extra_config = kv_transfer_config.kv_connector_extra_config
        self._model_name = extra_config.get(
            'model_name',
            extra_config.get('model_namespace', 'unnamed-model'),
        )
        self._cache_prefix = extra_config.get('cache_prefix', '')
        if not isinstance(self._model_name, str) or not self._model_name:
            raise ValueError('model_name must be a non-empty string')
        if not isinstance(self._cache_prefix, str):
            raise TypeError('cache_prefix must be a string')

        config_path = extra_config.get('mooncake_config_path')
        self.store_config = MooncakeStoreConfig.load_from_config(config_path)
        local_hostname = _get_local_hostname()
        factory = store_factory if store_factory is not None else _load_mooncake_store_factory()
        store = self._create_store(factory)
        try:
            self._setup_store(store, local_hostname)
        except Exception:
            self._close_store(store)
            raise
        self.store = store

    def _rank_fields(self) -> tuple[int, int, int]:
        return self.global_rank, self.tp_rank, self.tp_size

    def _start_lookup_server(self) -> None:
        if self.global_rank == 0 and self.lookup_server is None:
            self.lookup_server = LookupKeyServer(self, self._cache_config)

    def _prepare_sender_layout(
        self,
        registrations: tuple[MooncakeStoreRegistration, ...],
    ) -> tuple[tuple[int, ...], MooncakeStoreKeyMetadata]:
        num_gpu_blocks = self._cache_config.num_gpu_blocks
        if (isinstance(num_gpu_blocks, bool) or not isinstance(num_gpu_blocks, int)
                or num_gpu_blocks <= 0):
            raise ValueError('num_gpu_blocks must be a positive integer before KV cache registration')
        row_block_sizes = []
        for registration in registrations:
            block_bytes, remainder = divmod(registration.size, num_gpu_blocks)
            if remainder or block_bytes <= 0:
                raise ValueError(
                    f'registered region {registration.name!r} size {registration.size} is not '
                    f'divisible into {num_gpu_blocks} GPU blocks')
            row_block_sizes.append(block_bytes)
        row_block_sizes_tuple = tuple(row_block_sizes)
        key_metadata = MooncakeStoreKeyMetadata(
            model_name=self._model_name,
            cache_prefix=self._cache_prefix,
            tp_size=self.tp_size,
            block_size=self._cache_config.block_size,
            kv_head_replica_num=self.kv_head_replica_num,
        )
        return row_block_sizes_tuple, key_metadata

    def _mark_save_finished(self, save_id: int) -> None:
        with self._completion_lock:
            was_inflight = save_id in self._inflight_save_ids
            self._inflight_save_ids.discard(save_id)
            if save_id <= self._acknowledged_save_watermark and not was_inflight:
                return
            self._completed_save_ids.add(save_id)

    def _mark_load_finished(
        self,
        load_id: int,
        failed_block_ids: set[int],
    ) -> None:
        """Publish one sticky receive completion and any failed
        destinations."""
        with self._completion_lock:
            was_inflight = load_id in self._inflight_load_ids
            self._inflight_load_ids.discard(load_id)
            if load_id <= self._acknowledged_load_watermark and not was_inflight:
                return
            self._completed_load_ids.add(load_id)
            if failed_block_ids:
                self._failed_load_ids.add(load_id)
                self._load_error_block_ids.update(failed_block_ids)

    def _start_sender(self) -> None:
        if self.kv_send_thread is not None:
            return
        registrations = self._registered_regions
        row_block_sizes = self._row_block_sizes
        key_metadata = self._key_metadata
        if registrations is None or row_block_sizes is None or key_metadata is None:
            raise RuntimeError('Mooncake sender cannot start before KV cache registration')
        assert self.store is not None
        sender = KVCacheStoreSendingThread(
            store=self.store,
            registrations=registrations,
            row_block_sizes=row_block_sizes,
            num_gpu_blocks=self._cache_config.num_gpu_blocks,
            key_metadata=key_metadata,
            global_rank=self.global_rank,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            completion_callback=self._mark_save_finished,
            replicate_config=self._replicate_config,
        )
        sender.start()
        self.kv_send_thread = sender

    def _start_receiver(self) -> None:
        if self.kv_role not in ('kv_consumer', 'kv_both'):
            return
        if self.kv_recv_thread is not None:
            return
        registrations = self._registered_regions
        row_block_sizes = self._row_block_sizes
        key_metadata = self._key_metadata
        if registrations is None or row_block_sizes is None or key_metadata is None:
            raise RuntimeError('Mooncake receiver cannot start before KV cache registration')
        assert self.store is not None
        receiver = KVCacheStoreRecvingThread(
            store=self.store,
            registrations=registrations,
            row_block_sizes=row_block_sizes,
            num_gpu_blocks=self._cache_config.num_gpu_blocks,
            key_metadata=key_metadata,
            global_rank=self.global_rank,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            completion_callback=self._mark_load_finished,
        )
        receiver.start()
        self.kv_recv_thread = receiver

    def _create_store(self, store_factory: StoreFactory) -> Any:
        logger.info(
            'Mooncake Store interaction before: operation=create global_rank=%d tp_rank=%d tp_size=%d',
            *self._rank_fields(),
        )
        start = time.perf_counter()
        try:
            store = store_factory()
        except Exception as e:
            logger.error(
                'Mooncake Store interaction after: operation=create global_rank=%d tp_rank=%d tp_size=%d '
                'status=error elapsed_ms=%.3f error=%s',
                *self._rank_fields(),
                (time.perf_counter() - start) * 1000,
                e,
                exc_info=True,
            )
            raise RuntimeError('failed to create MooncakeDistributedStore') from e
        logger.info(
            'Mooncake Store interaction after: operation=create global_rank=%d tp_rank=%d tp_size=%d '
            'status=ok elapsed_ms=%.3f',
            *self._rank_fields(),
            (time.perf_counter() - start) * 1000,
        )
        return store

    def _setup_store(self, store: Any, local_hostname: str) -> None:
        config = self.store_config
        logger.info(
            'Mooncake Store interaction before: operation=setup global_rank=%d tp_rank=%d tp_size=%d '
            'local_hostname=%s metadata_server=%s global_segment_size=%d local_buffer_size=%d protocol=%s '
            'device_name=%s master_server_address=%s',
            *self._rank_fields(),
            local_hostname,
            config.metadata_server,
            config.global_segment_size,
            config.local_buffer_size,
            config.protocol,
            config.device_name,
            config.master_server_address,
        )
        start = time.perf_counter()
        try:
            ret = store.setup(
                local_hostname,
                config.metadata_server,
                config.global_segment_size,
                config.local_buffer_size,
                config.protocol,
                config.device_name,
                config.master_server_address,
            )
        except Exception as e:
            logger.error(
                'Mooncake Store interaction after: operation=setup global_rank=%d tp_rank=%d tp_size=%d '
                'status=error elapsed_ms=%.3f error=%s',
                *self._rank_fields(),
                (time.perf_counter() - start) * 1000,
                e,
                exc_info=True,
            )
            raise RuntimeError('MooncakeDistributedStore.setup raised an exception') from e

        status = 'ok' if ret == 0 else 'error'
        log = logger.info if ret == 0 else logger.error
        log(
            'Mooncake Store interaction after: operation=setup global_rank=%d tp_rank=%d tp_size=%d '
            'status=%s elapsed_ms=%.3f ret=%s',
            *self._rank_fields(),
            status,
            (time.perf_counter() - start) * 1000,
            ret,
        )
        if ret != 0:
            raise RuntimeError(f'MooncakeDistributedStore.setup failed with return code {ret}')

    @staticmethod
    def _iter_cache_rows(kv_caches: Mapping[str, KVCacheValue]):
        for cache_name, value in kv_caches.items():
            if not isinstance(cache_name, str) or not cache_name:
                raise ValueError('KV cache names must be non-empty strings')
            if _is_tensor(value):
                yield cache_name, value
                continue
            if not isinstance(value, Sequence):
                raise TypeError(f'KV cache {cache_name!r} must be a tensor or a sequence of tensors')
            if not value:
                raise ValueError(f'KV cache {cache_name!r} contains no rows')
            for index, row in enumerate(value):
                if not _is_tensor(row):
                    raise TypeError(f'KV cache {cache_name!r} row {index} is not a tensor')
                yield f'{cache_name}[{index}]', row

    @classmethod
    def _build_registrations(
        cls,
        kv_caches: Mapping[str, KVCacheValue],
    ) -> tuple[tuple[MooncakeStoreRegistration, ...], int]:
        registrations = []
        backing_storages = set()
        for name, row in cls._iter_cache_rows(kv_caches):
            if not row.is_cuda:
                raise ValueError(f'KV cache row {name!r} must be a CUDA tensor')
            if not row.is_contiguous():
                raise ValueError(f'KV cache row {name!r} must be contiguous')
            registrations.append(
                MooncakeStoreRegistration(
                    name=name,
                    address=int(row.data_ptr()),
                    size=int(row.numel()) * int(row.element_size()),
                ))
            backing_storages.add(int(row.untyped_storage().data_ptr()))
        return tuple(registrations), len(backing_storages)

    def _register_buffer(
        self,
        registration: MooncakeStoreRegistration,
        index: int,
        total: int,
    ) -> None:
        logger.info(
            'Mooncake Store interaction before: operation=register_buffer global_rank=%d tp_rank=%d tp_size=%d '
            'index=%d/%d name=%s addr=%#x bytes=%d',
            *self._rank_fields(),
            index,
            total,
            registration.name,
            registration.address,
            registration.size,
        )
        start = time.perf_counter()
        try:
            ret = self.store.register_buffer(registration.address, registration.size)
        except Exception as e:
            logger.error(
                'Mooncake Store interaction after: operation=register_buffer global_rank=%d tp_rank=%d '
                'tp_size=%d index=%d/%d name=%s status=error elapsed_ms=%.3f error=%s',
                *self._rank_fields(),
                index,
                total,
                registration.name,
                (time.perf_counter() - start) * 1000,
                e,
                exc_info=True,
            )
            raise RuntimeError(f'Mooncake register_buffer raised for {registration.name!r}') from e

        status = 'ok' if ret == 0 else 'error'
        log = logger.info if ret == 0 else logger.error
        log(
            'Mooncake Store interaction after: operation=register_buffer global_rank=%d tp_rank=%d tp_size=%d '
            'index=%d/%d name=%s status=%s elapsed_ms=%.3f ret=%s',
            *self._rank_fields(),
            index,
            total,
            registration.name,
            status,
            (time.perf_counter() - start) * 1000,
            ret,
        )
        if ret != 0:
            raise RuntimeError(
                f'Mooncake register_buffer failed for {registration.name!r} with return code {ret}')

    def register_kv_caches(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        """Register each contiguous physical KV-cache row with Mooncake."""
        if not kv_caches:
            raise ValueError('No KV cache rows were provided for Mooncake Store registration')
        if self.store is None:
            raise RuntimeError('MooncakeDistributedStore is not available')

        registrations, backing_storages = self._build_registrations(kv_caches)
        if self._registered_regions is not None:
            if frozenset(registrations) == frozenset(self._registered_regions):
                try:
                    self._start_sender()
                    self._start_receiver()
                    self._start_lookup_server()
                except Exception:
                    self.shutdown()
                    raise
                logger.info(
                    'Mooncake KV cache registration already complete; skipping identical mapping: '
                    'global_rank=%d tp_rank=%d tp_size=%d regions=%d',
                    *self._rank_fields(),
                    len(registrations),
                )
                return None
            raise RuntimeError('Mooncake KV caches were already registered with a different mapping')

        row_block_sizes, key_metadata = self._prepare_sender_layout(registrations)
        total = len(registrations)
        total_bytes = sum(registration.size for registration in registrations)
        try:
            for index, registration in enumerate(registrations, start=1):
                self._register_buffer(registration, index, total)
        except Exception:
            self.shutdown()
            raise

        self._registered_regions = registrations
        self._row_block_sizes = row_block_sizes
        self._key_metadata = key_metadata
        try:
            self._start_sender()
            self._start_receiver()
            self._start_lookup_server()
        except Exception:
            self.shutdown()
            raise
        logger.info(
            'Mooncake KV cache registration complete: global_rank=%d tp_rank=%d tp_size=%d '
            'backing_storages=%d registered_regions=%d bytes=%d',
            *self._rank_fields(),
            backing_storages,
            total,
            total_bytes,
        )
        return None

    def handle_preemptions(self, connector_metadata: MooncakeStoreConnectorMetadata) -> None:
        """Finish unsubmitted preempted waves without cancelling GPU
        readers."""
        with self._completion_lock:
            for save_id in connector_metadata.preempted_save_ids:
                if (
                    save_id not in self._inflight_save_ids
                    and save_id > self._acknowledged_save_watermark
                ):
                    self._completed_save_ids.add(save_id)
        return None

    @staticmethod
    def has_pending_step_transfers(
        connector_metadata: MooncakeStoreConnectorMetadata,
    ) -> bool:
        """Return whether this step has load or save requests to enqueue."""
        return bool(
            connector_metadata.load_requests
            or connector_metadata.save_requests
        )

    @staticmethod
    def has_pending_step_loads(
        connector_metadata: MooncakeStoreConnectorMetadata,
    ) -> bool:
        """Return whether this step has load requests to enqueue."""
        return bool(connector_metadata.load_requests)

    @staticmethod
    def has_pending_step_saves(
        connector_metadata: MooncakeStoreConnectorMetadata,
    ) -> bool:
        """Return whether this step has save requests to enqueue."""
        return bool(connector_metadata.save_requests)

    def submit_loads(
        self,
        connector_metadata: MooncakeStoreConnectorMetadata,
    ) -> None:
        """Submit this step's loads before their requests can run forward."""
        receiver = self.kv_recv_thread
        if connector_metadata.load_requests and receiver is None:
            raise RuntimeError('Mooncake KV caches must be registered before submitting loads')

        assert receiver is not None or not connector_metadata.load_requests
        for request in connector_metadata.load_requests:
            with self._completion_lock:
                if (
                    request.load_id in self._inflight_load_ids
                    or request.load_id in self._completed_load_ids
                    or request.load_id <= self._acknowledged_load_watermark
                ):
                    continue
                self._inflight_load_ids.add(request.load_id)
            try:
                assert receiver is not None
                receiver.add_request(request)
            except Exception:
                # A submission failure is terminal for the load wave. Publish
                # it like a Store GET failure so the scheduler can unpin the
                # blocks and fall back instead of waiting forever.
                self._mark_load_finished(request.load_id, set(request.block_ids))
                raise
        return None

    def submit_saves(
        self,
        connector_metadata: MooncakeStoreConnectorMetadata,
        *,
        save_ready_event: Any | None = None,
    ) -> None:
        """Submit this step's save waves without polling completions."""
        sender = self.kv_send_thread
        if connector_metadata.save_requests and sender is None:
            raise RuntimeError('Mooncake KV caches must be registered before submitting saves')

        assert sender is not None or not connector_metadata.save_requests
        for request in connector_metadata.save_requests:
            with self._completion_lock:
                if (
                    request.save_id in self._inflight_save_ids
                    or request.save_id in self._completed_save_ids
                    or request.save_id <= self._acknowledged_save_watermark
                ):
                    continue
                self._inflight_save_ids.add(request.save_id)
            try:
                assert sender is not None
                sender.add_request(request, save_ready_event)
            except Exception:
                self._mark_save_finished(request.save_id)
                raise
        return None

    def submit_transfers(
        self,
        connector_metadata: MooncakeStoreConnectorMetadata,
        *,
        save_ready_event: Any | None = None,
    ) -> None:
        """Compatibility hook submitting loads first, then save waves."""
        self.submit_loads(connector_metadata)
        self.submit_saves(
            connector_metadata,
            save_ready_event=save_ready_event,
        )
        return None

    def get_finished(
        self,
        finished_req_ids: set[RequestId],
        connector_metadata: MooncakeStoreConnectorMetadata,
        *,
        ready_event: Any | None = None,
    ) -> KVConnectorOutput:
        """Compatibility wrapper combining submission and sticky polling."""
        del finished_req_ids
        self.submit_transfers(
            connector_metadata,
            save_ready_event=ready_event,
        )
        return self.poll_finished()

    @staticmethod
    def _validate_acknowledged_ids(
        acknowledged_ids: set[int],
        field_name: str,
    ) -> None:
        if any(
            isinstance(operation_id, bool)
            or not isinstance(operation_id, int)
            or operation_id < 0
            for operation_id in acknowledged_ids
        ):
            raise ValueError(f'{field_name} must contain non-negative integers')

    def poll_finished(
        self,
        acknowledged_sending: set[int] | None = None,
        acknowledged_recving: set[int] | None = None,
    ) -> KVConnectorOutput:
        """Acknowledge and poll sticky save/load operation completions."""
        acknowledged_sending = acknowledged_sending or set()
        acknowledged_recving = acknowledged_recving or set()
        self._validate_acknowledged_ids(
            acknowledged_sending,
            'acknowledged_sending',
        )
        self._validate_acknowledged_ids(
            acknowledged_recving,
            'acknowledged_recving',
        )
        with self._completion_lock:
            self._completed_save_ids.difference_update(acknowledged_sending)
            if acknowledged_sending:
                self._acknowledged_save_watermark = max(
                    self._acknowledged_save_watermark,
                    max(acknowledged_sending),
                )
            self._completed_load_ids.difference_update(acknowledged_recving)
            self._failed_load_ids.difference_update(acknowledged_recving)
            if acknowledged_recving:
                self._acknowledged_load_watermark = max(
                    self._acknowledged_load_watermark,
                    max(acknowledged_recving),
                )
            return KVConnectorOutput(
                completed_save_ids=set(self._completed_save_ids),
                completed_load_ids=set(self._completed_load_ids),
                failed_load_ids=set(self._failed_load_ids),
            )

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Consume physical GPU block IDs whose latest Store GET failed."""
        with self._completion_lock:
            failed_block_ids = set(self._load_error_block_ids)
            self._load_error_block_ids.clear()
        return failed_block_ids

    def lookup(self, token_len: int, block_hashes: Sequence[bytes]) -> int:
        """Return the longest prefix present for every unique KV-head shard."""
        if isinstance(token_len, bool) or not isinstance(token_len, int) or token_len < 0:
            raise ValueError('token_len must be a non-negative integer')
        store = self.store
        key_metadata = self._key_metadata
        if store is None or key_metadata is None:
            logger.warning('Mooncake lookup skipped before KV cache registration.')
            return 0

        full_blocks = min(token_len // key_metadata.block_size, len(block_hashes))
        if full_blocks == 0:
            return 0
        unique_kv_ranks = key_metadata.num_kv_head_shards
        keys = [
            build_store_key(key_metadata, rank, block_hashes[block_index])
            for block_index in range(full_blocks)
            for rank in range(unique_kv_ranks)
        ]
        logger.info(
            'Mooncake Store interaction before: operation=lookup_batch_is_exist '
            'global_rank=%d tp_rank=%d tp_size=%d token_len=%d blocks=%d candidate_keys=%d '
            'first_key=%s last_key=%s',
            *self._rank_fields(),
            token_len,
            full_blocks,
            len(keys),
            keys[0],
            keys[-1],
        )
        start = time.perf_counter()
        try:
            exists_states = store.batch_is_exist(keys)
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
                'Mooncake Store interaction after: operation=lookup_batch_is_exist '
                'global_rank=%d tp_rank=%d tp_size=%d token_len=%d blocks=%d candidate_keys=%d '
                'status=error elapsed_ms=%.3f error=%s',
                *self._rank_fields(),
                token_len,
                full_blocks,
                len(keys),
                (time.perf_counter() - start) * 1000,
                e,
                exc_info=True,
            )
            return 0

        matched_blocks = 0
        for block_index in range(full_blocks):
            offset = block_index * unique_kv_ranks
            if not all(
                    exists_states[offset + rank] == 1
                    for rank in range(unique_kv_ranks)):
                break
            matched_blocks += 1
        logger.info(
            'Mooncake Store interaction after: operation=lookup_batch_is_exist '
            'global_rank=%d tp_rank=%d tp_size=%d token_len=%d blocks=%d candidate_keys=%d '
            'status=ok matched_blocks=%d matched_tokens=%d elapsed_ms=%.3f result_codes=%s',
            *self._rank_fields(),
            token_len,
            full_blocks,
            len(keys),
            matched_blocks,
            matched_blocks * key_metadata.block_size,
            (time.perf_counter() - start) * 1000,
            _result_histogram(exists_states),
        )
        return matched_blocks * key_metadata.block_size

    def shutdown(self) -> None:
        """Release Mooncake resources exactly once."""
        lookup_server = self.lookup_server
        self.lookup_server = None
        recv_thread = self.kv_recv_thread
        self.kv_recv_thread = None
        send_thread = self.kv_send_thread
        self.kv_send_thread = None
        store = self.store
        self.store = None
        self._registered_regions = None
        self._row_block_sizes = None
        self._key_metadata = None
        try:
            if lookup_server is not None:
                lookup_server.close()
        finally:
            try:
                if recv_thread is not None:
                    recv_thread.close()
            finally:
                try:
                    if send_thread is not None:
                        send_thread.close()
                finally:
                    if store is not None:
                        self._close_store(store)
        return None

    def _close_store(self, store: Any) -> None:
        logger.info(
            'Mooncake Store interaction before: operation=close global_rank=%d tp_rank=%d tp_size=%d',
            *self._rank_fields(),
        )
        start = time.perf_counter()
        try:
            ret = store.close()
        except Exception as e:
            logger.warning(
                'Mooncake Store interaction after: operation=close global_rank=%d tp_rank=%d tp_size=%d '
                'status=error elapsed_ms=%.3f error=%s',
                *self._rank_fields(),
                (time.perf_counter() - start) * 1000,
                e,
                exc_info=True,
            )
            return
        status = 'ok' if ret in (None, 0) else 'error'
        log = logger.info if status == 'ok' else logger.warning
        log(
            'Mooncake Store interaction after: operation=close global_rank=%d tp_rank=%d tp_size=%d '
            'status=%s elapsed_ms=%.3f ret=%s',
            *self._rank_fields(),
            status,
            (time.perf_counter() - start) * 1000,
            ret,
        )


class LookupKeyServer:
    """ZMQ lookup server owned by Mooncake worker rank 0."""

    def __init__(self, store_worker: MooncakeStoreWorker, cache_config: CacheConfig) -> None:
        self.store_worker = store_worker
        self.socket_path = get_lookup_rpc_path(cache_config)
        self._ipc_path = self.socket_path.removeprefix('ipc://')
        self._lock_path = f'{self._ipc_path}.lock'
        self._lock_fd: int | None = None
        self._owns_ipc_path = False
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._bind_error: BaseException | None = None
        self._closed = False
        self.running = True
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None

        self.thread = threading.Thread(
            target=self._process_requests,
            name='MooncakeLookupServer',
            daemon=True,
        )
        self.thread.start()
        if not self._ready_event.wait(timeout=5.0):
            self._stop_event.set()
            self.thread.join()
            raise RuntimeError('LookupKeyServer did not start within 5 seconds')
        if self._bind_error is not None:
            self.thread.join()
            raise RuntimeError(f'LookupKeyServer failed to bind {self.socket_path!r}') from self._bind_error

    @staticmethod
    def _frame_bytes(frame: Any) -> bytes:
        return bytes(frame)

    def _acquire_endpoint_lock(self) -> None:
        lock_fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            os.close(lock_fd)
            raise RuntimeError(f'lookup endpoint is already in use: {self.socket_path}') from None
        self._lock_fd = lock_fd

    def _release_endpoint_lock(self) -> None:
        lock_fd = self._lock_fd
        self._lock_fd = None
        if lock_fd is None:
            return
        try:
            if os.path.exists(self._lock_path):
                os.unlink(self._lock_path)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _handle_lookup(self, frames: list[Any]) -> bytes:
        if len(frames) != 4:
            raise ValueError(f'lookup request must have 4 frames, got {len(frames)}')
        token_frame = self._frame_bytes(frames[1])
        hash_len_frame = self._frame_bytes(frames[2])
        if len(token_frame) != 4 or len(hash_len_frame) != 2:
            raise ValueError('lookup request has an invalid integer frame width')

        token_len = int.from_bytes(token_frame, byteorder='big')
        hash_len = int.from_bytes(hash_len_frame, byteorder='big')
        payload_frame = frames[3]
        if hasattr(payload_frame, 'buffer'):
            payload = payload_frame.buffer
        else:
            payload = memoryview(self._frame_bytes(payload_frame))
        block_hashes = BlobBlockHashes(payload, hash_len)
        result = self.store_worker.lookup(token_len, block_hashes)
        if (isinstance(result, bool) or not isinstance(result, int) or result < 0 or result > token_len
                or result >= 2**32):
            raise ValueError(f'lookup result must be a u32 integer, got {result!r}')
        return result.to_bytes(4, byteorder='big')

    def _handle_reset(self) -> bytes:
        rank_fields_fn = getattr(self.store_worker, '_rank_fields', None)
        rank_fields = rank_fields_fn() if rank_fields_fn is not None else (0, 0, 1)
        logger.info(
            'Mooncake Store interaction before: operation=remove_all global_rank=%d '
            'tp_rank=%d tp_size=%d force=true',
            *rank_fields,
        )
        start = time.perf_counter()
        try:
            recv_thread = getattr(self.store_worker, 'kv_recv_thread', None)
            if recv_thread is not None:
                recv_thread.request_queue.join()
            send_thread = getattr(self.store_worker, 'kv_send_thread', None)
            if send_thread is not None:
                send_thread.request_queue.join()
            store = self.store_worker.store
            if store is None:
                raise RuntimeError('MooncakeDistributedStore is not available')
            store.remove_all(force=True)
            logger.info(
                'Mooncake Store interaction after: operation=remove_all global_rank=%d '
                'tp_rank=%d tp_size=%d force=true status=ok elapsed_ms=%.3f',
                *rank_fields,
                (time.perf_counter() - start) * 1000,
            )
            return RESP_OK
        except Exception as e:
            logger.error(
                'Mooncake Store interaction after: operation=remove_all global_rank=%d '
                'tp_rank=%d tp_size=%d force=true status=error elapsed_ms=%.3f error=%s',
                *rank_fields,
                (time.perf_counter() - start) * 1000,
                e,
                exc_info=True,
            )
            return RESP_ERR

    def _dispatch(self, frames: list[Any]) -> bytes:
        if not frames:
            logger.warning('LookupKeyServer received an empty request.')
            return RESP_ERR
        msg_type = self._frame_bytes(frames[0])
        if msg_type == LOOKUP_MSG:
            try:
                return self._handle_lookup(frames)
            except Exception as e:
                logger.error('Mooncake lookup request failed: %s', e, exc_info=True)
                return (0).to_bytes(4, byteorder='big')
        if msg_type == RESET_MSG:
            if len(frames) != 1:
                logger.warning('LookupKeyServer received a malformed reset request.')
                return RESP_ERR
            return self._handle_reset()
        logger.warning('LookupKeyServer received unknown msg_type: %r', msg_type)
        return RESP_ERR

    def _process_requests(self) -> None:
        try:
            self.context = zmq.Context()
            self._acquire_endpoint_lock()
            if os.path.exists(self._ipc_path):
                os.unlink(self._ipc_path)
            self.socket = _make_zmq_socket(self.context, self.socket_path, zmq.REP, bind=True)
            self._owns_ipc_path = True
            self._ready_event.set()
            while not self._stop_event.is_set():
                assert self.socket is not None
                if self.socket.poll(_LOOKUP_POLL_INTERVAL_MS, zmq.POLLIN) == 0:
                    continue
                frames = self.socket.recv_multipart(copy=False)
                self.socket.send(self._dispatch(frames))
        except Exception as e:
            if not self._ready_event.is_set():
                self._bind_error = e
                self._ready_event.set()
                return
            if isinstance(e, zmq.ContextTerminated):
                return
            if isinstance(e, zmq.ZMQError) and self._stop_event.is_set():
                return
            logger.exception('LookupKeyServer request loop failed.')
        finally:
            if self.socket is not None:
                self.socket.close(linger=0)
                self.socket = None
            if self.context is not None:
                self.context.term()
                self.context = None
            if self._owns_ipc_path and os.path.exists(self._ipc_path):
                os.unlink(self._ipc_path)
            self._owns_ipc_path = False
            self._release_endpoint_lock()
            self.running = False

    def close(self) -> None:
        """Stop the server thread and remove its IPC socket path."""
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        self.thread.join()


class LookupKeyClient:
    """Scheduler-side ZMQ client with non-blocking Future polling."""

    def __init__(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.socket_path = get_lookup_rpc_path(cache_config)
        self.timeout_ms = _get_lookup_rpc_timeout_ms(cache_config)
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='MooncakeLookupClient')
        self.futures: dict[RequestId, Future[int]] = {}
        self._closed = False

    def _make_socket(self) -> zmq.Socket:
        assert self.context is not None
        return _make_zmq_socket(
            self.context,
            self.socket_path,
            zmq.REQ,
            bind=False,
            timeout_ms=self.timeout_ms,
        )

    def _ensure_socket(self) -> zmq.Socket:
        if self.socket is None:
            if self.context is None:
                self.context = zmq.Context()
            self.socket = self._make_socket()
        return self.socket

    def _reconnect_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if not self._closed:
            self.socket = self._make_socket()

    def _close_transport(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None

    @staticmethod
    def _encode_lookup(token_len: int, block_hashes: Sequence[bytes]) -> tuple[bytes, bytes, bytes, bytes]:
        if isinstance(token_len, bool) or not isinstance(token_len, int) or token_len < 0 or token_len >= 2**32:
            raise ValueError('token_len must be a u32 integer')

        hashes = []
        for block_hash in block_hashes:
            if not isinstance(block_hash, (bytes, bytearray, memoryview)):
                raise TypeError('block hashes must be bytes-like values')
            hashes.append(bytes(block_hash))
        hash_len = len(hashes[0]) if hashes else 0
        if hashes and hash_len == 0:
            raise ValueError('block hashes must not be empty')
        if hash_len >= 2**16:
            raise ValueError('block hash length must fit in a u16 integer')
        if any(len(block_hash) != hash_len for block_hash in hashes):
            raise ValueError('all block hashes must have the same length')

        return (
            LOOKUP_MSG,
            token_len.to_bytes(4, byteorder='big'),
            hash_len.to_bytes(2, byteorder='big'),
            b''.join(hashes),
        )

    def _lookup(self, token_len: int, block_hashes: Sequence[bytes]) -> int:
        frames = self._encode_lookup(token_len, block_hashes)
        rpc_socket = self._ensure_socket()
        try:
            rpc_socket.send_multipart(frames, copy=False)
            response = rpc_socket.recv()
        except zmq.ZMQError:
            self._reconnect_socket()
            raise
        if len(response) != 4:
            raise RuntimeError(f'lookup response must have 4 bytes, got {len(response)}')
        result = int.from_bytes(response, byteorder='big')
        if result > token_len:
            raise RuntimeError(f'lookup response {result} exceeds token_len {token_len}')
        return result

    def lookup(
        self,
        req_id: RequestId,
        token_len: int,
        block_hashes: Sequence[bytes],
        non_block: bool = True,
    ) -> int | None:
        """Submit once per request and poll without blocking by default."""
        if self._closed:
            raise RuntimeError('LookupKeyClient is closed')
        future = self.futures.get(req_id)
        if future is None:
            future = self.executor.submit(self._lookup, token_len, tuple(block_hashes))
            self.futures[req_id] = future
        if non_block and not future.done():
            return None
        try:
            return future.result()
        except Exception as e:
            logger.error('Asynchronous Mooncake lookup failed for %s: %s', req_id, e, exc_info=True)
            return 0
        finally:
            self.futures.pop(req_id, None)

    def discard(self, req_id: RequestId) -> None:
        """Drop an in-flight or completed lookup for an aborted request."""
        future = self.futures.pop(req_id, None)
        if future is not None:
            future.cancel()

    def _reset(self) -> bool:
        rpc_socket = self._ensure_socket()
        try:
            rpc_socket.send(RESET_MSG)
            response = rpc_socket.recv()
        except zmq.ZMQError:
            self._reconnect_socket()
            raise
        return bytes(response) == RESP_OK

    def reset(self) -> bool:
        if self._closed:
            raise RuntimeError('LookupKeyClient is closed')
        try:
            return self.executor.submit(self._reset).result()
        except Exception as e:
            logger.error('Mooncake reset RPC failed: %s', e, exc_info=True)
            return False

    def close(self) -> None:
        """Cancel queued lookups and release the client socket exactly once."""
        if self._closed:
            return
        self._closed = True
        futures = list(self.futures.values())
        self.futures.clear()
        for future in futures:
            future.cancel()
        close_future = self.executor.submit(self._close_transport)
        close_future.result()
        self.executor.shutdown(wait=True, cancel_futures=True)
