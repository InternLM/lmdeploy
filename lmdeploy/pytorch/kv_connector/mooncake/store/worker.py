# Copyright (c) OpenMMLab. All rights reserved.
"""Worker-side implementation for the Mooncake Store connector."""

from __future__ import annotations

import fcntl
import os
import socket
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import torch
import zmq

from lmdeploy.pytorch.kv_connector.base import KVCacheValue, RequestId
from lmdeploy.utils import get_logger

from .data import BlobBlockHashes, MooncakeStoreConfig, MooncakeStoreConnectorMetadata, MooncakeStoreRegistration
from .protocol import LOOKUP_MSG, RESET_MSG, RESP_ERR, RESP_OK

if TYPE_CHECKING:
    from lmdeploy.messages import KVTransferConfig
    from lmdeploy.pytorch.config import CacheConfig

logger = get_logger('lmdeploy')

StoreFactory = Callable[[], Any]

_LOOKUP_POLL_INTERVAL_MS = 100
_LOOKUP_RPC_TIMEOUT_MS = 5000


def prepare_lookup_rpc_path(cache_config: CacheConfig) -> str:
    """Create one endpoint before ``CacheConfig`` is copied to workers."""
    transfer_config = cast('KVTransferConfig', cache_config.kv_transfer_config)

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


def _get_lookup_rpc_timeout_ms(cache_config: CacheConfig) -> int:
    transfer_config = cast('KVTransferConfig', cache_config.kv_transfer_config)
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


class MooncakeStoreWorker:
    """Worker-side component of the Mooncake Store connector."""

    def __init__(
        self,
        cache_config: CacheConfig,
        *,
        global_rank: int = 0,
        tp_rank: int = 0,
        tp_size: int = 1,
        store_factory: StoreFactory | None = None,
    ) -> None:
        kv_transfer_config = cast('KVTransferConfig', cache_config.kv_transfer_config)
        if global_rank < 0:
            raise ValueError('global_rank must be non-negative')
        if tp_size <= 0:
            raise ValueError('tp_size must be greater than 0')
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(f'tp_rank must be in [0, {tp_size}), got {tp_rank}')

        self._cache_config = cache_config
        self.global_rank = global_rank
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.lookup_server: LookupKeyServer | None = None

        config_path = kv_transfer_config.kv_connector_extra_config.get('mooncake_config_path')
        self.store_config = MooncakeStoreConfig.load_from_config(config_path)
        local_hostname = _get_local_hostname()
        factory = store_factory if store_factory is not None else _load_mooncake_store_factory()
        self.store: Any | None = self._create_store(factory)
        self._setup_store(self.store, local_hostname)

    def _rank_fields(self) -> tuple[int, int, int]:
        return self.global_rank, self.tp_rank, self.tp_size

    def _start_lookup_server(self) -> None:
        if self.global_rank == 0 and self.lookup_server is None:
            self.lookup_server = LookupKeyServer(self, self._cache_config)

    def _create_store(self, store_factory: StoreFactory) -> Any:
        logger.info(
            'Mooncake Store interaction before: operation=create global_rank=%d tp_rank=%d tp_size=%d',
            *self._rank_fields(),
        )
        start = time.perf_counter()
        store = store_factory()
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
        ret = store.setup(
            local_hostname,
            config.metadata_server,
            config.global_segment_size,
            config.local_buffer_size,
            config.protocol,
            config.device_name,
            config.master_server_address,
        )

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
            if torch.is_tensor(value):
                yield cache_name, value
                continue
            if not isinstance(value, Sequence):
                raise TypeError(f'KV cache {cache_name!r} must be a tensor or a sequence of tensors')
            if not value:
                raise ValueError(f'KV cache {cache_name!r} contains no rows')
            for index, row in enumerate(value):
                if not torch.is_tensor(row):
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
        ret = self.store.register_buffer(registration.address, registration.size)

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

        registrations, backing_storages = self._build_registrations(kv_caches)
        total = len(registrations)
        total_bytes = sum(registration.size for registration in registrations)
        for index, registration in enumerate(registrations, start=1):
            self._register_buffer(registration, index, total)
        self._start_lookup_server()
        logger.info(
            'Mooncake KV cache registration complete: global_rank=%d tp_rank=%d tp_size=%d '
            'backing_storages=%d registered_regions=%d bytes=%d',
            *self._rank_fields(),
            backing_storages,
            total,
            total_bytes,
        )

    def handle_preemptions(self, connector_metadata: MooncakeStoreConnectorMetadata) -> None:
        """Handle no preemption state until transfer support is implemented."""
        return None

    def get_finished(
        self,
        finished_req_ids: set[RequestId],
        connector_metadata: MooncakeStoreConnectorMetadata,
    ) -> tuple[set[RequestId] | None, set[RequestId] | None]:
        """Report no asynchronous completion before transfers are
        implemented."""
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Report no load errors before external loading is implemented."""
        return set()

    def lookup(self, token_len: int, block_hashes: Sequence[bytes]) -> int:
        """Return no external hit until task 6 defines the stored key space."""
        return 0

    def shutdown(self) -> None:
        """Release Mooncake resources exactly once."""
        lookup_server = self.lookup_server
        self.lookup_server = None
        if lookup_server is not None:
            lookup_server.close()

        store = self.store
        self.store = None
        if store is not None:
            self._close_store(store)

    def _close_store(self, store: Any) -> None:
        logger.info(
            'Mooncake Store interaction before: operation=close global_rank=%d tp_rank=%d tp_size=%d',
            *self._rank_fields(),
        )
        start = time.perf_counter()
        ret = store.close()
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
        self.socket_path = prepare_lookup_rpc_path(cache_config)
        self._ipc_path = self.socket_path.removeprefix('ipc://')
        self._lock_path = f'{self._ipc_path}.lock'
        self._lock_fd: int | None = None
        self._owns_ipc_path = False
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._bind_error: BaseException | None = None
        self._closed = False
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None

        self.thread = threading.Thread(
            target=self._process_requests,
            name='MooncakeLookupServer',
            daemon=True,
        )
        self.thread.start()
        if not self._ready_event.wait(timeout=5.0):
            raise RuntimeError('LookupKeyServer did not start within 5 seconds')
        if self._bind_error is not None:
            raise RuntimeError(f'LookupKeyServer failed to bind {self.socket_path!r}') from self._bind_error

    def _acquire_endpoint_lock(self) -> None:
        lock_fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(lock_fd)
            raise RuntimeError(f'lookup endpoint is already in use: {self.socket_path}') from None
        self._lock_fd = lock_fd

    def _release_endpoint_lock(self) -> None:
        lock_fd = self._lock_fd
        self._lock_fd = None
        if lock_fd is None:
            return
        try:
            os.unlink(self._lock_path)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _handle_lookup(self, frames: list[zmq.Frame]) -> bytes:
        if len(frames) != 4:
            raise ValueError(f'lookup request must have 4 frames, got {len(frames)}')
        token_frame = bytes(frames[1])
        hash_len_frame = bytes(frames[2])
        if len(token_frame) != 4 or len(hash_len_frame) != 2:
            raise ValueError('lookup request has an invalid integer frame width')

        token_len = int.from_bytes(token_frame, byteorder='big')
        hash_len = int.from_bytes(hash_len_frame, byteorder='big')
        block_hashes = BlobBlockHashes(frames[3].buffer, hash_len)
        result = self.store_worker.lookup(token_len, block_hashes)
        if (isinstance(result, bool) or not isinstance(result, int) or result < 0 or result > token_len
                or result >= 2**32):
            raise ValueError(f'lookup result must be a u32 integer, got {result!r}')
        return result.to_bytes(4, byteorder='big')

    def _handle_reset(self) -> bytes:
        try:
            store = self.store_worker.store
            store.remove_all(force=True)
            logger.info('Mooncake Store reset via remove_all succeeded.')
            return RESP_OK
        except Exception as e:
            logger.error('Mooncake Store remove_all failed: %s', e, exc_info=True)
            return RESP_ERR

    def _dispatch(self, frames: list[zmq.Frame]) -> bytes:
        if not frames:
            logger.warning('LookupKeyServer received an empty request.')
            return RESP_ERR
        msg_type = bytes(frames[0])
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
            rpc_socket = self.socket
            self._owns_ipc_path = True
            self._ready_event.set()
            while not self._stop_event.is_set():
                if rpc_socket.poll(_LOOKUP_POLL_INTERVAL_MS, zmq.POLLIN) == 0:
                    continue
                frames = rpc_socket.recv_multipart(copy=False)
                rpc_socket.send(self._dispatch(frames))
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
        self.socket_path = prepare_lookup_rpc_path(cache_config)
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
