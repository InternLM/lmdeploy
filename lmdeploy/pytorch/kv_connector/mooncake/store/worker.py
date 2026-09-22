# Copyright (c) OpenMMLab. All rights reserved.
"""Worker-side implementation for the Mooncake Store connector."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from lmdeploy.pytorch.kv_connector.base import KVCacheValue, KVConnectorOutput, RequestId
from lmdeploy.utils import get_logger

from .data import (
    MooncakeStoreConfig,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreKeyMetadata,
    MooncakeStoreRegistration,
    build_store_key,
)
from .lookup import LookupKeyServer
from .worker_threads import KVCacheStoreRecvingThread, KVCacheStoreSendingThread

if TYPE_CHECKING:
    from lmdeploy.messages import KVTransferConfig
    from lmdeploy.pytorch.config import CacheConfig

logger = get_logger('lmdeploy')

StoreFactory = Callable[[], Any]


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
        kv_head_replica_num: int = 1,
        store_factory: StoreFactory | None = None,
        replicate_config: Any = None,
    ) -> None:
        kv_transfer_config = cast('KVTransferConfig', cache_config.kv_transfer_config)
        if global_rank < 0:
            raise ValueError('global_rank must be non-negative')
        if tp_size <= 0:
            raise ValueError('tp_size must be greater than 0')
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(f'tp_rank must be in [0, {tp_size}), got {tp_rank}')
        if cache_config.states_shapes:
            raise ValueError('Mooncake Store does not support linear-attention state caches')
        if cache_config.window_size > 1:
            raise ValueError('Mooncake Store does not support sliding-window KV caches')

        self._cache_config = cache_config
        self.kv_role = kv_transfer_config.kv_role
        self.global_rank = global_rank
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.lookup_server: LookupKeyServer | None = None
        self.kv_recv_thread: KVCacheStoreRecvingThread | None = None
        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        self._registered_regions: tuple[MooncakeStoreRegistration, ...] | None = None
        self._row_block_sizes: tuple[int, ...] | None = None
        self._replicate_config = replicate_config
        self._completion_lock = threading.Lock()
        self._inflight_loads: set[RequestId] = set()
        self._completed_loads: dict[RequestId, set[int]] = {}
        self._inflight_save_ids: set[int] = set()
        self._completed_save_ids: set[int] = set()

        extra_config = kv_transfer_config.kv_connector_extra_config
        self.key_metadata = MooncakeStoreKeyMetadata(
            model_name=extra_config.get('model_name', 'unnamed-model'),
            cache_prefix=extra_config.get('cache_prefix', ''),
            tp_size=tp_size,
            block_size=cache_config.block_size,
            kv_head_replica_num=kv_head_replica_num,
        )

        config_path = extra_config.get('mooncake_config_path')
        self.store_config = MooncakeStoreConfig.load_from_config(config_path)
        local_hostname = _get_local_hostname()
        factory = store_factory if store_factory is not None else _load_mooncake_store_factory()
        self.store: Any | None = self._create_store(factory)
        self._setup_store(self.store, local_hostname)

    def _rank_fields(self) -> tuple[int, int, int]:
        return self.global_rank, self.tp_rank, self.tp_size

    def _start_lookup_server(self) -> None:
        if (self.kv_role in ('kv_consumer', 'kv_both')
                and self.tp_rank == 0 and self.lookup_server is None):
            self.lookup_server = LookupKeyServer(self, self._cache_config)

    def _prepare_transfer_layout(
        self,
        registrations: tuple[MooncakeStoreRegistration, ...],
    ) -> tuple[int, ...]:
        num_gpu_blocks = self._cache_config.num_gpu_blocks
        if num_gpu_blocks <= 0:
            raise ValueError('Mooncake Store requires num_gpu_blocks greater than 0')
        row_block_sizes = []
        for registration in registrations:
            block_size, remainder = divmod(registration.size, num_gpu_blocks)
            if remainder != 0:
                raise ValueError(
                    f'registered region {registration.name!r} size '
                    f'{registration.size} is not divisible by num_gpu_blocks '
                    f'{num_gpu_blocks}')
            if block_size <= 0:
                raise ValueError(
                    f'registered region {registration.name!r} has an empty block')
            row_block_sizes.append(block_size)
        return tuple(row_block_sizes)

    def _mark_load_finished(
        self,
        request_id: RequestId,
        failed_block_ids: set[int],
    ) -> None:
        with self._completion_lock:
            self._inflight_loads.discard(request_id)
            self._completed_loads[request_id] = failed_block_ids

    def _mark_save_finished(self, save_id: int) -> None:
        with self._completion_lock:
            self._inflight_save_ids.discard(save_id)
            self._completed_save_ids.add(save_id)

    def _start_receiver(self) -> None:
        if self.kv_role not in ('kv_consumer', 'kv_both') or self.kv_recv_thread is not None:
            return
        registrations = self._registered_regions
        row_block_sizes = self._row_block_sizes
        if registrations is None or row_block_sizes is None:
            raise RuntimeError('Mooncake receiver cannot start before KV cache registration')
        store = self.store
        if store is None:
            raise RuntimeError('Mooncake Store is closed')

        receiver = KVCacheStoreRecvingThread(
            store=store,
            registrations=registrations,
            row_block_sizes=row_block_sizes,
            num_gpu_blocks=self._cache_config.num_gpu_blocks,
            key_metadata=self.key_metadata,
            global_rank=self.global_rank,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            completion_callback=self._mark_load_finished,
        )
        receiver.start()
        self.kv_recv_thread = receiver

    def _start_sender(self) -> None:
        if self.kv_role not in ('kv_producer', 'kv_both') or self.kv_send_thread is not None:
            return
        registrations = self._registered_regions
        row_block_sizes = self._row_block_sizes
        if registrations is None or row_block_sizes is None:
            raise RuntimeError('Mooncake sender cannot start before KV cache registration')
        store = self.store
        if store is None:
            raise RuntimeError('Mooncake Store is closed')

        sender = KVCacheStoreSendingThread(
            store=store,
            registrations=registrations,
            row_block_sizes=row_block_sizes,
            num_gpu_blocks=self._cache_config.num_gpu_blocks,
            key_metadata=self.key_metadata,
            global_rank=self.global_rank,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            completion_callback=self._mark_save_finished,
            replicate_config=self._replicate_config,
        )
        sender.start()
        self.kv_send_thread = sender

    def _create_store(self, store_factory: StoreFactory) -> Any:
        logger.debug(
            'Mooncake Store interaction before: operation=create global_rank=%d tp_rank=%d tp_size=%d',
            *self._rank_fields(),
        )
        start = time.perf_counter()
        store = store_factory()
        logger.debug(
            'Mooncake Store interaction after: operation=create global_rank=%d tp_rank=%d tp_size=%d '
            'status=ok elapsed_ms=%.3f',
            *self._rank_fields(),
            (time.perf_counter() - start) * 1000,
        )
        return store

    def _setup_store(self, store: Any, local_hostname: str) -> None:
        config = self.store_config
        logger.debug(
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
        log = logger.debug if ret == 0 else logger.error
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
        logger.debug(
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
        log = logger.debug if ret == 0 else logger.error
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
        row_block_sizes = self._prepare_transfer_layout(registrations)
        total = len(registrations)
        total_bytes = sum(registration.size for registration in registrations)
        for index, registration in enumerate(registrations, start=1):
            self._register_buffer(registration, index, total)
        self._registered_regions = registrations
        self._row_block_sizes = row_block_sizes
        self._start_receiver()
        self._start_sender()
        self._start_lookup_server()
        logger.debug(
            'Mooncake KV cache registration complete: global_rank=%d tp_rank=%d tp_size=%d '
            'backing_storages=%d registered_regions=%d bytes=%d',
            *self._rank_fields(),
            backing_storages,
            total,
            total_bytes,
        )

    def start_load_kv(self, connector_metadata: MooncakeStoreConnectorMetadata) -> None:
        """Submit each new request to the background receiver once."""
        receiver = self.kv_recv_thread
        if connector_metadata.load_requests and receiver is None:
            raise RuntimeError('Mooncake KV-cache receiver is not initialized')
        for request in connector_metadata.load_requests:
            request_id = request.request_id
            with self._completion_lock:
                if request_id in self._inflight_loads or request_id in self._completed_loads:
                    continue
                self._inflight_loads.add(request_id)
            assert receiver is not None
            receiver.add_request(request)

    def start_save_kv(self, connector_metadata: MooncakeStoreConnectorMetadata) -> None:
        """Fence the compute stream and submit immutable full-block saves."""
        requests = connector_metadata.save_requests
        if not requests:
            return
        sender = self.kv_send_thread
        if sender is None:
            raise RuntimeError('Mooncake KV-cache sender is not initialized')

        ready_event = torch.cuda.Event()
        ready_event.record()
        for request in requests:
            save_id = request.save_id
            with self._completion_lock:
                if save_id in self._inflight_save_ids or save_id in self._completed_save_ids:
                    continue
                self._inflight_save_ids.add(save_id)
            try:
                sender.add_request(request, ready_event)
            except Exception:
                self._mark_save_finished(save_id)
                raise

    def get_finished(self) -> KVConnectorOutput:
        """Return rank-local terminal transfer progress since the last poll."""
        with self._completion_lock:
            completed_loads = set(self._completed_loads)
            invalid_block_ids = set()
            for failed_blocks in self._completed_loads.values():
                invalid_block_ids.update(failed_blocks)
            self._completed_loads.clear()
            completed_save_ids = set(self._completed_save_ids)
            self._completed_save_ids.clear()
        return KVConnectorOutput(
            completed_save_ids=completed_save_ids or None,
            finished_receiving=completed_loads or None,
            invalid_block_ids=invalid_block_ids,
        )

    def lookup(self, token_len: int, block_hashes: Sequence[bytes]) -> int:
        """Return the longest prefix present for every unique KV-head shard."""
        store = self.store
        if store is None:
            return 0

        key_metadata = self.key_metadata
        full_blocks = min(token_len // key_metadata.block_size, len(block_hashes))
        if full_blocks == 0:
            return 0

        unique_kv_ranks = key_metadata.num_kv_head_shards
        keys = [
            build_store_key(key_metadata, rank, block_hashes[block_index])
            for block_index in range(full_blocks)
            for rank in range(unique_kv_ranks)
        ]
        logger.debug(
            'Mooncake Store interaction before: operation=lookup_batch_is_exist '
            'global_rank=%d tp_rank=%d tp_size=%d token_len=%d blocks=%d candidate_keys=%d',
            *self._rank_fields(),
            token_len,
            full_blocks,
            len(keys),
        )
        start = time.perf_counter()
        try:
            exists_states = self.store.batch_is_exist(keys)
            if len(exists_states) != len(keys):
                raise ValueError(
                    f'batch_is_exist returned {len(exists_states)} states for {len(keys)} keys')
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
        matched_tokens = matched_blocks * key_metadata.block_size
        logger.debug(
            'Mooncake Store interaction after: operation=lookup_batch_is_exist '
            'global_rank=%d tp_rank=%d tp_size=%d token_len=%d blocks=%d candidate_keys=%d '
            'status=ok matched_blocks=%d matched_tokens=%d elapsed_ms=%.3f',
            *self._rank_fields(),
            token_len,
            full_blocks,
            len(keys),
            matched_blocks,
            matched_tokens,
            (time.perf_counter() - start) * 1000,
        )
        return matched_tokens

    def shutdown(self) -> None:
        """Release Mooncake resources exactly once."""
        lookup_server = self.lookup_server
        self.lookup_server = None
        if lookup_server is not None:
            lookup_server.close()

        receiver = self.kv_recv_thread
        self.kv_recv_thread = None
        if receiver is not None:
            receiver.close()

        sender = self.kv_send_thread
        self.kv_send_thread = None
        if sender is not None:
            sender.close()

        store = self.store
        self.store = None
        if store is not None:
            self._close_store(store)

    def _close_store(self, store: Any) -> None:
        logger.debug(
            'Mooncake Store interaction before: operation=close global_rank=%d tp_rank=%d tp_size=%d',
            *self._rank_fields(),
        )
        start = time.perf_counter()
        ret = store.close()
        status = 'ok' if ret in (None, 0) else 'error'
        log = logger.debug if status == 'ok' else logger.warning
        log(
            'Mooncake Store interaction after: operation=close global_rank=%d tp_rank=%d tp_size=%d '
            'status=%s elapsed_ms=%.3f ret=%s',
            *self._rank_fields(),
            status,
            (time.perf_counter() - start) * 1000,
            ret,
        )
