# Copyright (c) OpenMMLab. All rights reserved.
"""Worker-side implementation for the Mooncake Store connector."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from lmdeploy.pytorch.kv_connector.base import KVCacheValue, KVConnectorOutput, RequestId
from lmdeploy.utils import get_logger

from .data import (
    MooncakeStoreConfig,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreKeyMetadata,
    MooncakeStoreRegistration,
    build_store_key,
    validate_kv_head_replica_num,
)
from .lookup import LookupKeyClient, LookupKeyServer
from .utils import (
    StoreFactory,
    _get_local_hostname,
    _is_tensor,
    _load_mooncake_store_factory,
    _result_histogram,
)
from .worker_threads import KVCacheStoreRecvingThread, KVCacheStoreSendingThread

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig

logger = get_logger('lmdeploy')

__all__ = [
    'KVCacheStoreRecvingThread',
    'KVCacheStoreSendingThread',
    'LookupKeyClient',
    'LookupKeyServer',
    'MooncakeStoreWorker',
]

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
        local_hostname = extra_config.get('local_hostname')
        if local_hostname is None:
            local_hostname = _get_local_hostname()
        elif not isinstance(local_hostname, str) or not local_hostname:
            raise ValueError('local_hostname must be a non-empty string')
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
