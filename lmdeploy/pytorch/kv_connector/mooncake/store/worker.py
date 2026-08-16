# Copyright (c) OpenMMLab. All rights reserved.
"""Worker-side implementation for the Mooncake Store connector."""

from __future__ import annotations

import socket
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from lmdeploy.pytorch.kv_connector.base import KVCacheValue, RequestId
from lmdeploy.utils import get_logger

from .data import MooncakeStoreConfig, MooncakeStoreConnectorMetadata, MooncakeStoreRegistration

if TYPE_CHECKING:
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


def _is_tensor(value: object) -> bool:
    """Keep the production tensor check strict while allowing test patching."""
    return isinstance(value, torch.Tensor)


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

        self._cache_config = cache_config
        self._kv_transfer_config = kv_transfer_config
        self.kv_role = kv_transfer_config.kv_role
        self.global_rank = global_rank
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.store: Any | None = None
        self._registered_regions: tuple[MooncakeStoreRegistration, ...] | None = None

        config_path = kv_transfer_config.kv_connector_extra_config.get('mooncake_config_path')
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
                logger.info(
                    'Mooncake KV cache registration already complete; skipping identical mapping: '
                    'global_rank=%d tp_rank=%d tp_size=%d regions=%d',
                    *self._rank_fields(),
                    len(registrations),
                )
                return None
            raise RuntimeError('Mooncake KV caches were already registered with a different mapping')

        total = len(registrations)
        total_bytes = sum(registration.size for registration in registrations)
        try:
            for index, registration in enumerate(registrations, start=1):
                self._register_buffer(registration, index, total)
        except Exception:
            self.shutdown()
            raise

        self._registered_regions = registrations
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

    def shutdown(self) -> None:
        """Release Mooncake resources exactly once."""
        store = self.store
        if store is None:
            return None
        self.store = None
        self._registered_regions = None
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
