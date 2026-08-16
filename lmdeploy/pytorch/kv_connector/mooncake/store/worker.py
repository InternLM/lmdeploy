# Copyright (c) OpenMMLab. All rights reserved.
"""Worker-side skeleton for the Mooncake Store connector."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from lmdeploy.pytorch.kv_connector.base import KVCacheValue, RequestId

from .data import MooncakeStoreConnectorMetadata

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig


class MooncakeStoreWorker:
    """Worker-side component of the Mooncake Store connector."""

    def __init__(self, cache_config: CacheConfig) -> None:
        kv_transfer_config = cache_config.kv_transfer_config
        if kv_transfer_config is None or not kv_transfer_config.is_kv_transfer_instance:
            raise ValueError('MooncakeStoreWorker requires an enabled kv_transfer_config')

        self._cache_config = cache_config
        self._kv_transfer_config = kv_transfer_config
        self.kv_role = kv_transfer_config.kv_role

    def register_kv_caches(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        """Register no cache memory until Mooncake setup is implemented."""
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
        """Release worker resources once Mooncake setup owns any."""
        return None
