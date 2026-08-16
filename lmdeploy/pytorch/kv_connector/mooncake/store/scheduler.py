# Copyright (c) OpenMMLab. All rights reserved.
"""Scheduler-side skeleton for the Mooncake Store connector."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .data import MooncakeStoreConnectorMetadata
from .worker import LookupKeyClient

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig
    from lmdeploy.pytorch.messages import SchedulerSequence
    from lmdeploy.pytorch.paging.scheduler import SchedulerOutput


class MooncakeStoreScheduler:
    """Scheduler-side component of the Mooncake Store connector."""

    def __init__(self, cache_config: CacheConfig) -> None:
        kv_transfer_config = cache_config.kv_transfer_config
        if kv_transfer_config is None or not kv_transfer_config.is_kv_transfer_instance:
            raise ValueError('MooncakeStoreScheduler requires an enabled kv_transfer_config')

        self._cache_config = cache_config
        self._kv_transfer_config = kv_transfer_config
        self.kv_role = kv_transfer_config.kv_role
        self.lookup_async = True
        self.client: LookupKeyClient | None = LookupKeyClient(cache_config)

    def get_num_new_matched_tokens(
        self,
        request: SchedulerSequence,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Return no external hit until the load path can consume it."""
        return 0, False

    def update_state_after_alloc(
        self,
        request: SchedulerSequence,
        block_ids: Sequence[int],
        num_external_tokens: int,
    ) -> None:
        """Record no allocation state until external loading is implemented."""
        return None

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> MooncakeStoreConnectorMetadata:
        """Build empty, serializable metadata for the current engine step."""
        return MooncakeStoreConnectorMetadata()

    def on_new_request(self, request: SchedulerSequence) -> None:
        """Record no request state until lookup support is implemented."""
        return None

    def update_connector_output(self, connector_output: Any) -> None:
        """Consume no worker output until asynchronous I/O is implemented."""
        return None

    def request_finished(
        self,
        request: SchedulerSequence,
        block_ids: Sequence[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Do not take ownership of finished request blocks yet."""
        request_id = getattr(request, 'seq_id', None)
        if self.client is not None and request_id is not None:
            self.client.discard(request_id)
        return False, None

    def shutdown(self) -> None:
        """Cancel pending lookups and release the scheduler client."""
        client = self.client
        self.client = None
        if client is not None:
            client.close()
        return None
