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
        self.client = LookupKeyClient(cache_config)

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
        self.client.discard(request.seq_id)
        return False, None

    def shutdown(self) -> None:
        """Cancel pending lookups and release the scheduler client."""
        self.client.close()
