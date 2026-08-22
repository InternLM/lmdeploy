# Copyright (c) OpenMMLab. All rights reserved.
"""Role adapter for the Mooncake Store scheduler and worker components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from lmdeploy.pytorch.kv_connector.base import (
    KVCacheValue,
    KVConnectorBase,
    KVConnectorMetadata,
    KVConnectorRole,
    RequestId,
)

from .data import MooncakeStoreConnectorMetadata
from .scheduler import MooncakeStoreScheduler
from .worker import MooncakeStoreWorker

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig
    from lmdeploy.pytorch.messages import SchedulerSequence
    from lmdeploy.pytorch.paging.scheduler import SchedulerOutput


class MooncakeStoreConnector(KVConnectorBase):
    """Delegate connector calls to the component for this process role."""

    def __init__(
        self,
        role: KVConnectorRole,
        cache_config: CacheConfig,
        *,
        global_rank: int = 0,
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> None:
        super().__init__(role)

        kv_transfer_config = cache_config.kv_transfer_config
        if kv_transfer_config is None or not kv_transfer_config.is_kv_transfer_instance:
            raise ValueError('MooncakeStoreConnector requires an enabled kv_transfer_config')
        if kv_transfer_config.kv_connector != 'MooncakeStoreConnector':
            raise ValueError(
                f'MooncakeStoreConnector cannot use kv_connector={kv_transfer_config.kv_connector!r}')

        self.kv_role = kv_transfer_config.kv_role

        self.connector_scheduler: MooncakeStoreScheduler | None = None
        self.connector_worker: MooncakeStoreWorker | None = None
        if role is KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeStoreScheduler(cache_config)
        else:
            self.connector_worker = MooncakeStoreWorker(
                cache_config,
                global_rank=global_rank,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )

    def _require_scheduler(self) -> MooncakeStoreScheduler:
        scheduler = self.connector_scheduler
        if scheduler is None:
            raise RuntimeError('scheduler-side method called on a worker MooncakeStoreConnector')
        return scheduler

    def _require_worker(self) -> MooncakeStoreWorker:
        worker = self.connector_worker
        if worker is None:
            raise RuntimeError('worker-side method called on a scheduler MooncakeStoreConnector')
        return worker

    # Scheduler-side methods.

    def get_num_new_matched_tokens(
        self,
        request: SchedulerSequence,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return self._require_scheduler().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: SchedulerSequence,
        block_ids: Sequence[int],
        num_external_tokens: int,
    ) -> None:
        return self._require_scheduler().update_state_after_alloc(request, block_ids, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> MooncakeStoreConnectorMetadata:
        return self._require_scheduler().build_connector_meta(scheduler_output)

    def on_new_request(self, request: SchedulerSequence) -> None:
        return self._require_scheduler().on_new_request(request)

    def update_connector_output(self, connector_output: Any) -> None:
        return self._require_scheduler().update_connector_output(connector_output)

    def request_finished(
        self,
        request: SchedulerSequence,
        block_ids: Sequence[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._require_scheduler().request_finished(request, block_ids)

    # Worker-side methods.

    def register_kv_caches(self, kv_caches: Mapping[str, KVCacheValue]) -> None:
        return self._require_worker().register_kv_caches(kv_caches)

    def handle_preemptions(self, connector_metadata: KVConnectorMetadata) -> None:
        if not isinstance(connector_metadata, MooncakeStoreConnectorMetadata):
            raise TypeError('connector_metadata must be a MooncakeStoreConnectorMetadata')
        return self._require_worker().handle_preemptions(connector_metadata)

    def get_finished(
        self,
        finished_req_ids: set[RequestId],
    ) -> tuple[set[RequestId] | None, set[RequestId] | None]:
        worker = self._require_worker()
        connector_metadata = self._get_connector_metadata()
        if not isinstance(connector_metadata, MooncakeStoreConnectorMetadata):
            raise TypeError('bound connector metadata must be a MooncakeStoreConnectorMetadata')
        return worker.get_finished(finished_req_ids, connector_metadata)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self._require_worker().get_block_ids_with_load_errors()

    def shutdown(self) -> None:
        if self.connector_scheduler is not None:
            return self.connector_scheduler.shutdown()
        return self._require_worker().shutdown()
