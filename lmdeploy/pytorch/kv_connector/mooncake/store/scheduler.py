# Copyright (c) OpenMMLab. All rights reserved.
"""Scheduler-side asynchronous lookup for the Mooncake Store connector."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lmdeploy.pytorch.kv_connector.base import KVConnectorOutput, KVLoadResult, RequestId

from .data import (
    MooncakeStoreConnectorMetadata,
    MooncakeStoreLoadRequest,
    build_prefix_block_hashes,
)
from .worker import LookupKeyClient

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig
    from lmdeploy.pytorch.messages import SchedulerSequence
    from lmdeploy.pytorch.paging.scheduler import SchedulerOutput


@dataclass
class _RequestHashTracker:
    """Stable hashes already computed for an append-only request prefix."""

    adapter_identity: str
    block_hashes: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class _LookupPlan:
    """Immutable positive lookup result retained through block allocation."""

    remote_token_len: int
    block_hashes: tuple[bytes, ...]


class MooncakeStoreScheduler:
    """Scheduler-side component of the Mooncake Store connector."""

    def __init__(self, cache_config: CacheConfig) -> None:
        kv_transfer_config = cache_config.kv_transfer_config
        if kv_transfer_config is None or not kv_transfer_config.is_kv_transfer_instance:
            raise ValueError('MooncakeStoreScheduler requires an enabled kv_transfer_config')
        if cache_config.states_shapes:
            raise ValueError('Mooncake Store does not support linear-attention state caches')

        lookup_async = kv_transfer_config.kv_connector_extra_config.get('lookup_async', True)
        if lookup_async is not True:
            raise ValueError('Mooncake Store requires lookup_async=true')

        self._cache_config = cache_config
        self._kv_transfer_config = kv_transfer_config
        self.lookup_async = True
        self.client = LookupKeyClient(cache_config)
        self._request_hash_trackers: dict[int, _RequestHashTracker] = {}
        self._lookup_plans: dict[RequestId, _LookupPlan] = {}
        self._pending_loads: dict[RequestId, MooncakeStoreLoadRequest] = {}
        self._inflight_loads: dict[RequestId, MooncakeStoreLoadRequest] = {}
        self._invalid_block_ids: set[int] = set()
        self._failed_load_requests: set[RequestId] = set()

    def get_num_new_matched_tokens(
        self,
        request: SchedulerSequence,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Poll the remote prefix beyond the locally matched GPU prefix."""
        if not self._kv_transfer_config.is_kv_consumer:
            return 0, False
        if (not request.history_multimodals.empty()
                or len(request.history_embeddings) > 0):
            return 0, False

        block_size = self._cache_config.block_size
        token_len = request.get_prefix_cache_max_match_step()
        token_len = token_len // block_size * block_size
        if token_len < block_size or num_computed_tokens >= token_len:
            return 0, False

        req_id = int(request.seq_id)
        if req_id in self._failed_load_requests:
            return 0, False

        plan = self._lookup_plans.get(req_id)
        if plan is not None:
            if plan.remote_token_len > num_computed_tokens:
                return plan.remote_token_len - num_computed_tokens, True
            self._lookup_plans.pop(req_id)

        block_hashes = self._get_request_block_hashes(
            request,
            req_id,
            token_len,
            block_size,
        )
        remote_token_len = self.client.lookup(
            req_id,
            token_len,
            block_hashes,
            non_block=self.lookup_async,
        )
        if remote_token_len is None:
            return None, False

        num_external_tokens = max(0, int(remote_token_len) - int(num_computed_tokens))
        if num_external_tokens == 0:
            return 0, False
        self._lookup_plans[req_id] = _LookupPlan(
            remote_token_len=int(remote_token_len),
            block_hashes=block_hashes,
        )
        return num_external_tokens, True

    def _get_request_block_hashes(
        self,
        request: SchedulerSequence,
        req_id: int,
        token_len: int,
        block_size: int,
    ) -> tuple[bytes, ...]:
        """Return cached hashes, extending them only for newly full blocks."""
        num_blocks = token_len // block_size
        adapter_identity = request.adapter_name or ''
        tracker = self._request_hash_trackers.get(req_id)
        if tracker is None or tracker.adapter_identity != adapter_identity:
            tracker = _RequestHashTracker(adapter_identity=adapter_identity)
            self._request_hash_trackers[req_id] = tracker

        if num_blocks <= len(tracker.block_hashes):
            return tracker.block_hashes[:num_blocks]

        tracker.block_hashes = build_prefix_block_hashes(
            request.all_ids[:token_len],
            block_size,
            extra_identity=adapter_identity,
            previous_hashes=tracker.block_hashes,
        )
        return tracker.block_hashes

    def update_state_after_alloc(
        self,
        request: SchedulerSequence,
        block_ids: Sequence[int],
        num_external_tokens: int,
    ) -> None:
        """Bind a positive lookup to its scheduler-allocated destinations."""
        if num_external_tokens <= 0:
            return
        req_id = int(request.seq_id)
        plan = self._lookup_plans.pop(req_id)
        block_size = self._cache_config.block_size
        remote_token_len = plan.remote_token_len
        local_token_len = remote_token_len - int(num_external_tokens)
        if (local_token_len < 0 or local_token_len % block_size != 0
                or remote_token_len % block_size != 0):
            raise ValueError('Mooncake load token bounds must be non-negative and block aligned')
        if req_id in self._pending_loads or req_id in self._inflight_loads:
            raise RuntimeError(f'request {req_id} already has an asynchronous load')

        local_block = local_token_len // block_size
        remote_block = remote_token_len // block_size
        block_hashes = plan.block_hashes[local_block:remote_block]
        if len(block_ids) != len(block_hashes):
            raise ValueError(
                f'allocated load blocks ({len(block_ids)}) do not match external hashes '
                f'({len(block_hashes)})')

        load_request = MooncakeStoreLoadRequest(
            request_id=req_id,
            block_ids=block_ids,
            block_hashes=block_hashes,
        )
        self._pending_loads[req_id] = load_request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> MooncakeStoreConnectorMetadata | None:
        """Dispatch new loads and keep emitting polling steps while I/O
        runs."""
        if not self._pending_loads and not self._inflight_loads:
            return None

        load_requests = tuple(self._pending_loads.values())
        self._inflight_loads.update(self._pending_loads)
        self._pending_loads.clear()
        return MooncakeStoreConnectorMetadata(
            load_requests=load_requests,
        )

    def on_new_request(self, request: SchedulerSequence) -> None:
        """Drop stale request-local hashes before first scheduling."""
        req_id = int(request.seq_id)
        self._request_hash_trackers.pop(req_id, None)
        self._lookup_plans.pop(req_id, None)
        self._failed_load_requests.discard(req_id)
        return None

    def is_lookup_pending(self, request_id: int) -> bool:
        """Return whether the request's lookup Future is still running."""
        return self.client.is_pending(int(request_id))

    def cancel_lookup(self, request_id: int) -> None:
        """Cancel only the current lookup, retaining incremental hashes."""
        self.client.discard(int(request_id))

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> tuple[KVLoadResult, ...]:
        """Convert all-TP worker completion into backend-neutral load
        results."""
        self._invalid_block_ids.update(connector_output.invalid_block_ids)
        completed = connector_output.finished_receiving or set()
        results = []
        for req_id in sorted(completed):
            request = self._inflight_loads.pop(req_id, None)
            if request is None:
                continue
            request_blocks = set(request.block_ids)
            failed = not request_blocks.isdisjoint(self._invalid_block_ids)
            self._invalid_block_ids.difference_update(request_blocks)
            if failed:
                self._failed_load_requests.add(req_id)
            results.append(KVLoadResult(request_id=req_id, success=not failed))
        return tuple(results)

    def request_finished(
        self,
        request: SchedulerSequence,
        block_ids: Sequence[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Do not take ownership of finished request blocks yet."""
        req_id = int(request.seq_id)
        self.client.discard(req_id)
        self._request_hash_trackers.pop(req_id, None)
        self._lookup_plans.pop(req_id, None)
        pending = self._pending_loads.pop(req_id, None)
        inflight = self._inflight_loads.pop(req_id, None)
        load_request = pending or inflight
        if load_request is not None:
            self._invalid_block_ids.difference_update(load_request.block_ids)
        self._failed_load_requests.discard(req_id)
        return False, None

    def shutdown(self) -> None:
        """Cancel pending lookups and release the scheduler client."""
        self.client.close()
        self._request_hash_trackers.clear()
        self._lookup_plans.clear()
        self._pending_loads.clear()
        self._inflight_loads.clear()
        self._invalid_block_ids.clear()
        self._failed_load_requests.clear()
