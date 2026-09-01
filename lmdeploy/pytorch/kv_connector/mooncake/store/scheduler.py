# Copyright (c) OpenMMLab. All rights reserved.
"""Scheduler-side lookup, load, and save lifecycle for Mooncake Store."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lmdeploy.pytorch.kv_connector.base import (
    KVConnectorOutput,
    KVConnectorResult,
    KVLoadResult,
    RequestId,
)

from .data import (
    MooncakeStoreConnectorMetadata,
    MooncakeStoreLoadRequest,
    MooncakeStoreSaveRequest,
    build_prefix_block_hashes,
)
from .lookup import LookupKeyClient

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
    """Positive lookup snapshot waiting for paging block allocation.

    The remote boundary and its block hashes must come from the same lookup.
    ``update_state_after_alloc`` consumes this snapshot exactly once after the
    paging scheduler has assigned destination GPU blocks.
    """

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
        if cache_config.window_size > 1:
            raise ValueError('Mooncake Store does not support sliding-window KV caches')

        if kv_transfer_config.is_kv_consumer:
            lookup_async = kv_transfer_config.kv_connector_extra_config.get('lookup_async', True)
            if lookup_async is not True:
                raise ValueError('Mooncake Store requires lookup_async=true')

        self._cache_config = cache_config
        self._kv_transfer_config = kv_transfer_config
        self.client = (
            LookupKeyClient(cache_config)
            if kv_transfer_config.is_kv_consumer else None
        )
        self._request_hash_trackers: dict[int, _RequestHashTracker] = {}
        # Positive lookup snapshots keyed by request ID. A snapshot can survive
        # several schedule attempts while paging waits for allocation capacity.
        self._lookup_plans: dict[RequestId, _LookupPlan] = {}
        # A request cannot progress while its single active load is pending,
        # so the request ID also uniquely identifies the load operation.
        self._pending_loads: dict[RequestId, MooncakeStoreLoadRequest] = {}
        self._inflight_loads: dict[RequestId, MooncakeStoreLoadRequest] = {}
        self._invalid_block_ids: set[int] = set()
        # A failed load falls back to local compute for the rest of this
        # sequence. A later conversation turn has a new request ID.
        self._failed_load_requests: set[RequestId] = set()
        # Saves do not block request progress. Chunked prefill may therefore
        # create several in-flight saves for one request, each with its own ID.
        self._next_save_id = 0
        # The next block eligible for save in each request. Scheduling advances
        # it optimistically; save failures are retried only by a later request.
        self._next_save_block: dict[RequestId, int] = {}
        self._inflight_save_ids: set[int] = set()

    def get_num_new_matched_tokens(
        self,
        request: SchedulerSequence,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Poll the remote prefix beyond the locally matched GPU prefix.

        ``num_computed_tokens`` is an exact token position and may lie inside a
        block. The returned delta is therefore not necessarily block aligned;
        paging aligns the actual load range before calling
        ``update_state_after_alloc``.
        """
        if not self._kv_transfer_config.is_kv_consumer:
            return 0, False
        if (not request.history_multimodals.empty()
                or len(request.history_embeddings) > 0):
            return 0, False

        block_size = self._cache_config.block_size
        token_len = request.get_prefix_cache_max_match_step()
        # Mooncake stores complete KV blocks, so do not query the incomplete
        # block at the end of the request.
        token_len = token_len // block_size * block_size
        if token_len < block_size or num_computed_tokens >= token_len:
            return 0, False

        req_id = int(request.seq_id)
        if req_id in self._failed_load_requests:
            return 0, False

        plan = self._lookup_plans.get(req_id)
        if plan is not None:
            # Allocation may have failed after an earlier positive lookup. Reuse
            # that result while it still extends the local prefix. Once local KV
            # catches up, discard the stale plan and lookup the current (possibly
            # longer) request prefix below.
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
            non_block=True,
        )
        # ``None`` means the asynchronous RPC is still running; zero is a
        # completed lookup with no remotely reusable suffix.
        if remote_token_len is None:
            return None, False

        # Keep the exact token delta for scheduler accounting. If the local
        # position is inside a block, paging expands the load start down to the
        # previous block boundary and loads that whole block.
        num_external_tokens = max(0, int(remote_token_len) - int(num_computed_tokens))
        if num_external_tokens == 0:
            if self._kv_transfer_config.is_kv_producer:
                self._next_save_block[req_id] = int(remote_token_len) // block_size
            return 0, False
        # Retain the boundary and hashes together until paging binds this lookup
        # result to concrete destination block IDs.
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
            remote_block_count=remote_block,
        )
        self._pending_loads[req_id] = load_request

    def _build_save_requests(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[MooncakeStoreSaveRequest, ...]:
        """Build newly completed full-block suffixes for prefill work."""
        token_lens = scheduler_output.connector_token_lens
        if not self._kv_transfer_config.is_kv_producer or not token_lens:
            return ()

        running = scheduler_output.running
        block_ids = scheduler_output.connector_block_ids
        logical_block_ids = scheduler_output.connector_logical_block_ids
        if not (len(running) == len(token_lens) == len(block_ids) == len(logical_block_ids)):
            raise ValueError('connector save fields must contain one value per running request')

        block_size = self._cache_config.block_size
        save_requests = []
        for request, token_len, request_blocks, request_logical_blocks in zip(
                running, token_lens, block_ids, logical_block_ids, strict=True):
            if (not request.history_multimodals.empty()
                    or len(request.history_embeddings) > 0):
                continue

            request_id = int(request.seq_id)
            full_blocks = int(token_len) // block_size
            first_block = self._next_save_block.get(request_id, 0)
            if full_blocks <= first_block:
                continue
            if full_blocks > len(request_blocks) or full_blocks > len(request_logical_blocks):
                raise RuntimeError(
                    f'request {request_id} has fewer connector blocks than its '
                    f'{full_blocks * block_size}-token save boundary')

            block_hashes = self._get_request_block_hashes(
                request,
                request_id,
                full_blocks * block_size,
                block_size,
            )
            save_id = self._next_save_id
            self._next_save_id += 1
            save_requests.append(
                MooncakeStoreSaveRequest(
                    save_id=save_id,
                    request_id=request_id,
                    start_block=first_block,
                    block_ids=tuple(request_blocks[first_block:full_blocks]),
                    logical_block_ids=tuple(request_logical_blocks[first_block:full_blocks]),
                    block_hashes=block_hashes[first_block:full_blocks],
                ))
            self._next_save_block[request_id] = full_blocks
            self._inflight_save_ids.add(save_id)
        return tuple(save_requests)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> MooncakeStoreConnectorMetadata | None:
        """Dispatch new work and keep emitting polling steps while I/O runs."""
        save_requests = self._build_save_requests(scheduler_output)
        if (not save_requests and not self._pending_loads
                and not self._inflight_loads and not self._inflight_save_ids):
            return None

        load_requests = tuple(self._pending_loads.values())
        self._inflight_loads.update(self._pending_loads)
        self._pending_loads.clear()
        return MooncakeStoreConnectorMetadata(
            load_requests=load_requests,
            save_requests=save_requests,
        )

    def on_new_request(self, request: SchedulerSequence) -> None:
        """Reset request-local lookup and save planning state."""
        req_id = int(request.seq_id)
        self._request_hash_trackers.pop(req_id, None)
        self._lookup_plans.pop(req_id, None)
        self._failed_load_requests.discard(req_id)
        self._next_save_block.pop(req_id, None)
        return None

    def is_lookup_pending(self, request_id: int) -> bool:
        """Return whether the request's lookup Future is still running."""
        client = self.client
        return client is not None and client.is_pending(int(request_id))

    def cancel_lookup(self, request_id: int) -> None:
        """Cancel only the current lookup, retaining incremental hashes."""
        if self.client is not None:
            self.client.discard(int(request_id))

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> KVConnectorResult:
        """Convert all-TP worker completion into backend-neutral updates."""
        completed_save_ids = frozenset(
            save_id
            for save_id in connector_output.completed_save_ids or ()
            if save_id in self._inflight_save_ids
        )
        self._inflight_save_ids.difference_update(completed_save_ids)

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
                local_block_count = (
                    request.remote_block_count - len(request.block_ids)
                )
                if self._kv_transfer_config.is_kv_producer:
                    self._next_save_block[req_id] = local_block_count
            else:
                self._failed_load_requests.discard(req_id)
                if self._kv_transfer_config.is_kv_producer:
                    self._next_save_block[req_id] = request.remote_block_count
            results.append(KVLoadResult(request_id=req_id, success=not failed))
        return KVConnectorResult(
            load_results=tuple(results),
            completed_save_ids=completed_save_ids,
        )

    def request_finished(
        self,
        request: SchedulerSequence,
    ) -> None:
        """Discard request-local lookup and save-boundary state."""
        req_id = int(request.seq_id)
        if self.client is not None:
            self.client.discard(req_id)
        self._request_hash_trackers.pop(req_id, None)
        self._lookup_plans.pop(req_id, None)
        pending = self._pending_loads.pop(req_id, None)
        inflight = self._inflight_loads.pop(req_id, None)
        load_request = pending or inflight
        if load_request is not None:
            self._invalid_block_ids.difference_update(load_request.block_ids)
        self._failed_load_requests.discard(req_id)
        self._next_save_block.pop(req_id, None)
        return None

    def finish_transfers_after_worker_drain(self) -> None:
        """Forget completions whose worker outputs were discarded by sleep."""
        self._pending_loads.clear()
        self._inflight_loads.clear()
        self._invalid_block_ids.clear()
        self._inflight_save_ids.clear()

    def shutdown(self) -> None:
        """Cancel pending lookups and release the scheduler client."""
        if self.client is not None:
            self.client.close()
        self._request_hash_trackers.clear()
        self._lookup_plans.clear()
        self._pending_loads.clear()
        self._inflight_loads.clear()
        self._invalid_block_ids.clear()
        self._failed_load_requests.clear()
        self._next_save_block.clear()
        self._inflight_save_ids.clear()
