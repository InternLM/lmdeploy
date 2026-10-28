# Copyright (c) OpenMMLab. All rights reserved.
"""Scheduler-side lookup, load, and save lifecycle for Mooncake Store."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lmdeploy.pytorch.kv_connector.base import (
    KVConnectorOutput,
    KVConnectorResult,
    KVConnectorStepInput,
    KVLoadResult,
    RequestId,
)
from lmdeploy.pytorch.long_context import get_last_prefill_save_step
from lmdeploy.pytorch.paging.state_manager import StateAllocator
from lmdeploy.utils import get_logger

from .data import (
    MooncakeStoreConnectorMetadata,
    MooncakeStoreLoadRequest,
    MooncakeStoreSaveRequest,
    MooncakeStoreStateSave,
    build_prefix_block_hashes,
)
from .lookup import LookupKeyClient

logger = get_logger('lmdeploy')

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig
    from lmdeploy.pytorch.messages import SchedulerSequence


@dataclass
class _RequestHashTracker:
    """Stable hashes already computed for an append-only request prefix."""

    adapter_identity: str
    block_hashes: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class _LookupPlan:
    """Positive lookup snapshot waiting for paging block allocation.

    The remote boundary has already excluded any recompute overlap. It and
    its block hashes must come from the same lookup.
    ``update_state_after_alloc`` consumes this snapshot exactly once after the
    paging scheduler has assigned destination GPU blocks.
    """

    remote_token_len: int
    block_hashes: tuple[bytes, ...]


@dataclass
class _SaveOperation:
    """A dispatched suffix and the input turn its success may advance."""

    save: MooncakeStoreSaveRequest
    decode_request: SchedulerSequence | None
    input_end_pos: int
    failed: bool = False


class MooncakeStoreScheduler:
    """Scheduler-side component of the Mooncake Store connector."""

    def __init__(self, cache_config: CacheConfig) -> None:
        kv_transfer_config = cache_config.kv_transfer_config
        if kv_transfer_config is None or not kv_transfer_config.is_kv_transfer_instance:
            raise ValueError('MooncakeStoreScheduler requires an enabled kv_transfer_config')
        if cache_config.window_size > 1:
            raise ValueError('Mooncake Store does not support sliding-window KV caches')

        if kv_transfer_config.is_kv_consumer:
            lookup_async = kv_transfer_config.kv_connector_extra_config.get('lookup_async', True)
            if lookup_async is not True:
                raise ValueError('Mooncake Store requires lookup_async=true')

        self._cache_config = cache_config
        self._kv_transfer_config = kv_transfer_config
        self._is_hybrid = bool(cache_config.states_shapes)
        if self._is_hybrid and cache_config.prefill_save_alignment % cache_config.block_size != 0:
            raise ValueError('prefill_save_alignment must be a multiple of block_size')
        num_snapshots = cache_config.num_store_state_caches
        self._state_slots: StateAllocator | None = None
        if num_snapshots:
            num_states = cache_config.num_state_caches
            if num_states is None or num_states <= num_snapshots:
                raise ValueError('Mooncake snapshot slots must be included in num_state_caches')
            self._state_slots = StateAllocator(num_snapshots, offset=num_states - num_snapshots)
        self._num_skipped_state_saves = 0
        self.client = LookupKeyClient(cache_config) if kv_transfer_config.is_kv_consumer else None
        self._request_hash_trackers: dict[int, _RequestHashTracker] = {}
        # Positive lookup snapshots keyed by request ID. A snapshot can survive
        # several schedule attempts while paging waits for allocation capacity.
        self._lookup_plans: dict[RequestId, _LookupPlan] = {}
        # A request cannot progress while its single active load is pending,
        # so the request ID also uniquely identifies the load operation.
        self._pending_loads: dict[RequestId, MooncakeStoreLoadRequest] = {}
        self._inflight_loads: dict[RequestId, MooncakeStoreLoadRequest] = {}
        # Failures may precede all-worker completion. Retain them both for
        # terminal rollback and to avoid retrying Store for this sequence.
        # A later conversation turn has a new request ID.
        self._failed_load_requests: set[RequestId] = set()
        # Saves do not block request progress. Chunked prefill may therefore
        # create several in-flight saves for one request, each with its own ID.
        self._next_save_id = 0
        # Prefill advances this frontier on dispatch to avoid resending each
        # chunk. Failure rewinds it; decode also covers unfinished earlier saves.
        self._next_save_block: dict[RequestId, int] = {}
        self._inflight_saves: dict[int, _SaveOperation] = {}
        self._ready_decode_saves: list[MooncakeStoreSaveRequest] = []

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
        if len(request.history_embeddings) > 0:
            return 0, False

        block_size = self._cache_config.block_size
        token_len = request.clamp_prefix_cache_match_step(
            request.get_prefix_cache_max_candidate_step())
        recompute_blocks = max(0, request.prefix_cache.recompute_overlap.recompute_blocks)
        recompute_tokens = recompute_blocks * block_size
        lookup_alignment = self._cache_config.state_lookup_alignment if self._is_hybrid else block_size
        max_hit_tokens = max(0, token_len - recompute_tokens) // lookup_alignment * lookup_alignment
        if max_hit_tokens == 0 or num_computed_tokens >= max_hit_tokens:
            return 0, False
        # Check the rewind margin beyond the latest possible state boundary.
        # A matching FA block there can make that state reusable with MTP.
        token_len = max_hit_tokens + recompute_tokens

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
            recompute_blocks=recompute_blocks,
        )
        # ``None`` means the asynchronous RPC is still running; zero is a
        # completed lookup with no remotely reusable suffix.
        if remote_token_len is None:
            return None, False

        if self._is_hybrid:
            # Save excludes boundaries inside multimodal spans, whose full
            # extents also participate in the hash. A matching state must be
            # safe as-is; shortening it would select an unverified checkpoint.
            if (remote_token_len % lookup_alignment != 0
                    or request.clamp_prefix_cache_match_step(remote_token_len) != remote_token_len):
                logger.error('Mooncake hybrid lookup returned an invalid state boundary: request_id=%s boundary=%d',
                             req_id, remote_token_len)
                return 0, False
        else:
            # FA can reuse a shorter prefix if the hit ends inside a media span.
            remote_token_len = request.clamp_prefix_cache_match_step(remote_token_len)

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

        block_extras = {}
        lookahead = self._cache_config.prefix_cache_token_lookahead if self._is_hybrid else 0
        if not request.history_multimodals.empty() or lookahead:
            for block_index in range(len(tracker.block_hashes), num_blocks):
                start = block_index * block_size
                end = start + block_size
                spans = request.get_prefix_cache_multimodal_identity(start, end + lookahead)
                identity = [(span.modality, span.content_hash, span.start - start, span.end - start)
                            for span in spans]
                if lookahead:
                    # A shifted MTP row also depends on the next accepted token.
                    # Include it for both prompt and generated Store prefixes.
                    identity.append(('mtp_lookahead', tuple(int(x) for x in request.all_ids[end:end + lookahead])))
                if identity:
                    block_extras[block_index] = json.dumps(identity, separators=(',', ':')).encode('utf-8')

        tracker.block_hashes = build_prefix_block_hashes(
            request.all_ids[:token_len],
            block_size,
            extra_identity=adapter_identity,
            previous_hashes=tracker.block_hashes,
            block_extra_identities=block_extras,
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
        plan = self._lookup_plans[req_id]
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

        state_slot = None
        if self._is_hybrid:
            state_slot = int(request.logical_state)
            runtime_pool_end = self._cache_config.num_state_caches - self._cache_config.num_store_state_caches
            if not 0 < state_slot < runtime_pool_end:
                raise ValueError('Mooncake hybrid load requires an allocated runtime state slot')

        load_request = MooncakeStoreLoadRequest(
            request_id=req_id,
            block_ids=block_ids,
            block_hashes=block_hashes,
            remote_block_count=remote_block,
            state_slot=state_slot,
        )
        # Commit only after every destination has been validated. A failed
        # binding leaves the lookup available for a later allocation attempt.
        self._lookup_plans.pop(req_id)
        self._pending_loads[req_id] = load_request

    def _build_save_requests(
        self,
        step_input: KVConnectorStepInput,
    ) -> tuple[MooncakeStoreSaveRequest, ...]:
        """Build immutable suffixes and track unconfirmed decode snapshots."""
        token_lens = step_input.connector_token_lens
        if not self._kv_transfer_config.is_kv_producer or not token_lens:
            return ()

        running = step_input.running
        block_ids = step_input.connector_block_ids
        logical_block_ids = step_input.connector_logical_block_ids
        if not (len(running) == len(token_lens) == len(block_ids) == len(logical_block_ids)):
            raise ValueError('connector save fields must contain one value per running request')
        if self._is_hybrid and len(step_input.connector_state_ids) != len(running):
            raise ValueError('hybrid saves require one runtime state slot per running request')

        block_size = self._cache_config.block_size
        pending_starts = {}
        if self._is_hybrid:
            for operation in self._inflight_saves.values():
                save = operation.save
                pending_starts[save.request_id] = min(
                    pending_starts.get(save.request_id, save.start_block), save.start_block)
        save_requests = []
        for index, (request, token_len, request_blocks, request_logical_blocks) in enumerate(zip(
                running, token_lens, block_ids, logical_block_ids, strict=True)):
            if len(request.history_embeddings) > 0:
                continue

            request_id = int(request.seq_id)
            full_blocks = int(token_len) // block_size
            first_block = self._next_save_block.get(request_id, 0)
            is_decode_save = step_input.is_decode_checkpoint or (
                self._is_hybrid and self._cache_config.prefix_cache_decode_save_first
                and token_len == request.input_end_pos)
            if is_decode_save:
                if request.prefix_cache.decode_store_save_pending:
                    continue
                # A later state is reusable only with a complete FA prefix.
                # Cover pending earlier waves so their failure cannot leave
                # a hole behind a successfully saved decode checkpoint.
                first_block = min(first_block, pending_starts.get(request_id, first_block))
            if full_blocks <= first_block:
                continue
            if self._is_hybrid:
                if not step_input.is_decode_checkpoint:
                    if self._cache_config.prefill_save_last_only and token_len != get_last_prefill_save_step(request):
                        continue
                alignment = (self._cache_config.state_lookup_alignment if step_input.is_decode_checkpoint
                             else self._cache_config.prefill_save_alignment)
                if (token_len % alignment != 0
                        or not request.is_prefix_cache_boundary_safe(token_len)):
                    continue
                lookahead = self._cache_config.prefix_cache_token_lookahead
                if lookahead and token_len + lookahead > request.num_valid_ids:
                    continue
                if step_input.connector_state_ids[index] <= 0:
                    raise ValueError('hybrid saves require an allocated runtime state slot')
                if not step_input.connector_snapshot_ids and self._state_slots.get_num_free() == 0:
                    self._num_skipped_state_saves += 1
                    logger.info('Mooncake state save skipped: request_id=%s boundary=%d slots=%d skipped=%d',
                                 request_id, token_len, self._state_slots.num_states, self._num_skipped_state_saves)
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
            state = None
            if self._is_hybrid:
                snapshot_slot = (step_input.connector_snapshot_ids[index] if step_input.connector_snapshot_ids
                                 else int(self._state_slots.allocate()))
                state = MooncakeStoreStateSave(source_slot=step_input.connector_state_ids[index],
                                              snapshot_slot=snapshot_slot,
                                              boundary_tokens=int(token_len))
            save = MooncakeStoreSaveRequest(
                save_id=save_id,
                request_id=request_id,
                start_block=first_block,
                block_ids=tuple(request_blocks[first_block:full_blocks]),
                logical_block_ids=tuple(request_logical_blocks[first_block:full_blocks]),
                block_hashes=block_hashes[first_block:full_blocks],
                state=state,
            )
            save_requests.append(save)
            self._next_save_block[request_id] = full_blocks
            self._inflight_saves[save_id] = _SaveOperation(
                save, request if is_decode_save else None, request.input_end_pos if is_decode_save else 0)
            if is_decode_save:
                request.prefix_cache.decode_store_save_pending = True
        return tuple(save_requests)

    def reserve_decode_snapshot(self, request: SchedulerSequence) -> int:
        """Reserve a private Store slot before the worker verifies a
        candidate."""
        if (not self._kv_transfer_config.is_kv_producer or not self._is_hybrid
                or request.prefix_cache.decode_store_save_pending or self._state_slots.get_num_free() == 0):
            return -1
        return int(self._state_slots.allocate())

    def release_decode_snapshot(self, slot: int) -> None:
        self._state_slots.free(slot)

    def queue_decode_save(self, step_input: KVConnectorStepInput) -> MooncakeStoreConnectorMetadata | None:
        """Adopt already frozen states; token hashes are now based on accepted
        output."""
        saves = self._build_save_requests(step_input)
        if not saves:
            return None
        self._ready_decode_saves.extend(saves)
        return MooncakeStoreConnectorMetadata(save_requests=saves)

    def build_connector_meta(
        self,
        step_input: KVConnectorStepInput,
    ) -> MooncakeStoreConnectorMetadata | None:
        """Dispatch new work and keep emitting polling steps while I/O runs."""
        save_requests = tuple(self._ready_decode_saves) + self._build_save_requests(step_input)
        self._ready_decode_saves.clear()
        if (not save_requests and not self._pending_loads
                and not self._inflight_loads and not self._inflight_saves):
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
        for save_id in connector_output.failed_save_ids:
            operation = self._inflight_saves.get(save_id)
            if operation is not None:
                operation.failed = True
        completed_save_ids = frozenset(
            save_id
            for save_id in connector_output.completed_save_ids or ()
            if save_id in self._inflight_saves
        )
        self._finish_saves(completed_save_ids, confirmed=True)

        self._failed_load_requests.update(
            request_id for request_id in connector_output.failed_receiving
            if request_id in self._inflight_loads
        )
        completed = connector_output.finished_receiving or set()
        results = []
        for req_id in sorted(completed):
            request = self._inflight_loads.pop(req_id, None)
            if request is None:
                continue
            failed = req_id in self._failed_load_requests
            if failed:
                if self._kv_transfer_config.is_kv_producer:
                    # Hybrid runtime state may be partially overwritten. Paging
                    # releases the whole request and retries local matching.
                    self._next_save_block[req_id] = (
                        0 if self._is_hybrid else request.remote_block_count - len(request.block_ids)
                    )
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
        self._pending_loads.pop(req_id, None)
        self._inflight_loads.pop(req_id, None)
        self._failed_load_requests.discard(req_id)
        self._next_save_block.pop(req_id, None)
        return None

    def finish_transfers_after_worker_drain(self) -> None:
        """Forget completions whose worker outputs were discarded by sleep."""
        self._pending_loads.clear()
        self._inflight_loads.clear()
        self._ready_decode_saves.clear()
        self._finish_saves(tuple(self._inflight_saves), confirmed=False)

    def _finish_saves(self, save_ids: Iterable[int], *, confirmed: bool) -> None:
        """Release terminal saves; only confirmed success advances a turn."""
        for save_id in save_ids:
            operation = self._inflight_saves.pop(save_id)
            save = operation.save
            if save.state is not None:
                self._state_slots.free(save.state.snapshot_slot)
            request = operation.decode_request
            current_turn = request is not None and request.input_end_pos == operation.input_end_pos
            if current_turn:
                request.prefix_cache.decode_store_save_pending = False
            # Ended requests still own I/O slots, but cannot recreate planning
            # state or advance a later request when their completions arrive.
            if save.request_id not in self._next_save_block:
                continue
            success = confirmed and not operation.failed
            if not success:
                self._next_save_block[save.request_id] = min(
                    self._next_save_block[save.request_id], save.start_block)
            elif current_turn:
                request.prefix_cache.last_decode_save_step = max(
                    request.prefix_cache.last_decode_save_step, save.state.boundary_tokens)

    def shutdown(self) -> None:
        """Cancel pending lookups and release the scheduler client."""
        if self.client is not None:
            self.client.close()
        self._request_hash_trackers.clear()
        self._lookup_plans.clear()
        self._pending_loads.clear()
        self._inflight_loads.clear()
        self._failed_load_requests.clear()
        self._next_save_block.clear()
        self._ready_decode_saves.clear()
        self._finish_saves(tuple(self._inflight_saves), confirmed=False)
