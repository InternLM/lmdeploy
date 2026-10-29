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
    MooncakeStoreCompactStateSave,
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
    from lmdeploy.pytorch.paging.compact_state import SSMStateManager, StateAccess


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
        self._compact_capacity = cache_config.compact_state_capacity
        self._compact_owner: SSMStateManager | None = None
        self._compact_loads: dict[RequestId, StateAccess] = {}
        if num_snapshots and self._compact_capacity is None:
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
        # The next block eligible for save in each request. Scheduling advances
        # it optimistically; save failures are retried only by a later request.
        self._next_save_block: dict[RequestId, int] = {}
        self._inflight_saves: dict[int, int | StateAccess | None] = {}

    def bind_compact_state_manager(self, owner: SSMStateManager) -> None:
        """Inject paging's single unit owner before any compact Store
        admission.

        Connector construction precedes paging construction. This explicit bind avoids allocating a second owner or the
        legacy snapshot tail. Worker records still carry only integer IDs, never these scheduler handles.
        """
        if self._compact_capacity is None or owner.capacity != self._compact_capacity:
            raise ValueError('Compact Store binding must preserve the configured capacity.')
        if (self._compact_owner is not None or self._lookup_plans or self._pending_loads
                or self._inflight_loads or self._inflight_saves):
            raise ValueError('Compact Store owner must be bound exactly once before admission.')
        self._compact_owner = owner

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
        if self._compact_capacity is not None and self._compact_owner is None:
            raise RuntimeError('Compact Store lookup requires the paging state owner.')
        if len(request.history_embeddings) > 0:
            return 0, False

        block_size = self._cache_config.block_size
        token_len = request.clamp_prefix_cache_match_step(
            request.get_prefix_cache_max_candidate_step())
        recompute_blocks = max(0, request.prefix_cache.recompute_overlap.recompute_blocks)
        recompute_tokens = recompute_blocks * block_size
        lookup_alignment = self._cache_config.prefill_save_alignment if self._is_hybrid else block_size
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
        if not request.history_multimodals.empty():
            # Store keys omit MTP lookahead, including media beyond this block.
            for block_index in range(len(tracker.block_hashes), num_blocks):
                start = block_index * block_size
                spans = request.get_prefix_cache_multimodal_identity(start, start + block_size)
                if spans:
                    identity = [(span.modality, span.content_hash, span.start - start, span.end - start)
                                for span in spans]
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

        state_slot = state_unit_id = None
        load_access = None
        if self._compact_capacity is not None:
            if self._compact_owner is None:
                raise RuntimeError('Compact Store requires the paging state owner before load admission.')
            bank = self._compact_owner.bank_for(request)
            # Remaining record validation precedes the owner lease below.
            state_unit_id = bank.unit_ids[0]
        elif self._is_hybrid:
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
            state_unit_id=state_unit_id,
        )
        if state_unit_id is not None:
            load_access = self._compact_owner.submit_load(bank, remote_token_len)
        # Commit only after every destination has been validated. A failed
        # binding leaves the lookup available for a later allocation attempt.
        self._lookup_plans.pop(req_id)
        self._pending_loads[req_id] = load_request
        if load_access is not None:
            self._compact_loads[req_id] = load_access

    def _build_save_requests(
        self,
        step_input: KVConnectorStepInput,
    ) -> tuple[MooncakeStoreSaveRequest, ...]:
        """Build newly completed full-block suffixes for prefill work."""
        token_lens = step_input.connector_token_lens
        if not self._kv_transfer_config.is_kv_producer or not token_lens:
            return ()

        if self._compact_capacity is not None:
            return self._build_compact_save_requests(step_input)

        running = step_input.running
        block_ids = step_input.connector_block_ids
        logical_block_ids = step_input.connector_logical_block_ids
        if not (len(running) == len(token_lens) == len(block_ids) == len(logical_block_ids)):
            raise ValueError('connector save fields must contain one value per running request')
        if self._is_hybrid and len(step_input.connector_state_ids) != len(running):
            raise ValueError('hybrid saves require one runtime state slot per running request')

        block_size = self._cache_config.block_size
        save_requests = []
        for index, (request, token_len, request_blocks, request_logical_blocks) in enumerate(zip(
                running, token_lens, block_ids, logical_block_ids, strict=True)):
            if len(request.history_embeddings) > 0:
                continue

            request_id = int(request.seq_id)
            full_blocks = int(token_len) // block_size
            first_block = self._next_save_block.get(request_id, 0)
            if full_blocks <= first_block:
                continue
            if self._is_hybrid:
                if self._cache_config.prefill_save_last_only and token_len != get_last_prefill_save_step(request):
                    continue
                if (token_len % self._cache_config.prefill_save_alignment != 0
                        or not request.is_prefix_cache_boundary_safe(token_len)):
                    continue
                if step_input.connector_state_ids[index] <= 0:
                    raise ValueError('hybrid saves require an allocated runtime state slot')
                if self._state_slots.get_num_free() == 0:
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
                state = MooncakeStoreStateSave(source_slot=step_input.connector_state_ids[index],
                                              snapshot_slot=int(self._state_slots.allocate()),
                                              boundary_tokens=int(token_len))
            save_requests.append(
                MooncakeStoreSaveRequest(
                    save_id=save_id,
                    request_id=request_id,
                    start_block=first_block,
                    block_ids=tuple(request_blocks[first_block:full_blocks]),
                    logical_block_ids=tuple(request_logical_blocks[first_block:full_blocks]),
                    block_hashes=block_hashes[first_block:full_blocks],
                    state=state,
                ))
            self._next_save_block[request_id] = full_blocks
            self._inflight_saves[save_id] = state.snapshot_slot if state is not None else None
        return tuple(save_requests)

    def plan_compact_save_steps(self, running, token_lens, checkpoint_steps, *, skipped=None):
        """Select Store intent from the SAME c/e plan before reserving
        producers.

        No resource acquisition or high-water advancement occurs here. The CPU binder merges these intents with local
        Trie reservations, submits one producer per row, then hands its physical references back below.
        """
        if not self._kv_transfer_config.is_kv_producer or not token_lens:
            return (0,) * len(running)
        owner = self._compact_owner
        if owner is None:
            raise RuntimeError('Compact Store saves require the paging state owner.')
        if not (len(running) == len(token_lens) == len(checkpoint_steps)):
            raise ValueError('Compact Store save plans must match model rows.')
        credits = owner.get_num_free_store_saves()
        steps = []
        block_size = self._cache_config.block_size
        for seq, end, boundary in zip(running, token_lens, checkpoint_steps):
            pending = owner.bank_for(seq).pending_checkpoint
            eligible = ((pending is None or pending.phase == 'reserved')
                        and 0 < boundary <= end and not seq.history_embeddings
                        and (not self._cache_config.prefill_save_last_only
                             or boundary == get_last_prefill_save_step(seq))
                        and boundary % self._cache_config.prefill_save_alignment == 0
                        and seq.is_prefix_cache_boundary_safe(boundary)
                        and boundary // block_size > self._next_save_block.get(int(seq.seq_id), 0))
            if eligible and credits == 0:
                if skipped is not None:
                    skipped.append((int(seq.seq_id), boundary, 'credit'))
                eligible = False
            steps.append(boundary if eligible else 0)
            credits -= int(eligible)
        return tuple(steps)

    def record_compact_save_skips(self, skipped):
        """Count only committed preparation, never rolled-back CPU attempts."""
        for request_id, boundary, reason in skipped:
            self._num_skipped_state_saves += 1
            logger.info('Mooncake state save skipped: request_id=%s boundary=%d reason=%s skipped=%d',
                        request_id, boundary, reason, self._num_skipped_state_saves)

    def _build_compact_save_requests(self, step_input):
        """Atomically retain direct producer outputs and build c-keyed
        saves."""
        owner = self._compact_owner
        if owner is None:
            raise RuntimeError('Compact Store saves require the paging state owner.')
        running, ends, references = (step_input.running, step_input.connector_token_lens,
                                      step_input.compact_state_references)
        blocks, logical = step_input.connector_block_ids, step_input.connector_logical_block_ids
        if not (len(running) == len(ends) == len(references) == len(blocks) == len(logical)):
            raise ValueError('Compact saves require one producer reference per model row.')
        if len({int(seq.seq_id) for seq in running}) != len(running):
            raise ValueError('Compact Store saves require distinct request rows.')
        block_size = self._cache_config.block_size
        requests, sources = [], []
        for seq, end, reference, ids, logical_ids in zip(running, ends, references, blocks, logical):
            if reference is None:
                continue
            boundary = reference.boundary
            if (not reference.active or reference.phase not in ('submitted', 'ready')
                    or not 0 < boundary <= end or boundary % self._cache_config.prefill_save_alignment
                    or seq.history_embeddings or not seq.is_prefix_cache_boundary_safe(boundary)):
                raise ValueError('Compact Store save requires an exact, safe, ordered-readable producer.')
            request_id, full_blocks = int(seq.seq_id), int(end) // block_size
            first_block = self._next_save_block.get(request_id, 0)
            if full_blocks <= first_block:
                continue
            if full_blocks > len(ids) or full_blocks > len(logical_ids):
                raise ValueError('Compact Store save KV extent exceeds allocated blocks.')
            hashes = self._get_request_block_hashes(seq, request_id, full_blocks * block_size, block_size)
            state = MooncakeStoreCompactStateSave(reference.unit_id, boundary, int(end),
                                                  hashes[boundary // block_size - 1])
            requests.append(MooncakeStoreSaveRequest(
                self._next_save_id + len(requests), request_id, first_block,
                tuple(ids[first_block:full_blocks]), tuple(logical_ids[first_block:full_blocks]),
                hashes[first_block:full_blocks], state))
            sources.append(reference)
        leases = []
        try:
            for reference in sources:
                retained = owner.retain(reference, 'store')
                try:
                    producer = retained.pending_producer
                    waits = (producer,) if producer is not None else ()
                    # Actual worker ready-event wait is reused, not a new CUDA
                    # stream or RPC. The host dependency protects completion order.
                    leases.append(owner.submit_save(retained, stream=-1, wait_for=waits))
                except Exception:
                    owner.release_reference(retained)
                    raise
        except Exception:
            for lease in leases:
                owner.cancel_unsubmitted_save(lease)
            raise
        for request, lease in zip(requests, leases):
            self._inflight_saves[request.save_id] = lease
            self._next_save_block[request.request_id] = request.start_block + len(request.block_ids)
        self._next_save_id += len(requests)
        return tuple(requests)

    def rollback_compact_metadata(self, metadata):
        """Undo ONLY a failed local CPU packet, before any worker can see
        it."""
        if self._compact_owner is None:
            raise RuntimeError('Compact rollback requires the paging state owner.')
        for request in reversed(metadata.save_requests):
            lease = self._inflight_saves.pop(request.save_id)
            self._compact_owner.cancel_unsubmitted_save(lease)
            self._next_save_block[request.request_id] = request.start_block
        for request in metadata.load_requests:
            self._pending_loads[request.request_id] = self._inflight_loads.pop(request.request_id)

    def build_connector_meta(
        self,
        step_input: KVConnectorStepInput,
    ) -> MooncakeStoreConnectorMetadata | None:
        """Dispatch new work and keep emitting polling steps while I/O runs."""
        save_requests = self._build_save_requests(step_input)
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
        completed_save_ids = frozenset(
            save_id
            for save_id in connector_output.completed_save_ids or ()
            if save_id in self._inflight_saves
        )
        self._release_save_slots(completed_save_ids)

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
            access = self._compact_loads.pop(req_id, None)
            if access is not None:
                self._compact_owner.complete(access, success=not failed)
                failed = failed or access.failed
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
        if req_id in self._compact_loads:
            # Pending metadata has never been dispatched. An in-flight direct
            # destination, however, survives even a premature terminal callback.
            if req_id not in self._inflight_loads:
                self._compact_owner.complete(self._compact_loads.pop(req_id), success=False)
            else:
                self._failed_load_requests.add(req_id)
        else:
            self._inflight_loads.pop(req_id, None)
            self._failed_load_requests.discard(req_id)
        self._next_save_block.pop(req_id, None)
        return None

    def finish_transfers_after_worker_drain(self) -> None:
        """Forget completions whose worker outputs were discarded by sleep."""
        for access in self._compact_loads.values():
            self._compact_owner.complete(access, success=False)
        self._compact_loads.clear()
        self._pending_loads.clear()
        self._inflight_loads.clear()
        self._release_save_slots(tuple(self._inflight_saves))

    def _release_save_slots(self, save_ids: Iterable[int]) -> None:
        """Release legacy snapshots or direct leases after all-rank
        completion/drain."""
        for save_id in save_ids:
            lease = self._inflight_saves[save_id]
            if self._compact_owner is not None and lease is not None:
                # A failure notification or an early rank result cannot arrive
                # here as terminal proof. Preserve the record if validation fails.
                self._compact_owner.complete(lease)
                self._compact_owner.release_reference(lease._store_reference)
            elif lease is not None:
                self._state_slots.free(lease)
            del self._inflight_saves[save_id]

    def shutdown(self) -> None:
        """Cancel pending lookups and release the scheduler client."""
        if self._compact_loads or (self._compact_owner is not None and self._inflight_saves):
            raise RuntimeError('Compact Store shutdown requires actual I/O completion or worker drain.')
        if self.client is not None:
            self.client.close()
        self._request_hash_trackers.clear()
        self._lookup_plans.clear()
        self._pending_loads.clear()
        self._inflight_loads.clear()
        self._failed_load_requests.clear()
        self._next_save_block.clear()
        self._release_save_slots(tuple(self._inflight_saves))
