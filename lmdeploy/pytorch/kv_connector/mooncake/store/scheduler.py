# Copyright (c) OpenMMLab. All rights reserved.
"""Scheduler-side metadata and lifecycle for Mooncake Store transfers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lmdeploy.utils import get_logger

from .data import (
    MooncakeStoreConnectorMetadata,
    MooncakeStoreLoadRequest,
    MooncakeStoreSaveRequest,
    build_prefix_block_hashes,
)
from .worker import LookupKeyClient

logger = get_logger('lmdeploy')

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig
    from lmdeploy.pytorch.messages import SchedulerSequence
    from lmdeploy.pytorch.paging.scheduler import SchedulerOutput


@dataclass
class _RequestSaveTracker:
    """Save progress for one request generation."""

    last_token_len: int = 0
    active_save_ids: set[int] = field(default_factory=set)


@dataclass
class _RequestHashTracker:
    """Stable hashes already computed for an append-only request prefix."""

    extra_identity: str
    block_hashes: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class _SaveWave:
    """Bookkeeping needed to finish or roll back one metadata wave."""

    tracker_key: tuple[int, int]
    token_len: int
    previous_token_len: int


@dataclass(frozen=True)
class _LookupSignature:
    """Identity of one immutable, block-aligned lookup candidate."""

    token_len: int
    last_block_hash: bytes
    extra_identity: str


@dataclass(frozen=True)
class _LookupPlan:
    """Completed lookup waiting for paging to allocate its GPU suffix."""

    signature: _LookupSignature
    local_token_len: int
    remote_token_len: int
    block_hashes: tuple[bytes, ...]


@dataclass
class _LoadWave:
    """One load kept until an all-TP terminal completion is observed."""

    request: MooncakeStoreLoadRequest
    signature: _LookupSignature
    dispatching: bool = False
    dispatched: bool = False
    cancelled: bool = False
    needs_fence: bool = False


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
        self._next_save_id = 0
        self._next_load_id = 0
        self._request_trackers: dict[tuple[int, int], _RequestSaveTracker] = {}
        self._request_hash_trackers: dict[int, _RequestHashTracker] = {}
        self._save_waves: dict[int, _SaveWave] = {}
        self._lookup_plans: dict[int, _LookupPlan] = {}
        self._load_waves: dict[int, _LoadWave] = {}
        self._request_load_ids: dict[int, set[int]] = {}
        self._failed_lookup_signatures: dict[int, set[_LookupSignature]] = {}
        self._latest_generations: dict[int, int] = {}
        self._finished_requests: set[int] = set()

    def get_num_new_matched_tokens(
        self,
        request: SchedulerSequence,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Poll an asynchronous lookup for the prefix beyond local GPU KV.

        The lookup candidate always leaves at least one token for a forward so
        a complete prompt hit cannot bypass sampling.  ``None`` means the ZMQ
        lookup is still running; paging must roll back its tentative local L1
        match and retry admission later.
        """
        if self.kv_role == 'kv_producer':
            return 0, False

        req_id = int(request.seq_id)
        block_size = self._cache_config.block_size
        max_match = getattr(request, 'get_prefix_cache_max_match_step', None)
        if max_match is None:
            token_len = max(0, (len(request.all_ids) - 1) // block_size * block_size)
        else:
            token_len = int(max_match())
        token_len = token_len // block_size * block_size
        local_token_len = int(num_computed_tokens) // block_size * block_size
        if token_len < block_size or local_token_len >= token_len:
            self._lookup_plans.pop(req_id, None)
            return 0, False

        block_hashes = self._get_request_block_hashes(
            request,
            req_id,
            token_len,
            block_size,
        )
        signature = _LookupSignature(
            token_len=token_len,
            last_block_hash=block_hashes[-1],
            extra_identity=self._request_hash_identity(request),
        )
        if signature in self._failed_lookup_signatures.get(req_id, set()):
            self._lookup_plans.pop(req_id, None)
            return 0, False

        existing = self._lookup_plans.get(req_id)
        if existing is not None and existing.signature == signature:
            remote_token_len = existing.remote_token_len
            if remote_token_len <= local_token_len:
                self._lookup_plans.pop(req_id, None)
                return 0, False
            if existing.local_token_len != local_token_len:
                existing = _LookupPlan(
                    signature=signature,
                    local_token_len=local_token_len,
                    remote_token_len=remote_token_len,
                    block_hashes=block_hashes,
                )
                self._lookup_plans[req_id] = existing
            return remote_token_len - local_token_len, True

        if existing is not None:
            client = self.client
            if client is not None:
                client.discard(req_id)
            self._lookup_plans.pop(req_id, None)

        client = self.client
        if client is None:
            return 0, False
        remote_token_len = client.lookup(
            req_id,
            token_len,
            block_hashes,
            non_block=self.lookup_async,
        )
        if remote_token_len is None:
            return None, False

        remote_token_len = min(int(remote_token_len), token_len)
        clamp_match = getattr(request, 'clamp_prefix_cache_match_step', None)
        if clamp_match is not None:
            # A shorter Store hit can end inside an image/video placeholder
            # even when the full lookup candidate has a safe boundary.  Apply
            # the same request-level boundary clamp as BlockTrie before the
            # local-prefix comparison and before persisting the load plan.
            remote_token_len = int(clamp_match(remote_token_len))
        if remote_token_len % block_size != 0:
            logger.warning(
                'Mooncake lookup returned a non-aligned prefix for request %s: %s; '
                'rounding down to block size %s',
                req_id,
                remote_token_len,
                block_size,
            )
            remote_token_len = remote_token_len // block_size * block_size
        if remote_token_len <= local_token_len:
            return 0, False

        self._lookup_plans[req_id] = _LookupPlan(
            signature=signature,
            local_token_len=local_token_len,
            remote_token_len=remote_token_len,
            block_hashes=block_hashes,
        )
        logger.info(
            'Mooncake lookup completed: req_id=%s local_tokens=%s remote_tokens=%s load_tokens=%s',
            req_id,
            local_token_len,
            remote_token_len,
            remote_token_len - local_token_len,
        )
        return remote_token_len - local_token_len, True

    def update_state_after_alloc(
        self,
        request: SchedulerSequence,
        block_ids: Sequence[int],
        num_external_tokens: int,
        generation: int = 0,
    ) -> MooncakeStoreLoadRequest | None:
        """Create a load request after paging owns the destination suffix."""
        req_id = int(request.seq_id)
        plan = self._lookup_plans.pop(req_id, None)
        if plan is None:
            if num_external_tokens:
                raise RuntimeError(
                    f'request {req_id} allocated {num_external_tokens} external tokens '
                    'without a completed Mooncake lookup')
            return None

        expected_external_tokens = plan.remote_token_len - plan.local_token_len
        if int(num_external_tokens) != expected_external_tokens:
            raise ValueError(
                f'request {req_id} external token mismatch: '
                f'{num_external_tokens} != {expected_external_tokens}')
        block_size = self._cache_config.block_size
        local_blocks = plan.local_token_len // block_size
        remote_blocks = plan.remote_token_len // block_size
        suffix_hashes = plan.block_hashes[local_blocks:remote_blocks]
        suffix_block_ids = tuple(int(block_id) for block_id in block_ids)
        if len(suffix_block_ids) != len(suffix_hashes):
            raise ValueError(
                f'request {req_id} load suffix has {len(suffix_block_ids)} blocks, '
                f'expected {len(suffix_hashes)}')

        load_id = self._next_load_id
        self._next_load_id += 1
        load_request = MooncakeStoreLoadRequest(
            req_id=req_id,
            load_id=load_id,
            generation=int(generation),
            local_token_len=plan.local_token_len,
            remote_token_len=plan.remote_token_len,
            block_ids=suffix_block_ids,
            block_hashes=suffix_hashes,
        )
        self._load_waves[load_id] = _LoadWave(
            request=load_request,
            signature=plan.signature,
        )
        self._request_load_ids.setdefault(req_id, set()).add(load_id)
        logger.info(
            'Mooncake load queued: req_id=%s load_id=%s generation=%s '
            'local_tokens=%s remote_tokens=%s blocks=%s',
            req_id,
            load_id,
            generation,
            plan.local_token_len,
            plan.remote_token_len,
            len(suffix_block_ids),
        )
        return load_request

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> MooncakeStoreConnectorMetadata:
        """Build new saves and ready, not-yet-dispatched loads."""
        preempted_save_ids = tuple(
            int(save_id)
            for save_id in getattr(scheduler_output, 'preempted_save_ids', ())
        )
        load_requests = self._ready_load_requests()
        running = tuple(getattr(scheduler_output, 'running', ()) or ())
        token_lens = tuple(getattr(scheduler_output, 'connector_token_lens', ()) or ())

        # Empty token lengths identify decode and cleanup-only forwards.  They
        # still carry preemption notifications but must not save one-step-stale
        # decode state.
        if self.kv_role == 'kv_consumer' or len(token_lens) == 0:
            metadata = MooncakeStoreConnectorMetadata(
                load_requests=load_requests,
                preempted_save_ids=preempted_save_ids,
            )
            self._mark_load_metadata_dispatching(metadata)
            return metadata

        block_ids = tuple(getattr(scheduler_output, 'connector_block_ids', ()) or ())
        generations = tuple(getattr(scheduler_output, 'connector_generations', ()) or ())
        expected_len = len(running)
        if len(token_lens) != expected_len or len(block_ids) != expected_len or len(generations) != expected_len:
            raise ValueError('connector scheduler fields must contain one value per running request')

        save_requests = []
        block_size = self._cache_config.block_size
        for request, token_len_value, request_block_ids, generation_value in zip(
                running, token_lens, block_ids, generations):
            req_id = int(request.seq_id)
            generation = int(generation_value)
            token_len = int(token_len_value)
            if token_len <= 0:
                continue
            if token_len % block_size != 0:
                raise ValueError('connector token length must be block aligned')

            tracker_key = (req_id, generation)
            latest_generation = self._latest_generations.get(req_id, generation)
            self._latest_generations[req_id] = max(latest_generation, generation)
            for old_tracker_key, old_tracker in tuple(self._request_trackers.items()):
                if old_tracker_key[0] == req_id and old_tracker_key[1] < generation:
                    self._discard_finished_tracker(old_tracker_key, old_tracker)
            tracker = self._request_trackers.setdefault(tracker_key, _RequestSaveTracker())
            if token_len <= tracker.last_token_len:
                continue

            num_blocks = token_len // block_size
            request_block_ids = tuple(int(block_id) for block_id in request_block_ids[:num_blocks])
            if len(request_block_ids) != num_blocks:
                raise RuntimeError(
                    f'request {req_id} has {len(request_block_ids)} connector blocks, expected {num_blocks}')

            block_hashes = self._get_request_block_hashes(
                request,
                req_id,
                token_len,
                block_size,
            )
            if len(block_hashes) != num_blocks:
                raise RuntimeError('stable block hash count does not match connector block count')

            save_id = self._next_save_id
            self._next_save_id += 1
            save_requests.append(
                MooncakeStoreSaveRequest(
                    req_id=req_id,
                    save_id=save_id,
                    generation=generation,
                    token_len=token_len,
                    block_ids=request_block_ids,
                    block_hashes=block_hashes,
                ))
            previous_token_len = tracker.last_token_len
            tracker.last_token_len = token_len
            tracker.active_save_ids.add(save_id)
            self._save_waves[save_id] = _SaveWave(
                tracker_key=tracker_key,
                token_len=token_len,
                previous_token_len=previous_token_len,
            )

        metadata = MooncakeStoreConnectorMetadata(
            save_requests=tuple(save_requests),
            load_requests=load_requests,
            preempted_save_ids=preempted_save_ids,
        )
        self._mark_load_metadata_dispatching(metadata)
        return metadata

    def _ready_load_requests(self) -> tuple[MooncakeStoreLoadRequest, ...]:
        """Return stable ready work until dispatch is explicitly committed."""
        return tuple(
            wave.request
            for _load_id, wave in sorted(self._load_waves.items())
            if (not wave.dispatching and not wave.dispatched
                and (not wave.cancelled or wave.needs_fence))
        )

    def _mark_load_metadata_dispatching(
        self,
        connector_metadata: MooncakeStoreConnectorMetadata,
    ) -> None:
        """Acquire the build-to-RPC lease for each serialized load."""
        for load_request in getattr(connector_metadata, 'load_requests', ()):
            wave = self._load_waves.get(int(load_request.load_id))
            if wave is None or (wave.cancelled and not wave.needs_fence):
                continue
            if wave.dispatched:
                raise RuntimeError(
                    f'load_id {load_request.load_id} was serialized after dispatch')
            wave.dispatching = True

    def mark_connector_meta_dispatched(
        self,
        connector_metadata: MooncakeStoreConnectorMetadata,
    ) -> None:
        """Commit successful delivery of load metadata to the executor."""
        for load_request in getattr(connector_metadata, 'load_requests', ()):
            wave = self._load_waves.get(int(load_request.load_id))
            if wave is None:
                continue
            if wave.request != load_request:
                raise RuntimeError(
                    f'load_id {load_request.load_id} metadata changed before dispatch')
            # A cancellation may race the RPC await.  Successful delivery
            # still makes that tombstone in-flight and it must retain its pin.
            wave.dispatching = False
            wave.dispatched = True
            wave.needs_fence = False

    def _get_request_block_hashes(
        self,
        request: SchedulerSequence,
        req_id: int,
        token_len: int,
        block_size: int,
    ) -> tuple[bytes, ...]:
        """Return cached hashes, extending them only for new full blocks."""
        token_ids = request.all_ids
        if len(token_ids) < token_len:
            raise RuntimeError(
                f'request {req_id} has {len(token_ids)} tokens, '
                f'expected at least {token_len}')

        num_blocks = token_len // block_size
        extra_identity = self._request_hash_identity(request)
        hash_tracker = self._request_hash_trackers.get(req_id)
        if (hash_tracker is None
                or hash_tracker.extra_identity != extra_identity):
            hash_tracker = _RequestHashTracker(
                extra_identity=extra_identity,
            )
            self._request_hash_trackers[req_id] = hash_tracker

        if num_blocks <= len(hash_tracker.block_hashes):
            return hash_tracker.block_hashes[:num_blocks]

        block_hashes = build_prefix_block_hashes(
            token_ids[:token_len],
            block_size,
            extra_identity=hash_tracker.extra_identity,
            previous_hashes=hash_tracker.block_hashes,
        )
        hash_tracker.block_hashes = block_hashes
        return block_hashes

    @staticmethod
    def _request_hash_identity(request: SchedulerSequence) -> str:
        """Return stable non-token identity affecting a request's KV cache."""
        adapter_name = getattr(request, 'adapter_name', None) or ''
        get_extra_identity = getattr(request, 'get_prefix_cache_extra_identity', None)
        if get_extra_identity is None:
            multimodal_identity = ()
        else:
            multimodal_identity = tuple(
                tuple(span)
                for span in get_extra_identity(0, len(request.all_ids))
            )
        return repr((adapter_name, multimodal_identity))

    def on_new_request(self, request: SchedulerSequence) -> None:
        """Drop any stale request-local hash state before first scheduling."""
        request_id = getattr(request, 'seq_id', None)
        if request_id is not None:
            request_id = int(request_id)
            self._request_hash_trackers.pop(request_id, None)
            self._lookup_plans.pop(request_id, None)
            self._failed_lookup_signatures.pop(request_id, None)
        return None

    def cancel_lookup(self, request_id: int) -> None:
        """Discard only lookup state when a request is temporarily stopped."""
        request_id = int(request_id)
        client = self.client
        if client is not None:
            client.discard(request_id)
        self._lookup_plans.pop(request_id, None)

    def update_connector_output(self, connector_output: Any) -> None:
        """Retire terminal waves and apply scheduler-local control events."""
        rolled_back = self._get_output_ids(connector_output, 'rolled_back_save_ids')
        for save_id in rolled_back:
            self._rollback_save_wave(save_id)

        for load_id in self._get_output_ids(connector_output, 'dispatched_load_ids'):
            wave = self._load_waves.get(load_id)
            if wave is not None:
                wave.dispatching = False
                wave.dispatched = True
                wave.needs_fence = False

        for load_id in self._get_output_ids(connector_output, 'rolled_back_load_ids'):
            wave = self._load_waves.get(load_id)
            if wave is None:
                continue
            wave.dispatching = False
            # A distributed RPC error may mean only some TP ranks accepted
            # the load. A cancelled wave must therefore remain eligible for
            # idempotent re-dispatch until one all-rank RPC succeeds.
            if wave.cancelled and not wave.dispatched:
                wave.needs_fence = True

        for load_id in self._get_output_ids(connector_output, 'cancelled_load_ids'):
            wave = self._load_waves.get(load_id)
            if wave is None:
                continue
            wave.cancelled = True
            if (not wave.dispatching and not wave.dispatched
                    and not wave.needs_fence):
                self._retire_load_wave(load_id, failed=False)

        for save_id in self._completed_save_ids(connector_output):
            wave = self._save_waves.pop(save_id, None)
            if wave is None:
                continue
            tracker = self._request_trackers.get(wave.tracker_key)
            if tracker is not None:
                tracker.active_save_ids.discard(save_id)
                self._discard_finished_tracker(wave.tracker_key, tracker)

        failed_load_ids = self._get_output_ids(connector_output, 'failed_load_ids')
        for load_id in self._completed_load_ids(connector_output):
            self._retire_load_wave(load_id, failed=load_id in failed_load_ids)
        return None

    @staticmethod
    def _get_output_ids(connector_output: Any, name: str) -> set[int]:
        if isinstance(connector_output, dict):
            values = connector_output.get(name)
        else:
            values = getattr(connector_output, name, None)
        if values is None:
            return set()
        return {int(value) for value in values}

    @classmethod
    def _completed_save_ids(cls, connector_output: Any) -> set[int]:
        completed = cls._get_output_ids(connector_output, 'completed_save_ids')
        if completed:
            return completed
        completed = cls._get_output_ids(connector_output, 'finished_sending')
        if completed:
            return completed
        if isinstance(connector_output, tuple) and len(connector_output) == 2:
            values = connector_output[0]
        elif isinstance(connector_output, (set, frozenset, list)):
            values = connector_output
        else:
            values = None
        if values is None:
            return set()
        return {int(value) for value in values}

    @classmethod
    def _completed_load_ids(cls, connector_output: Any) -> set[int]:
        completed = cls._get_output_ids(connector_output, 'completed_load_ids')
        if completed:
            return completed
        completed = cls._get_output_ids(connector_output, 'finished_recving')
        if completed:
            return completed
        if isinstance(connector_output, tuple) and len(connector_output) == 2:
            values = connector_output[1]
            if values is not None:
                return {int(value) for value in values}
        return set()

    def _retire_load_wave(self, load_id: int, *, failed: bool) -> None:
        wave = self._load_waves.pop(int(load_id), None)
        if wave is None:
            return
        request = wave.request
        req_id = int(request.req_id)
        request_load_ids = self._request_load_ids.get(req_id)
        if request_load_ids is not None:
            request_load_ids.discard(int(load_id))
            if not request_load_ids:
                self._request_load_ids.pop(req_id, None)
        if failed and not wave.cancelled:
            self._failed_lookup_signatures.setdefault(req_id, set()).add(wave.signature)
        logger.info(
            'Mooncake load retired: req_id=%s load_id=%s generation=%s '
            'failed=%s cancelled=%s',
            req_id,
            load_id,
            request.generation,
            failed,
            wave.cancelled,
        )

    def _rollback_save_wave(self, save_id: int) -> None:
        wave = self._save_waves.pop(save_id, None)
        if wave is None:
            return
        tracker = self._request_trackers.get(wave.tracker_key)
        if tracker is None:
            return
        tracker.active_save_ids.discard(save_id)
        if tracker.last_token_len == wave.token_len:
            tracker.last_token_len = wave.previous_token_len
        self._discard_finished_tracker(wave.tracker_key, tracker)

    def _discard_finished_tracker(
        self,
        tracker_key: tuple[int, int],
        tracker: _RequestSaveTracker,
    ) -> None:
        req_id, generation = tracker_key
        latest_generation = self._latest_generations.get(req_id, generation)
        is_obsolete = generation < latest_generation
        if (req_id in self._finished_requests or is_obsolete) and not tracker.active_save_ids:
            self._request_trackers.pop(tracker_key, None)
        if req_id in self._finished_requests and not any(
                key[0] == req_id for key in self._request_trackers):
            self._finished_requests.discard(req_id)
            self._latest_generations.pop(req_id, None)

    def request_finished(
        self,
        request: SchedulerSequence,
        block_ids: Sequence[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Do not take ownership of finished request blocks yet."""
        request_id = getattr(request, 'seq_id', None)
        if self.client is not None and request_id is not None:
            self.client.discard(request_id)
        if request_id is not None:
            request_id = int(request_id)
            self._lookup_plans.pop(request_id, None)
            for load_id in tuple(self._request_load_ids.get(request_id, ())):
                wave = self._load_waves.get(load_id)
                if wave is None:
                    continue
                wave.cancelled = True
                if (not wave.dispatching and not wave.dispatched
                        and not wave.needs_fence):
                    self._retire_load_wave(load_id, failed=False)
            self._request_hash_trackers.pop(request_id, None)
            self._failed_lookup_signatures.pop(request_id, None)
            self._finished_requests.add(request_id)
            for tracker_key, tracker in tuple(self._request_trackers.items()):
                if tracker_key[0] == request_id:
                    self._discard_finished_tracker(tracker_key, tracker)
            if not any(key[0] == request_id for key in self._request_trackers):
                self._finished_requests.discard(request_id)
                self._latest_generations.pop(request_id, None)
        return False, None

    def has_pending_kv_transfer_work(self) -> bool:
        """Return whether a save or load still awaits terminal completion."""
        return bool(self._save_waves or self._load_waves)

    def has_pending_kv_lookup_work(self) -> bool:
        """Return whether the scheduler-side async lookup client is busy."""
        client = self.client
        return bool(client is not None and getattr(client, 'futures', ()))

    def has_pending_kv_connector_work(self) -> bool:
        """Compatibility union of lookup and transfer work."""
        return self.has_pending_kv_transfer_work() or self.has_pending_kv_lookup_work()

    def shutdown(self) -> None:
        """Cancel pending lookups and release the scheduler client."""
        client = self.client
        self.client = None
        if client is not None:
            client.close()
        self._request_trackers.clear()
        self._request_hash_trackers.clear()
        self._save_waves.clear()
        self._lookup_plans.clear()
        self._load_waves.clear()
        self._request_load_ids.clear()
        self._failed_lookup_signatures.clear()
        self._latest_generations.clear()
        self._finished_requests.clear()
        return None
