# Copyright (c) OpenMMLab. All rights reserved.
"""Scheduler-side metadata and lifecycle for Mooncake Store saves."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .data import (
    MooncakeStoreConnectorMetadata,
    MooncakeStoreSaveRequest,
    build_prefix_block_hashes,
)
from .worker import LookupKeyClient

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
        self._request_trackers: dict[tuple[int, int], _RequestSaveTracker] = {}
        self._request_hash_trackers: dict[int, _RequestHashTracker] = {}
        self._save_waves: dict[int, _SaveWave] = {}
        self._latest_generations: dict[int, int] = {}
        self._finished_requests: set[int] = set()

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
        """Build serializable save work for complete prefill blocks only."""
        preempted_save_ids = tuple(
            int(save_id)
            for save_id in getattr(scheduler_output, 'preempted_save_ids', ())
        )
        running = tuple(getattr(scheduler_output, 'running', ()) or ())
        token_lens = tuple(getattr(scheduler_output, 'connector_token_lens', ()) or ())

        # Empty token lengths identify decode and cleanup-only forwards.  They
        # still carry preemption notifications but must not save one-step-stale
        # decode state.
        if self.kv_role == 'kv_consumer' or len(token_lens) == 0:
            return MooncakeStoreConnectorMetadata(preempted_save_ids=preempted_save_ids)

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

        return MooncakeStoreConnectorMetadata(
            save_requests=tuple(save_requests),
            preempted_save_ids=preempted_save_ids,
        )

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
            self._request_hash_trackers.pop(int(request_id), None)
        return None

    def update_connector_output(self, connector_output: Any) -> None:
        """Retire completed save waves or restore a failed dispatch."""
        rolled_back = self._get_output_ids(connector_output, 'rolled_back_save_ids')
        if rolled_back:
            for save_id in rolled_back:
                self._rollback_save_wave(save_id)
            return None

        for save_id in self._completed_save_ids(connector_output):
            wave = self._save_waves.pop(save_id, None)
            if wave is None:
                continue
            tracker = self._request_trackers.get(wave.tracker_key)
            if tracker is not None:
                tracker.active_save_ids.discard(save_id)
                self._discard_finished_tracker(wave.tracker_key, tracker)
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
            self._request_hash_trackers.pop(request_id, None)
            self._finished_requests.add(request_id)
            for tracker_key, tracker in tuple(self._request_trackers.items()):
                if tracker_key[0] == request_id:
                    self._discard_finished_tracker(tracker_key, tracker)
            if not any(key[0] == request_id for key in self._request_trackers):
                self._finished_requests.discard(request_id)
                self._latest_generations.pop(request_id, None)
        return False, None

    def has_pending_kv_connector_work(self) -> bool:
        """Return whether scheduler metadata has unacknowledged save waves."""
        return bool(self._save_waves)

    def shutdown(self) -> None:
        """Cancel pending lookups and release the scheduler client."""
        client = self.client
        self.client = None
        if client is not None:
            client.close()
        self._request_trackers.clear()
        self._request_hash_trackers.clear()
        self._save_waves.clear()
        self._latest_generations.clear()
        self._finished_requests.clear()
        return None
