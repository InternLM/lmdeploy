# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
"""Request scheduling and prefix-cache side-effect boundaries.

The scheduler is the first owner of prefix-cache side effects.  In prefill,
``BlockTrie.match()`` is intentionally called before eviction and allocation so
the scheduler can account for reused KV/state.  That match is tentative:
rollback is required if checkpoint pinning, KV eviction, or runtime state
allocation means the request cannot safely run now.  Long-context suffixes can
continue chunking from the accepted prefix hit.

Successful prefill scheduling keeps this order:

1. ``block_trie.match(seq)`` mutates sequence state to skip a cached prefix.
2. eviction and SSM runtime-state availability are checked.
3. ``block_manager.allocate(seq)`` allocates missing KV blocks.
4. ``block_trie.allocate(seq)`` publishes newly allocated full blocks.
5. For SSM, downstream input/model/engine code restores and saves checkpoint
   states; the scheduler only owns resource decisions and rollback.

SSM scheduling detail:

* ``block_trie.match(seq)`` may find a published checkpoint and record
  ``seq.prefix_cache.restore`` before the request owns a runtime state.
  The scheduler must treat that as tentative until KV blocks and one runtime
  state slot are guaranteed.
* A matched restore checkpoint can be pinned before eviction so checkpoint LRU
  cannot free the source slot.  If that pin prevents eviction from finding
  enough resources, the scheduler rolls the match back, releases the pin, and
  retries eviction once without the tentative hit.
* Runtime state availability is checked after KV eviction because old unpinned
  checkpoints may be dropped to free state-cache slots.  If no runtime slot can
  be recovered, the tentative prefix hit is rolled back and the request waits.
* ``state_manager.allocate(seq)`` assigns the request runtime state only after
  ``block_manager.allocate(seq)`` and ``block_trie.allocate(seq)`` succeed.
  Later, ``InputsMaker`` may reserve checkpoint saves for the exact produced
  step; scheduler code does not perform state-cache tensor copies or publish
  checkpoint readiness.
"""

import time
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from torch.profiler import record_function

from lmdeploy.messages import EventType, ScheduleMetrics
from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.long_context import get_long_context_chunk_limit, plan_long_context_chunk
from lmdeploy.utils import get_logger

from ..config import CacheConfig, SchedulerConfig
from ..messages import MessageStatus, SchedulerSequence, SchedulerSession, SequenceManager, SequenceMeta
from .block_manager import build_block_manager
from .block_trie import BlockTrie
from .eviction_helper import build_eviction_helper
from .state_manager import build_state_manager

if TYPE_CHECKING:
    from lmdeploy.pytorch.kv_connector.base import KVConnectorBase

logger = get_logger('lmdeploy')

MapType = dict[int, int]
SeqList = list[SchedulerSequence]


@dataclass
class SchedulerOutput:
    """Output of schedule."""

    running: SeqList
    swap_in_map: MapType
    swap_out_map: MapType
    copy_map: MapType
    connector_token_lens: tuple[int, ...] = ()
    connector_block_ids: tuple[tuple[int, ...], ...] = ()
    connector_logical_block_ids: tuple[tuple[int, ...], ...] = ()
    connector_generations: tuple[int, ...] = ()
    preempted_save_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class _PendingKVSave:
    """Scheduler-owned logical-block pins for one async save wave."""

    req_id: int
    generation: int
    logical_block_ids: np.ndarray


@dataclass
class _PendingKVLoad:
    """Paging ownership retained until one async load reaches all-TP
    terminal."""

    seq: SchedulerSequence
    req_id: int
    load_id: int
    generation: int
    local_token_len: int
    remote_token_len: int
    logical_block_ids: np.ndarray
    prefill_target_blocks: int
    fallback_step: int
    fallback_logical_block_ids: np.ndarray
    fallback_restored: bool = False
    dispatching: bool = False
    dispatched: bool = False
    cancelled: bool = False
    needs_fence: bool = False

    @property
    def prefill_reservation_blocks(self) -> int:
        """Unallocated rows still owed to this request's complete prefill."""
        return max(0, self.prefill_target_blocks - int(self.seq.num_blocks))


@dataclass(frozen=True)
class _PrefillReorderInfo:
    """Immutable pre-admission metadata used only for waiting-list ordering."""

    prefill_token_count: int
    is_nonfinal_long_prefill: bool
    estimated_long_chunks: int


class _PrefillReorderer:
    """Order waiting prefills without applying scheduler side effects."""

    def __init__(self, scheduler: 'Scheduler'):
        self.scheduler = scheduler
        self._info_cache: dict[int, _PrefillReorderInfo] = {}

    def reorder(self,
                waiting: SeqList,
                allow_long_prefill: bool,
                prefer_long_prefill: bool):
        """Return waiting requests in the order the prefill loop should try."""
        waiting = sorted(waiting, key=lambda seq: seq.arrive_time)
        original = waiting
        # A completed remote load owns populated private blocks but has not yet
        # run the suffix that makes those blocks useful.  Keep that one-shot
        # admission lane ahead of ordinary waiters.  Otherwise a later remote
        # load can repeatedly evict and reload the completed request under KV
        # pressure (the load/preemption storm fixed by vLLM #44560).
        remote_ready = [
            seq for seq in waiting
            if self.scheduler._has_remote_prefill_reservation(seq)
        ]
        waiting = [
            seq for seq in waiting
            if not self.scheduler._has_remote_prefill_reservation(seq)
        ]
        if prefer_long_prefill:
            # Long-work turns choose one long waiter first. The size policy only
            # reorders this long lane; it is not global shortest-prefill-first
            # admission.
            long_turn_order = self._reorder_for_long_turn(waiting)
            if long_turn_order is not None:
                reordered = remote_ready + long_turn_order
                return self._warn_if_not_permutation(original, reordered)

        if allow_long_prefill:
            reordered = remote_ready + waiting
            return self._warn_if_not_permutation(original, reordered)

        reordered = remote_ready + self._reorder_for_short_turn(waiting)
        return self._warn_if_not_permutation(original, reordered)

    def _warn_if_not_permutation(self, original: SeqList, reordered: SeqList):
        """Warn if reorder drops, duplicates, or substitutes waiting
        sequences."""
        original_ids = [id(seq) for seq in original]
        reordered_ids = [id(seq) for seq in reordered]
        if len(original_ids) == len(reordered_ids) and Counter(original_ids) == Counter(reordered_ids):
            return reordered

        logger.warning('Unexpected prefill reorder result: original_len=%s reordered_len=%s '
                       'original_sample=%s reordered_sample=%s',
                       len(original), len(reordered), self._seq_id_sample(original), self._seq_id_sample(reordered))
        return reordered

    @staticmethod
    def _seq_id_sample(seqs: SeqList):
        return [(seq.session_id, seq.seq_id) for seq in seqs[:5]]

    def _get_reorder_info(self, seq: SchedulerSequence):
        """Return reorder-only info before prefix-cache side effects.

        Prefix-cache match/rollback mutates the remaining prompt. Keep this cache confined to waiting-list ordering and
        recompute fresh values in the admission path.
        """
        seq_key = id(seq)
        info = self._info_cache.get(seq_key)
        if info is not None:
            return info

        scheduler = self.scheduler
        chunk_limit = scheduler._long_context_chunk_limit(seq)
        if seq.num_token_ids <= chunk_limit:
            info = _PrefillReorderInfo(prefill_token_count=seq.num_token_ids,
                                       is_nonfinal_long_prefill=False,
                                       estimated_long_chunks=1)
        else:
            kv_token_limit = scheduler._next_long_context_chunk_end(seq, chunk_limit)
            safe_chunk_limit = max(1, chunk_limit)
            info = _PrefillReorderInfo(
                prefill_token_count=max(0, kv_token_limit - seq.num_history_ids),
                is_nonfinal_long_prefill=True,
                estimated_long_chunks=max(1, (seq.num_token_ids + safe_chunk_limit - 1) // safe_chunk_limit),
            )
        self._info_cache[seq_key] = info
        return info

    def _long_priority_key(self, seq: SchedulerSequence, now: float):
        """Prefer smaller long prompts, with age credit to avoid starvation."""
        scheduler = self.scheduler
        info = self._get_reorder_info(seq)
        wait_age = max(0.0, now - seq.arrive_time)
        age_credit = int(wait_age // scheduler._long_prefill_aging_seconds_per_chunk)
        age_adjusted_chunks = info.estimated_long_chunks - age_credit
        return age_adjusted_chunks, info.estimated_long_chunks, seq.arrive_time

    def _split_by_prefill_kind(self, waiting: SeqList):
        """Split waiting requests into normal/final and non-final long
        prefill."""
        normal_waiting: SeqList = []
        long_waiting: SeqList = []
        for seq in waiting:
            if self._get_reorder_info(seq).is_nonfinal_long_prefill:
                long_waiting.append(seq)
            else:
                normal_waiting.append(seq)
        return normal_waiting, long_waiting

    def _sort_normal_prefills(self, waiting: SeqList):
        return sorted(waiting,
                      key=lambda seq: (self._get_reorder_info(seq).prefill_token_count, seq.arrive_time))

    def _sort_long_prefills(self, waiting: SeqList):
        scheduler = self.scheduler
        if scheduler._long_prefill_policy != 'size':
            return waiting
        now = time.perf_counter()
        return sorted(waiting, key=lambda seq: self._long_priority_key(seq, now))

    def _reorder_for_long_turn(self, waiting: SeqList):
        """Choose one long waiter, then fill the turn with normal prefills."""
        normal_waiting, long_waiting = self._split_by_prefill_kind(waiting)
        if len(long_waiting) == 0:
            return None

        long_waiting = self._sort_long_prefills(long_waiting)
        normal_waiting = self._sort_normal_prefills(normal_waiting)
        return [long_waiting[0]] + normal_waiting + long_waiting[1:]

    def _reorder_for_short_turn(self, waiting: SeqList):
        """Prioritize normal/final prefills while preserving long waiters."""
        normal_waiting, long_waiting = self._split_by_prefill_kind(waiting)
        return self._sort_normal_prefills(normal_waiting) + long_waiting


@dataclass(frozen=True)
class _PrefillAdmissionResult:
    """Outcome from trying to admit one waiting prefill request."""

    admitted: bool
    prefill_token_count: int = 0
    should_skip: bool = False
    remote_loading: bool = False

    @classmethod
    def admit(cls, prefill_token_count: int):
        return cls(admitted=True, prefill_token_count=prefill_token_count)

    @classmethod
    def skip(cls):
        return cls(admitted=False, should_skip=True)

    @classmethod
    def stop(cls):
        return cls(admitted=False)

    @classmethod
    def load_remote(cls):
        return cls(admitted=False, remote_loading=True)

    @property
    def should_stop(self):
        return not self.admitted and not self.should_skip and not self.remote_loading


@dataclass(frozen=True)
class _PrefixMatchBaseline:
    """Sequence-owned state that a tentative trie match must not discard."""

    step: int
    num_blocks: int
    trie_cursor: Any
    match_start_step: int
    cached_tokens: int
    kv_token_limit: int | None
    fresh_block_range: range | None
    trie_block_map: dict[int, int]


class _PrefillAdmissionAttempt:
    """Try to admit one waiting prefill sequence.

    The attempt owns all tentative prefix-cache side effects for the sequence:
    match, SSM restore pinning, eviction, runtime-state checks, allocation, and
    rollback. The outer scheduler loop still owns queue traversal and decides
    whether a rejected candidate is skipped or ends the current prefill turn.
    """

    def __init__(self,
                 scheduler: 'Scheduler',
                 seq: SchedulerSequence,
                 evictable_waiting: SeqList,
                 prealloc_size: int,
                 token_count: int,
                 has_admitted: bool,
                 allow_long_prefill: bool):
        self.scheduler = scheduler
        self.seq = seq
        self.evictable_waiting = evictable_waiting
        self.prealloc_size = prealloc_size
        self.token_count = token_count
        self.has_admitted = has_admitted
        # A completed remote transfer must get one opportunity to consume its
        # reservation even on a short-prefill turn.  Deferring it behind the
        # long-prefill cadence leaves populated blocks exposed indefinitely.
        self.allow_long_prefill = (
            allow_long_prefill
            or scheduler._has_remote_prefill_reservation(seq)
        )
        self._alloc_size = prealloc_size
        self._gate_match_stats_snapshot = None
        self._gate_match_rollback_result = None
        self._remote_local_token_len: int | None = None
        self._remote_token_len: int | None = None
        self._remote_fallback_step: int | None = None
        overlap = seq.prefix_cache.recompute_overlap
        self._prefix_match_baseline = _PrefixMatchBaseline(
            step=int(seq.num_history_ids),
            num_blocks=int(seq.num_blocks),
            trie_cursor=seq.prefix_cache.trie_cursor,
            match_start_step=int(seq.prefix_cache.match_start_step),
            cached_tokens=int(seq.cached_tokens),
            kv_token_limit=seq.kv_token_limit,
            fresh_block_range=overlap.fresh_block_range,
            trie_block_map=dict(overlap.trie_block_map),
        )

    def run(self):
        """Run the admission route for one waiting prefill.

        1. Check prefill gates.
        2. Return skip/stop if a gate rejects the candidate.
        3. Try resource admission, including prefix-cache rollback on failure.
        4. Return skip/stop if resources block the candidate.
        5. On success, allocate blocks/states and publish any prefix-cache hit.
        """
        gate_result = self._check_prefill_admission_gates()
        if gate_result is not None:
            return self._check_result(gate_result)

        resource_result = self._admit_resources()
        if resource_result is not None:
            return self._check_result(resource_result)

        return self._check_result(self._finish_admission())

    def _rollback_gate(self, stats_snapshot, reason: str):
        """Rollback a tentative prefix hit and return any gate-only rejection.

        A prefill gate may do a tentative prefix-cache match before resource
        admission. If that match is rolled back, the candidate should follow
        the gate's original skip/stop result. Matches created after the gate
        return ``None`` so the resource branch keeps its own retry/stop behavior.
        """
        self._rollback_prefix_match(stats_snapshot, reason)
        return self._gate_match_rollback_result

    def _check_result(self, result: _PrefillAdmissionResult):
        if result.admitted and result.should_skip:
            self._warn_unexpected_state(
                f'admission result both admits and skips: prefill_token_count={result.prefill_token_count}')
        if not result.admitted and result.prefill_token_count != 0:
            self._warn_unexpected_state(
                f'rejected admission result carries token count: prefill_token_count={result.prefill_token_count}')
        if result.remote_loading and (result.admitted or result.should_skip):
            self._warn_unexpected_state('remote load result has conflicting admission flags')
        return result

    def _warn_unexpected_state(self, message: str):
        seq = self.seq
        logger.warning('Unexpected prefill admission state: session_id=%s seq_id=%s %s',
                       seq.session_id, seq.seq_id, message)

    def _admit_resources(self):
        if self.scheduler.block_trie.enabled:
            return self._admit_prefix_cache_resources()
        lookup_result = self._query_external_prefix(None)
        if lookup_result is not None:
            return lookup_result
        if not self._prepare_and_evict():
            return _PrefillAdmissionResult.stop()
        return None

    def _admit_prefix_cache_resources(self):
        """Admit resources for prefix-cache scheduling.

        Route map:
        1. Use or create the tentative prefix-cache match.
        2. Pin any SSM restore state required by the match.
        3. Prepare allocation limits and evict KV/state resources.
        4. For SSM, verify a runtime state slot is still available.

        Any failure rolls the tentative match back. A match created only to pass
        a prefill gate returns that gate's skip/stop result after rollback;
        normal resource failures keep their local retry/stop behavior here.
        """
        scheduler = self.scheduler
        seq = self.seq
        stats_snapshot = self._gate_match_stats_snapshot
        if stats_snapshot is None and not self._has_private_local_tail():
            stats_snapshot = scheduler.block_trie.stats.snapshot()
            scheduler.block_trie.match(seq)

        lookup_result = self._query_external_prefix(stats_snapshot)
        if lookup_result is not None:
            return lookup_result

        had_ssm_restore = scheduler.is_ssm and seq.prefix_cache.restore.is_selected
        if not scheduler._pin_ssm_restore_if_needed(seq):
            result = self._rollback_gate(stats_snapshot, 'failed to pin SSM restore checkpoint')
            if result is not None:
                return result

        if not self._prepare_and_evict():
            if not had_ssm_restore:
                result = self._rollback_gate(stats_snapshot, 'eviction failed')
                if result is not None:
                    return result
                return _PrefillAdmissionResult.stop()

            # A matched SSM restore may be pinning the only checkpoint state
            # that eviction would otherwise free. Roll it back once and retry
            # eviction before declaring the sequence unschedulable.
            result = self._rollback_gate(stats_snapshot, 'eviction failed with pinned SSM restore')
            if result is not None:
                return result
            if not self._prepare_and_evict():
                return _PrefillAdmissionResult.stop()

        if scheduler.is_ssm and not scheduler._ensure_runtime_state_available():
            result = self._rollback_gate(stats_snapshot, 'no runtime SSM state available')
            if result is not None:
                return result
            if not self._prepare_and_evict():
                return _PrefillAdmissionResult.stop()
            if not scheduler._ensure_runtime_state_available():
                seq.kv_token_limit = None
                return _PrefillAdmissionResult.stop()

        return None

    def _query_external_prefix(self, stats_snapshot):
        """Poll remote prefix lookup after the tentative local L1 match."""
        scheduler = self.scheduler
        connector = scheduler.kv_connector
        seq = self.seq
        # Mooncake Store task 7 intentionally excludes recurrent/SSM state and
        # routed-expert replay, which is not part of the stored value schema.
        if connector is None or scheduler.is_ssm or seq.return_routed_experts:
            return None
        # The just-completed lookup/load already established the longest
        # available remote prefix for this request snapshot.  Re-querying
        # before its first forward can only delay admission and, under memory
        # pressure, used to start another identical full-prefix load.
        if scheduler._has_remote_prefill_reservation(seq):
            return None

        fallback_step = int(seq.num_history_ids)
        local_token_len = fallback_step // seq.block_size * seq.block_size
        num_external_tokens, load_async = connector.get_num_new_matched_tokens(
            seq,
            local_token_len,
        )
        if num_external_tokens is None:
            if stats_snapshot is not None:
                scheduler._rollback_unscheduled_prefix_match(
                    seq,
                    stats_snapshot,
                    connector_preempted=False,
                    baseline=self._prefix_match_baseline,
                )
            return _PrefillAdmissionResult.skip()

        num_external_tokens = int(num_external_tokens)
        if num_external_tokens <= 0:
            return None
        remote_token_len = min(
            local_token_len + num_external_tokens,
            int(seq.get_prefix_cache_max_match_step()),
        )
        remote_token_len = remote_token_len // seq.block_size * seq.block_size
        if remote_token_len <= local_token_len:
            return None
        if not load_async:
            logger.debug(
                'Connector returned a remote prefix without async flag; '
                'LMDeploy still isolates the destination from forward: seq_id=%s',
                seq.seq_id,
            )
        self._remote_local_token_len = local_token_len
        self._remote_token_len = remote_token_len
        self._remote_fallback_step = fallback_step
        return None

    def _match_prefix_for_prefill_gate(self):
        """Tentatively match once so a request can be rechecked by a gate."""
        scheduler = self.scheduler
        if (not scheduler.block_trie.enabled
                or self._has_private_local_tail()):
            return None
        stats_snapshot = scheduler.block_trie.stats.snapshot()
        scheduler.block_trie.match(self.seq)
        return stats_snapshot

    def _has_private_local_tail(self) -> bool:
        """Whether GPU blocks after the trie cursor must stay authoritative.

        ``BlockTrie.match`` appends matches after its cursor.  For a preserved
        multi-turn request with a partial or preallocated tail, appending a
        full match would leave that private block at the same logical index and
        place the match one slot too late.  Keep the local table intact and let
        the external lookup start at the preceding full block boundary.
        """
        seq = self.seq
        if self.scheduler.is_ssm:
            return False
        step = int(seq.num_history_ids)
        return seq.num_blocks > step // seq.block_size

    def _keep_gate_prefix_match(self, stats_snapshot, rollback_result: _PrefillAdmissionResult):
        """Keep a gate-enabling match for the following resource admission."""
        self._gate_match_stats_snapshot = stats_snapshot
        self._gate_match_rollback_result = rollback_result

    def _token_budget_rejection(self):
        if self.allow_long_prefill:
            return _PrefillAdmissionResult.stop()
        return _PrefillAdmissionResult.skip()

    def _check_prefill_admission_gates(self):
        """Apply prefill gates, tentatively matching only when it may help."""
        scheduler = self.scheduler
        seq = self.seq
        token_budget = scheduler.cache_config.max_prefill_token_num
        prefill_token_count = scheduler._prefill_admission_token_count(seq)
        is_nonfinal_long_prefill = scheduler._prefill_kv_token_limit(seq) is not None

        if is_nonfinal_long_prefill and not self.allow_long_prefill:
            stats_snapshot = self._match_prefix_for_prefill_gate()
            if stats_snapshot is None:
                return _PrefillAdmissionResult.skip()
            if scheduler._prefill_kv_token_limit(seq) is not None:
                self._rollback_prefix_match(stats_snapshot, 'still non-final long prefill on short turn')
                return _PrefillAdmissionResult.skip()
            self._keep_gate_prefix_match(stats_snapshot, _PrefillAdmissionResult.skip())
            prefill_token_count = scheduler._prefill_admission_token_count(seq)

        exceeds_token_budget = self.has_admitted and self.token_count + prefill_token_count > token_budget
        if not exceeds_token_budget:
            return None

        if self._gate_match_stats_snapshot is None:
            stats_snapshot = self._match_prefix_for_prefill_gate()
            if stats_snapshot is not None:
                prefill_token_count = scheduler._prefill_admission_token_count(seq)
                if self.token_count + prefill_token_count <= token_budget:
                    self._keep_gate_prefix_match(stats_snapshot, self._token_budget_rejection())
                    return None
                self._rollback_prefix_match(stats_snapshot, 'still exceeds prefill token budget')
        else:
            self._rollback_prefix_match(self._gate_match_stats_snapshot, 'still exceeds prefill token budget')
        return self._token_budget_rejection()

    def _prepare_and_evict(self):
        """Apply chunk allocation limits and evict for this prefill."""
        scheduler = self.scheduler
        seq = self.seq
        if self._remote_token_len is not None:
            seq.kv_token_limit = self._remote_token_len
            # The remote prefix ends at a block boundary and deliberately
            # leaves at least one token for sampling. Reserve that forward
            # destination now; otherwise a single load can consume the whole
            # pool and deadlock when it later tries to run its tail token.
            alloc_size = max(1, self.prealloc_size)
            evict_alloc_size = alloc_size
            # Reserve this candidate's complete remaining prefill as well as
            # the load destination itself. Reserving only the first chunk can
            # admit several loads that all make one step and then deadlock.
            load_required_blocks = scheduler.block_manager.num_required_blocks(
                seq, alloc_size)
            candidate_required_blocks = (
                scheduler._estimate_remote_prefill_reservation(
                    seq,
                    remote_token_len=self._remote_token_len,
                    prealloc_size=alloc_size,
                ))
            candidate_reservation_blocks = max(
                0, candidate_required_blocks - load_required_blocks)
            evict_alloc_size += candidate_reservation_blocks * seq.block_size
            if self._remote_fallback_step != self._remote_local_token_len:
                # Replacing an existing partial block needs one additional
                # physical row while the original is pinned for fallback.
                evict_alloc_size += seq.block_size
        else:
            alloc_size = scheduler._prepare_prefill_allocation(seq, self.prealloc_size)
            evict_alloc_size = alloc_size
            if (scheduler.kv_connector is not None
                    and seq.kv_token_limit is not None):
                # A local long prefill competes with remote loads for the same
                # pool. Always gate against the complete *current* ISL before
                # admitting a chunk.  A multi-turn sequence can still carry
                # the completed prior turn's target until its next scheduler
                # output is observed; membership alone therefore cannot prove
                # that the new, longer turn has already reserved its tail.
                required_blocks = scheduler.block_manager.num_required_blocks(
                    seq, alloc_size)
                target_blocks = scheduler._prefill_target_blocks(
                    seq, self.prealloc_size)
                target_required_blocks = max(
                    0, target_blocks - int(seq.num_blocks))
                extra_blocks = max(
                    0, target_required_blocks - required_blocks)
                evict_alloc_size += extra_blocks * seq.block_size
        # In-flight prefills still need free rows for later chunks. Make that
        # abstract headroom part of every admission check, excluding the
        # current request because its own allocation consumes its reservation.
        reserved_blocks = scheduler._remote_prefill_reserved_blocks(
            exclude_req_id=int(seq.seq_id),
        )
        evict_alloc_size += reserved_blocks * seq.block_size
        self._alloc_size = alloc_size
        if self._evict_for_seq(evict_alloc_size):
            return True
        seq.kv_token_limit = None
        return False

    def _evict_for_seq(self, alloc_size: int):
        """Evict stopped or skipped waiters until this sequence can run."""
        from itertools import chain
        scheduler = self.scheduler
        hanging = reversed(scheduler.hanging)
        waiting = reversed(self.evictable_waiting)
        evictable = scheduler._exclude_remote_prefill_victims(
            list(chain(hanging, waiting)))
        return scheduler.eviction_helper.evict_for_seq(self.seq, evictable, alloc_size)

    def _rollback_prefix_match(self, stats_snapshot, reason: str):
        seq = self.seq
        logger.debug('Rollback tentative prefix-cache match: session_id=%s seq_id=%s reason=%s '
                     'num_history_ids=%s restore_state=%s', seq.session_id, seq.seq_id, reason, seq.num_history_ids,
                     seq.prefix_cache.restore.slot)
        self.scheduler._rollback_unscheduled_prefix_match(
            seq,
            stats_snapshot,
            baseline=self._prefix_match_baseline,
        )

    def _finish_admission(self):
        scheduler = self.scheduler
        seq = self.seq
        # Prefix-cache matching can advance the sequence step and shrink the
        # remaining prefill tail. Charge the admitted batch with the
        # post-match/post-rollback cost, not the conservative pre-match
        # estimate used to decide whether this sequence is worth trying.
        prefill_token_count = scheduler._prefill_admission_token_count(seq)
        if self._remote_token_len is not None:
            assert self._remote_local_token_len is not None
            assert self._remote_fallback_step is not None
            scheduler._start_remote_kv_load(
                seq,
                local_token_len=self._remote_local_token_len,
                remote_token_len=self._remote_token_len,
                fallback_step=self._remote_fallback_step,
                prealloc_size=self._alloc_size,
            )
            return _PrefillAdmissionResult.load_remote()
        scheduler.block_manager.allocate(seq, self._alloc_size)
        if scheduler.block_trie.enabled:
            scheduler.block_trie.allocate(seq)
        if scheduler.is_ssm:
            scheduler.state_manager.allocate(seq)
        if scheduler.block_trie.enabled:
            scheduler._finish_prefix_cache_schedule(seq)
        if seq.kv_token_limit is not None:
            scheduler._register_prefill_reservation(
                seq,
                prealloc_size=self.prealloc_size,
            )
        scheduler._consume_remote_prefill_reservation(seq)
        return _PrefillAdmissionResult.admit(prefill_token_count)


class Scheduler:
    """Tools to schedule next step.

    Args:
        scheduler_config (SchedulerConfig): The config of scheduler.
        cache_config (CacheConfig): The config of cache info.
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        seq_meta: SequenceMeta | None = None,
        kv_connector: 'KVConnectorBase | None' = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.sessions: dict[int, SchedulerSession] = OrderedDict()
        self.kv_connector = kv_connector

        # For Disaggregation
        self.locked_sessions: dict[int, SchedulerSession] = OrderedDict()

        self.state_manager = build_state_manager(self.cache_config)
        self.block_manager = build_block_manager(cache_config)
        self.is_ssm = len(self.cache_config.states_shapes) > 0
        checkpoint_state_manager = self.state_manager if self.is_ssm else None
        self.block_trie = BlockTrie(allocator=self.block_manager.allocator,
                                   block_size=self.cache_config.block_size,
                                   enabled=self.cache_config.enable_prefix_caching,
                                   checkpoint_state_manager=checkpoint_state_manager)

        self.eviction_helper = build_eviction_helper(self, self.scheduler_config.eviction_type)

        seq_meta = seq_meta or SequenceMeta(self.cache_config.block_size)
        self.seq_meta = seq_meta
        self.seq_manager = SequenceManager(seq_meta)
        self.scheduler_tick = 0
        self._long_prefill_policy = _envs.opt_ttft_policy
        self._long_prefill_aging_seconds_per_chunk = max(0.001, _envs.opt_ttft_aging_sec)

        # Async connector saves own one reference to every logical block they
        # read.  Request eviction can then release its reference immediately
        # without allowing the physical cache rows to be reused before all TP
        # workers finish the save wave.
        self._kv_seq_generations: dict[int, int] = {}
        self._pending_kv_saves: dict[int, _PendingKVSave] = {}
        self._pending_kv_loads: dict[int, _PendingKVLoad] = {}
        self._active_kv_load_by_req: dict[int, int] = {}
        # Successful remote loads retain a one-shot admission marker until
        # their first real prefill allocation.  Capacity is tracked separately
        # through the complete prompt: a long prefill must not admit another
        # load (or grow another request) into rows needed by a later chunk.
        self._remote_prefill_reservations: dict[int, int] = {}
        self._prefill_reservation_targets: dict[
            int, tuple[SchedulerSequence, int]
        ] = {}

    def tick(self):
        """Mark one scheduler progress step (once per forward dispatch)."""
        self.scheduler_tick += 1

    def _has_remote_prefill_reservation(self, seq: SchedulerSequence) -> bool:
        """Whether a completed remote load still awaits its first forward."""
        return int(seq.seq_id) in self._remote_prefill_reservations

    def _consume_remote_prefill_reservation(self, seq: SchedulerSequence) -> None:
        """Consume only the completed-load first-admission marker."""
        self._remote_prefill_reservations.pop(int(seq.seq_id), None)

    def _release_prefill_reservation(self, seq: SchedulerSequence) -> None:
        """Release both remote-ready protection and full-prefill headroom."""
        req_id = int(seq.seq_id)
        self._remote_prefill_reservations.pop(req_id, None)
        self._prefill_reservation_targets.pop(req_id, None)

    def _prefill_target_blocks(
        self,
        seq: SchedulerSequence,
        prealloc_size: int,
    ) -> int:
        """Return the absolute rows required by the complete prompt."""
        block_size = self.cache_config.block_size
        target_tokens = int(seq.num_all_ids) + max(0, int(prealloc_size))
        return (target_tokens + block_size - 1) // block_size

    def _register_prefill_reservation(
        self,
        seq: SchedulerSequence,
        *,
        prealloc_size: int = 0,
        target_blocks: int | None = None,
    ) -> None:
        """Retain complete-ISL capacity until the final prefill completes."""
        if self.kv_connector is None:
            return
        req_id = int(seq.seq_id)
        if target_blocks is None:
            target_blocks = self._prefill_target_blocks(seq, prealloc_size)
        previous = self._prefill_reservation_targets.get(req_id)
        if previous is not None:
            target_blocks = max(int(target_blocks), int(previous[1]))
        self._prefill_reservation_targets[req_id] = (seq, int(target_blocks))

    def _release_completed_prefill_reservations(
        self,
        seqs: SeqList,
    ) -> None:
        """Release capacity only after scheduler-visible final-prefill output.

        Long-context intermediate chunks advance ``num_history_ids`` before
        their forward finishes.  The last chunk does not: its model output
        advances the sequence to ``input_end_pos``.  This boundary therefore
        distinguishes an allocated final chunk from a completed one.
        """
        for seq in seqs:
            req_id = int(seq.seq_id)
            if req_id not in self._prefill_reservation_targets:
                continue
            if self._has_remote_prefill_reservation(seq):
                continue
            if int(seq.num_history_ids) >= int(seq.input_end_pos):
                self._prefill_reservation_targets.pop(req_id, None)

    def _exclude_remote_prefill_victims(self, seqs: SeqList) -> SeqList:
        """Keep successfully loaded waiters out of every eviction path."""
        return [
            seq for seq in seqs
            if not self._has_remote_prefill_reservation(seq)
        ]

    def _remote_prefill_reserved_blocks(
        self,
        *,
        exclude_req_id: int | None = None,
    ) -> int:
        """Return headroom owed to other in-flight prefills.

        Targets cover the entire remaining input, not merely the next chunk. They shrink dynamically as blocks are
        allocated. Pending remote loads are not registered until completion and are counted separately.
        """
        total = sum(
            max(0, int(target_blocks) - int(seq.num_blocks))
            for req_id, (seq, target_blocks)
            in self._prefill_reservation_targets.items()
            if req_id != exclude_req_id
        )
        total += sum(
            int(pending.prefill_reservation_blocks)
            for pending in self._pending_kv_loads.values()
            if not pending.cancelled and pending.req_id != exclude_req_id
        )
        return total

    def _estimate_remote_prefill_reservation(
        self,
        seq: SchedulerSequence,
        *,
        remote_token_len: int,
        prealloc_size: int,
    ) -> int:
        """Return rows needed to reach the complete-prefill allocation target.

        ``remote_token_len`` is validated by the caller and retained in the
        signature to make the admission contract explicit.  The reservation
        intentionally covers every remaining ISL chunk (vLLM #44560).
        """
        if remote_token_len > int(seq.num_all_ids):
            raise ValueError('remote prefix exceeds request token length')
        target_blocks = self._prefill_target_blocks(seq, prealloc_size)
        return max(0, target_blocks - int(seq.num_blocks))

    def shutdown(self) -> None:
        """Release scheduler-side connector resources exactly once."""
        connector = getattr(self, 'kv_connector', None)
        self.kv_connector = None
        try:
            if connector is not None:
                connector.shutdown()
        finally:
            self._release_all_kv_save_pins()
            self._release_all_kv_load_pins()

    def _start_remote_kv_load(
        self,
        seq: SchedulerSequence,
        *,
        local_token_len: int,
        remote_token_len: int,
        fallback_step: int,
        prealloc_size: int,
    ) -> Any:
        """Build private load destinations while preserving a partial tail."""
        connector = self.kv_connector
        if connector is None:
            raise RuntimeError('remote KV allocation requires a connector')
        block_size = self.cache_config.block_size
        if (local_token_len % block_size != 0
                or remote_token_len % block_size != 0
                or remote_token_len <= local_token_len):
            raise ValueError('remote KV token bounds must be increasing and block aligned')
        if not local_token_len <= fallback_step < local_token_len + block_size:
            raise ValueError('fallback_step must be within the local boundary block')

        local_blocks = local_token_len // block_size
        remote_blocks = remote_token_len // block_size
        fallback_ids = np.empty((0, ), dtype=np.int64)
        fallback_detached = False
        logical_ids = np.empty((0, ), dtype=np.int64)
        load_suffix_pinned = False
        generation = self._kv_seq_generations.setdefault(int(seq.seq_id), 0)
        load_request = None
        try:
            if fallback_step > local_token_len:
                all_logical_ids = seq.logical_blocks.get_real_blocks()
                if len(all_logical_ids) <= local_blocks:
                    raise RuntimeError(
                        f'request {seq.seq_id} has no partial fallback block '
                        f'at index {local_blocks}')
                fallback_ids = np.asarray(
                    all_logical_ids[local_blocks:local_blocks + 1],
                    dtype=np.int64,
                ).copy()
                self.block_manager.pin_logical_blocks(fallback_ids)
                self.block_manager.truncate(seq, local_blocks)
                fallback_detached = True

            self.block_manager.allocate(seq, prealloc_size)
            all_logical_ids = seq.logical_blocks.get_real_blocks()
            if len(all_logical_ids) < remote_blocks:
                raise RuntimeError(
                    f'request {seq.seq_id} owns {len(all_logical_ids)} blocks, '
                    f'but remote load needs {remote_blocks}')
            logical_ids = np.asarray(
                all_logical_ids[local_blocks:remote_blocks],
                dtype=np.int64,
            ).copy()
            physical_ids = self.block_manager.pin_logical_blocks(logical_ids)
            load_suffix_pinned = True
            load_request = connector.update_state_after_alloc(
                seq,
                tuple(int(block_id) for block_id in physical_ids),
                remote_token_len - local_token_len,
                generation=generation,
            )
            if load_request is None:
                raise RuntimeError(
                    f'connector did not create a load plan for request {seq.seq_id}')
            expected = (
                int(seq.seq_id),
                generation,
                local_token_len,
                remote_token_len,
            )
            actual = (
                int(load_request.req_id),
                int(load_request.generation),
                int(load_request.local_token_len),
                int(load_request.remote_token_len),
            )
            if actual != expected:
                raise RuntimeError(
                    f'connector load plan mismatch for request {seq.seq_id}: '
                    f'{actual} != {expected}')
            load_id = int(load_request.load_id)
            if load_id in self._pending_kv_loads:
                raise RuntimeError(f'duplicate connector load_id {load_id}')
            req_id = int(seq.seq_id)
            if req_id in self._active_kv_load_by_req:
                raise RuntimeError(f'request {req_id} already has an active remote load')
            if tuple(int(block_id) for block_id in load_request.block_ids) != tuple(
                    int(block_id) for block_id in physical_ids):
                raise RuntimeError(
                    f'connector physical load suffix changed for request {seq.seq_id}')
            prefill_target_blocks = self._prefill_target_blocks(
                seq, prealloc_size)
            self._pending_kv_loads[load_id] = _PendingKVLoad(
                seq=seq,
                req_id=int(seq.seq_id),
                load_id=load_id,
                generation=generation,
                local_token_len=local_token_len,
                remote_token_len=remote_token_len,
                logical_block_ids=logical_ids,
                prefill_target_blocks=prefill_target_blocks,
                fallback_step=fallback_step,
                fallback_logical_block_ids=fallback_ids,
            )
            self._active_kv_load_by_req[req_id] = load_id
            seq.state.begin_remote_load()
            return load_request
        except BaseException:
            if load_request is not None:
                load_id = int(load_request.load_id)
                self._pending_kv_loads.pop(load_id, None)
                if self._active_kv_load_by_req.get(int(seq.seq_id)) == load_id:
                    self._active_kv_load_by_req.pop(int(seq.seq_id), None)
                connector.update_connector_output({
                    'cancelled_load_ids': {load_id},
                })
            if load_suffix_pinned:
                self.block_manager.release_pinned_logical_blocks(logical_ids)
            self.block_manager.truncate(seq, local_blocks)
            if fallback_detached:
                # Adopt the fallback pin as the restored sequence reference.
                seq.logical_blocks.append(fallback_ids)
            elif len(fallback_ids) > 0:
                self.block_manager.release_pinned_logical_blocks(fallback_ids)
            seq.set_step(min(fallback_step, seq.num_all_ids))
            seq.kv_token_limit = None
            raise

    def mark_kv_connector_preempted(self, seq: SchedulerSequence) -> None:
        """Start a new connector generation before a sequence recomputes.

        A request ID survives recompute preemption in LMDeploy.  A generation therefore distinguishes work submitted
        against the old block table from work submitted after the request is allocated again.  Existing jobs remain
        pinned until workers acknowledge their save IDs.
        """
        self._release_prefill_reservation(seq)
        if self.kv_connector is None:
            return

        req_id = int(seq.seq_id)
        self._cancel_remote_kv_load(seq, make_waiting=True)
        old_generation = self._kv_seq_generations.get(req_id, 0)
        self._kv_seq_generations[req_id] = old_generation + 1

    def build_kv_connector_metadata(
        self,
        running: SeqList,
        token_lens: tuple[int, ...] | None = None,
    ) -> Any | None:
        """Build and pin metadata for one executor dispatch.

        ``token_lens`` is supplied only for prefill-like forwards.  Decode
        still emits empty metadata so preemption notifications reach workers,
        but it never creates save work: its token IDs can be one step stale due
        to engine prefetching.
        """
        connector = self.kv_connector
        if connector is None:
            return None

        if token_lens is None:
            token_lens = ()
        if len(token_lens) not in (0, len(running)):
            raise ValueError('token_lens must be empty or contain one value per running sequence')

        aligned_token_lens: list[int] = []
        logical_block_ids: list[tuple[int, ...]] = []
        physical_block_ids: list[tuple[int, ...]] = []
        generations: list[int] = []
        snapshots: dict[int, tuple[np.ndarray, tuple[int, ...], int]] = {}
        block_size = self.cache_config.block_size

        if token_lens:
            for seq, token_len_value in zip(running, token_lens):
                token_len = int(token_len_value)
                if token_len < 0:
                    raise ValueError('connector token length must be non-negative')
                token_len = token_len // block_size * block_size
                num_blocks = token_len // block_size

                all_logical_ids = seq.logical_blocks.get_real_blocks()
                if num_blocks > len(all_logical_ids):
                    raise RuntimeError(
                        f'request {seq.seq_id} has {len(all_logical_ids)} allocated blocks '
                        f'but connector save needs {num_blocks}')
                logical_ids = np.asarray(all_logical_ids[:num_blocks], dtype=np.int64).copy()
                physical_ids = self.block_manager.resolve_gpu_block_offsets(logical_ids)
                physical_tuple = tuple(int(block_id) for block_id in physical_ids)
                logical_tuple = tuple(int(block_id) for block_id in logical_ids)
                generation = self._kv_seq_generations.setdefault(int(seq.seq_id), 0)

                aligned_token_lens.append(token_len)
                logical_block_ids.append(logical_tuple)
                physical_block_ids.append(physical_tuple)
                generations.append(generation)
                snapshots[int(seq.seq_id)] = (logical_ids, physical_tuple, generation)

        scheduler_output = SchedulerOutput(
            running=running,
            swap_in_map={},
            swap_out_map={},
            copy_map={},
            connector_token_lens=tuple(aligned_token_lens),
            connector_block_ids=tuple(physical_block_ids),
            connector_logical_block_ids=tuple(logical_block_ids),
            connector_generations=tuple(generations),
            # Pinned old-generation jobs always drain to completion.  Do not
            # let a worker that has not enqueued the job yet interpret a
            # preemption notice as permission to acknowledge it early.
            preempted_save_ids=(),
        )
        metadata = connector.build_connector_meta(scheduler_output)
        try:
            self._lease_kv_load_metadata(metadata)
            self._pin_kv_connector_metadata(metadata, snapshots)
        except BaseException:
            self.rollback_kv_connector_metadata(metadata)
            raise
        return metadata

    def build_kv_connector_progress_metadata(self) -> Any | None:
        """Build connector-only work when there is no model forward.

        Only ready loads are emitted. Building acquires a temporary dispatch
        lease; the engine commits successful delivery with
        :meth:`mark_kv_connector_metadata_dispatched`.  A failed RPC therefore
        rolls the lease back and retries the same load ID without serializing
        in-flight work every poll.
        """
        connector = self.kv_connector
        if connector is None or not self._has_ready_kv_loads():
            return None
        scheduler_output = SchedulerOutput(
            running=[],
            swap_in_map={},
            swap_out_map={},
            copy_map={},
        )
        metadata = connector.build_connector_meta(scheduler_output)
        if not getattr(metadata, 'load_requests', ()):
            return None
        try:
            self._lease_kv_load_metadata(metadata)
        except BaseException:
            self.rollback_kv_connector_metadata(metadata)
            raise
        return metadata

    def _lease_kv_load_metadata(self, metadata: Any) -> None:
        """Protect the build-to-RPC await window from cancellation."""
        leased_ids = []
        try:
            for load_request in getattr(metadata, 'load_requests', ()):
                load_id = int(load_request.load_id)
                pending = self._pending_kv_loads.get(load_id)
                if pending is None:
                    raise RuntimeError(
                        f'connector metadata refers to unknown load_id {load_id}')
                if ((pending.cancelled and not pending.needs_fence)
                        or pending.dispatching or pending.dispatched):
                    raise RuntimeError(
                        f'connector load_id {load_id} is not ready for dispatch')
                pending.dispatching = True
                leased_ids.append(load_id)
        except BaseException:
            for load_id in leased_ids:
                pending = self._pending_kv_loads.get(load_id)
                if pending is not None:
                    pending.dispatching = False
            raise

    def mark_kv_connector_metadata_dispatched(self, metadata: Any | None) -> None:
        """Commit metadata only after its executor RPC was accepted."""
        if metadata is None:
            return
        load_requests = tuple(getattr(metadata, 'load_requests', ()) or ())
        if not load_requests:
            return
        connector = self.kv_connector
        if connector is not None:
            marker = getattr(connector, 'mark_connector_meta_dispatched', None)
            if marker is not None:
                marker(metadata)
            else:
                connector.update_connector_output({
                    'dispatched_load_ids': {
                        int(load_request.load_id)
                        for load_request in load_requests
                    },
                })
        for load_request in load_requests:
            pending = self._pending_kv_loads.get(int(load_request.load_id))
            if pending is None:
                continue
            if (pending.req_id != int(load_request.req_id)
                    or pending.generation != int(load_request.generation)):
                raise RuntimeError(
                    f'stale connector load dispatch for load_id {load_request.load_id}')
            pending.dispatching = False
            pending.dispatched = True
            pending.needs_fence = False

    def _has_ready_kv_loads(self) -> bool:
        return any(
            not pending.dispatching and not pending.dispatched
            and (not pending.cancelled or pending.needs_fence)
            for pending in self._pending_kv_loads.values()
        )

    def _pin_kv_connector_metadata(
        self,
        metadata: Any,
        snapshots: dict[int, tuple[np.ndarray, tuple[int, ...], int]],
    ) -> None:
        """Acquire one allocator reference for every metadata save wave."""
        save_requests = getattr(metadata, 'save_requests', ())
        pinned_save_ids: list[int] = []
        try:
            for save_request in save_requests:
                save_id = int(save_request.save_id)
                req_id = int(save_request.req_id)
                generation = int(save_request.generation)
                if save_id in self._pending_kv_saves:
                    raise RuntimeError(f'duplicate connector save_id {save_id}')
                if req_id not in snapshots:
                    raise RuntimeError(f'connector metadata refers to unscheduled request {req_id}')

                logical_ids, physical_ids, expected_generation = snapshots[req_id]
                num_blocks = int(save_request.token_len) // self.cache_config.block_size
                if int(save_request.token_len) % self.cache_config.block_size != 0:
                    raise ValueError('connector save token_len must be block aligned')
                if generation != expected_generation:
                    raise RuntimeError(
                        f'connector generation mismatch for request {req_id}: '
                        f'{generation} != {expected_generation}')
                if len(save_request.block_ids) != num_blocks or len(save_request.block_hashes) != num_blocks:
                    raise ValueError('connector save block IDs and hashes must match token_len')

                wave_logical_ids = logical_ids[:num_blocks].copy()
                if tuple(int(block_id) for block_id in save_request.block_ids) != physical_ids[:num_blocks]:
                    raise RuntimeError(f'connector physical block snapshot changed for request {req_id}')

                self.block_manager.pin_logical_blocks(wave_logical_ids)
                self._pending_kv_saves[save_id] = _PendingKVSave(
                    req_id=req_id,
                    generation=generation,
                    logical_block_ids=wave_logical_ids,
                )
                pinned_save_ids.append(save_id)
        except BaseException:
            self._release_kv_save_pins(pinned_save_ids)
            raise

    @staticmethod
    def _completed_kv_save_ids(connector_output: Any) -> set[int]:
        """Normalize executor connector output to completed save IDs."""
        if connector_output is None:
            return set()
        output = connector_output
        if isinstance(output, dict):
            output = output.get('completed_save_ids', output.get('finished_sending'))
        elif hasattr(output, 'completed_save_ids'):
            output = output.completed_save_ids
        elif hasattr(output, 'finished_sending'):
            output = output.finished_sending
        elif isinstance(output, tuple) and len(output) == 2:
            output = output[0]
        if output is None:
            return set()
        if isinstance(output, (set, frozenset, list, tuple)):
            return {int(save_id) for save_id in output}
        return set()

    def update_connector_output(self, connector_output: Any) -> None:
        """Consume all-TP completions and publish or roll back loaded KV."""
        completed_save_ids = self._completed_kv_save_ids(connector_output)
        completed_load_ids = self._completed_kv_load_ids(connector_output)
        failed_load_ids = self._failed_kv_load_ids(connector_output)
        connector = self.kv_connector
        try:
            if connector is not None:
                connector.update_connector_output(connector_output)
        finally:
            self._release_kv_save_pins(completed_save_ids)
            for load_id in completed_load_ids:
                self._finish_kv_load(
                    load_id,
                    failed=load_id in failed_load_ids,
                )

    @staticmethod
    def _output_ids(connector_output: Any, name: str) -> set[int]:
        if connector_output is None:
            return set()
        if isinstance(connector_output, dict):
            values = connector_output.get(name)
        else:
            values = getattr(connector_output, name, None)
        if values is None:
            return set()
        return {int(value) for value in values}

    @classmethod
    def _completed_kv_load_ids(cls, connector_output: Any) -> set[int]:
        completed = cls._output_ids(connector_output, 'completed_load_ids')
        if completed:
            return completed
        completed = cls._output_ids(connector_output, 'finished_recving')
        if completed:
            return completed
        if isinstance(connector_output, tuple) and len(connector_output) == 2:
            values = connector_output[1]
            if values is not None:
                return {int(value) for value in values}
        return set()

    @classmethod
    def _failed_kv_load_ids(cls, connector_output: Any) -> set[int]:
        return cls._output_ids(connector_output, 'failed_load_ids')

    def _finish_kv_load(self, load_id: int, *, failed: bool) -> None:
        pending = self._pending_kv_loads.pop(int(load_id), None)
        if pending is None:
            return
        seq = pending.seq
        current_generation = self._kv_seq_generations.get(pending.req_id)
        active = (
            not pending.cancelled
            and current_generation == pending.generation
            and self._active_kv_load_by_req.get(pending.req_id) == pending.load_id
            and seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
        )
        if self._active_kv_load_by_req.get(pending.req_id) == pending.load_id:
            self._active_kv_load_by_req.pop(pending.req_id, None)
        try:
            if not active:
                if (not pending.fallback_restored
                        and seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS):
                    self._detach_remote_kv_suffix(pending)
                return
            if failed:
                self._detach_remote_kv_suffix(pending)
            else:
                try:
                    seq.kv_token_limit = pending.remote_token_len
                    if self.block_trie.enabled:
                        # Publish only after every TP worker reports that the
                        # destination suffix is completely populated.
                        self.block_trie.allocate(seq)
                    seq.set_step(pending.remote_token_len)
                    seq.kv_token_limit = None
                    if self.block_trie.enabled:
                        self._finish_prefix_cache_schedule(seq)
                    self._register_prefill_reservation(
                        seq,
                        target_blocks=pending.prefill_target_blocks,
                    )
                    self._remote_prefill_reservations[pending.req_id] = (
                        pending.prefill_reservation_blocks)
                except BaseException:
                    logger.exception(
                        'Failed to publish completed remote KV; falling back '
                        'to local prefix: req_id=%s load_id=%s',
                        pending.req_id,
                        pending.load_id,
                    )
                    self._detach_remote_kv_suffix(pending)
            seq.state.finish_remote_load()
            logger.info(
                'Mooncake load applied: req_id=%s load_id=%s generation=%s '
                'failed=%s step=%s',
                pending.req_id,
                pending.load_id,
                pending.generation,
                failed,
                seq.num_history_ids,
            )
        finally:
            self.block_manager.release_pinned_logical_blocks(
                pending.logical_block_ids)
            if (len(pending.fallback_logical_block_ids) > 0
                    and not pending.fallback_restored):
                self.block_manager.release_pinned_logical_blocks(
                    pending.fallback_logical_block_ids)

    def _detach_remote_kv_suffix(self, pending: _PendingKVLoad) -> None:
        """Drop the request reference while preserving any transfer pin."""
        seq = pending.seq
        self._release_prefill_reservation(seq)
        local_blocks = pending.local_token_len // self.cache_config.block_size
        if len(seq.logical_blocks) >= local_blocks:
            self.block_manager.truncate(seq, local_blocks)
        if (len(pending.fallback_logical_block_ids) > 0
                and not pending.fallback_restored):
            # The fallback pin becomes the sequence's restored ownership.
            seq.logical_blocks.append(pending.fallback_logical_block_ids)
            pending.fallback_restored = True
        seq.set_step(min(pending.fallback_step, seq.num_all_ids))
        seq.kv_token_limit = None

    def cancel_remote_kv_load(
        self,
        seq: SchedulerSequence,
        *,
        make_waiting: bool = True,
    ) -> None:
        """Cancel scheduling ownership without racing an in-flight device
        write."""
        self._cancel_remote_kv_load(seq, make_waiting=make_waiting)

    def _cancel_remote_kv_load(
        self,
        seq: SchedulerSequence,
        *,
        make_waiting: bool,
    ) -> None:
        self._release_prefill_reservation(seq)
        pending_loads = [
            pending
            for pending in self._pending_kv_loads.values()
            if pending.req_id == int(seq.seq_id) and not pending.cancelled
        ]
        if not pending_loads:
            return
        cancelled_ids = set()
        for pending in pending_loads:
            pending.cancelled = True
            cancelled_ids.add(pending.load_id)
            if self._active_kv_load_by_req.get(pending.req_id) == pending.load_id:
                self._active_kv_load_by_req.pop(pending.req_id, None)
            if seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS:
                self._detach_remote_kv_suffix(pending)
                if make_waiting:
                    seq.state.finish_remote_load()
            if (not pending.dispatching and not pending.dispatched
                    and not pending.needs_fence):
                self._pending_kv_loads.pop(pending.load_id, None)
                self.block_manager.release_pinned_logical_blocks(
                    pending.logical_block_ids)
        connector = self.kv_connector
        if connector is not None and cancelled_ids:
            connector.update_connector_output({
                'cancelled_load_ids': cancelled_ids,
            })

    def rollback_kv_connector_metadata(self, metadata: Any | None) -> None:
        """Undo pins and save bookkeeping after executor dispatch fails."""
        if metadata is None:
            return
        save_ids = {
            int(save_request.save_id)
            for save_request in getattr(metadata, 'save_requests', ())
        }
        self._release_kv_save_pins(save_ids)
        load_ids = {
            int(load_request.load_id)
            for load_request in getattr(metadata, 'load_requests', ())
        }
        self._rollback_kv_load_dispatches(load_ids)
        connector = self.kv_connector
        if connector is not None and (save_ids or load_ids):
            rolled_back = {}
            if save_ids:
                rolled_back['rolled_back_save_ids'] = save_ids
            if load_ids:
                rolled_back['rolled_back_load_ids'] = load_ids
            connector.update_connector_output(rolled_back)

    def _rollback_kv_load_dispatches(self, load_ids) -> None:
        """Release a failed RPC lease, retaining the same ready load ID."""
        for load_id in load_ids:
            pending = self._pending_kv_loads.get(int(load_id))
            if pending is None:
                continue
            pending.dispatching = False
            if pending.cancelled and not pending.dispatched:
                # The failed collective may have submitted on only a subset
                # of TP ranks. Keep the tombstone pin and re-dispatch this
                # idempotent load ID until one all-rank RPC succeeds.
                pending.needs_fence = True

    def _release_kv_save_pins(self, save_ids) -> None:
        pending_saves = getattr(self, '_pending_kv_saves', {})
        for save_id in save_ids:
            save_id = int(save_id)
            pending = pending_saves.pop(save_id, None)
            if pending is not None:
                self.block_manager.release_pinned_logical_blocks(pending.logical_block_ids)

    def _release_all_kv_save_pins(self) -> None:
        pending_saves = getattr(self, '_pending_kv_saves', {})
        self._release_kv_save_pins(tuple(pending_saves))

    def _release_all_kv_load_pins(self) -> None:
        pending_loads = getattr(self, '_pending_kv_loads', {})
        for load_id, pending in tuple(pending_loads.items()):
            pending_loads.pop(load_id, None)
            self.block_manager.release_pinned_logical_blocks(
                pending.logical_block_ids)
            if (len(pending.fallback_logical_block_ids) > 0
                    and not pending.fallback_restored):
                self.block_manager.release_pinned_logical_blocks(
                    pending.fallback_logical_block_ids)
        getattr(self, '_active_kv_load_by_req', {}).clear()
        getattr(self, '_remote_prefill_reservations', {}).clear()
        getattr(self, '_prefill_reservation_targets', {}).clear()

    def has_pending_kv_transfer_work(self) -> bool:
        """Return whether scheduler-owned save/load block pins remain."""
        return bool(
            getattr(self, '_pending_kv_saves', {})
            or getattr(self, '_pending_kv_loads', {})
        )

    def has_pending_kv_lookup_work(self) -> bool:
        """Return whether a scheduler-side non-blocking lookup is running."""
        connector = self.kv_connector
        if connector is None:
            return False
        checker = getattr(connector, 'has_pending_kv_lookup_work', None)
        if checker is not None:
            return bool(checker())
        connector_scheduler = getattr(connector, 'connector_scheduler', None)
        checker = getattr(connector_scheduler, 'has_pending_kv_lookup_work', None)
        return bool(checker is not None and checker())

    def has_pending_kv_connector_work(self) -> bool:
        """Compatibility union used by existing engine-loop wakeup paths."""
        return (
            self.has_pending_kv_transfer_work()
            or self.has_pending_kv_lookup_work()
        )

    def request_kv_connector_finished(self, seq: SchedulerSequence) -> None:
        """Notify the connector before a sequence's request ref is freed."""
        connector = self.kv_connector
        try:
            if connector is not None:
                block_ids = tuple(
                    int(block_id)
                    for block_id in self.block_manager.get_block_table(seq)
                )
                connector.request_finished(seq, block_ids)
        finally:
            self._release_prefill_reservation(seq)
            self._cancel_remote_kv_load(seq, make_waiting=False)
            self._kv_seq_generations.pop(int(seq.seq_id), None)

    def _ensure_runtime_state_available(self):
        """Make one state-cache slot available for an SSM runtime state.

        Runtime states and frozen checkpoints share the same state-cache pool. Scheduling a request is more important
        than keeping an old checkpoint, so unpinned checkpoints are evicted before we give up.
        """
        if not self.is_ssm:
            return True
        if self.state_manager.get_num_free_runtime() > 0:
            return True
        self.block_trie.state_checkpoints.evict(1)
        return self.state_manager.get_num_free_runtime() > 0

    def _pin_ssm_restore_if_needed(self, seq: SchedulerSequence):
        """Pin a matched SSM checkpoint before scheduler-side eviction."""
        if not self.is_ssm or not seq.prefix_cache.restore.is_selected:
            return True
        return self.block_trie.state_checkpoints.pin_restore(seq)

    def _rollback_unscheduled_prefix_match(
        self,
        seq: SchedulerSequence,
        stats_snapshot=None,
        *,
        connector_preempted: bool = True,
        baseline: _PrefixMatchBaseline | None = None,
    ):
        """Drop a tentative prefix match that will not be used now.

        ``block_trie.match()`` mutates sequence state immediately: it advances
        the history step, appends shared blocks, and may pin a restore node.
        If later eviction or state allocation fails, undo those side effects so
        the waiting sequence can be scheduled cleanly in a later round.
        """
        self.block_trie.stats.restore(stats_snapshot)
        if baseline is not None and not self.is_ssm:
            if seq.num_blocks < baseline.num_blocks:
                raise RuntimeError(
                    'tentative prefix match removed sequence-owned baseline blocks')
            if seq.num_blocks > baseline.num_blocks:
                self.block_manager.truncate(seq, baseline.num_blocks)
            seq.set_step(baseline.step)
            seq.kv_token_limit = baseline.kv_token_limit
            prefix_cache = seq.prefix_cache
            prefix_cache.trie_cursor = baseline.trie_cursor
            prefix_cache.match_start_step = baseline.match_start_step
            overlap = prefix_cache.recompute_overlap
            overlap.fresh_block_range = baseline.fresh_block_range
            overlap.trie_block_map.clear()
            overlap.trie_block_map.update(baseline.trie_block_map)
            seq.cached_tokens = baseline.cached_tokens
            return
        if self.is_ssm:
            self.block_trie.state_checkpoints.unpin_restore(seq)
        if seq.num_blocks > 0 or seq.logical_state >= 0:
            if connector_preempted:
                self.mark_kv_connector_preempted(seq)
            seq.state.free()
        elif seq.num_history_ids > 0:
            seq.set_step(0)
        seq.kv_token_limit = None
        prefix_cache = seq.prefix_cache
        prefix_cache.trie_cursor = None
        prefix_cache.restore.clear()
        prefix_cache.match_start_step = -1
        prefix_cache.recompute_overlap.clear_tracking()
        seq.cached_tokens = 0

    @staticmethod
    def _finalize_prefix_cache_match(seq: SchedulerSequence):
        """Publish accepted cached-token count within the current prompt."""
        match_start = seq.prefix_cache.match_start_step
        if match_start < 0:
            seq.cached_tokens = 0
            return
        cached_start = match_start
        cached_end = seq.num_history_ids
        prompt_start = seq.input_start_pos
        prompt_end = seq.input_end_pos
        seq.cached_tokens = max(0, min(cached_end, prompt_end) - max(cached_start, prompt_start))

    @staticmethod
    def _finish_prefix_cache_schedule(seq: SchedulerSequence):
        """Publish match side effects after the sequence is accepted to run."""
        prefix_cache = seq.prefix_cache
        if prefix_cache.suppress_match_stats:
            seq.cached_tokens = 0
            prefix_cache.suppress_match_stats = False
            return
        Scheduler._finalize_prefix_cache_match(seq)

    def _long_context_chunk_limit(self, seq: SchedulerSequence):
        """Return the token budget for one long-context chunk."""
        return get_long_context_chunk_limit(seq, self.cache_config.max_prefill_token_num)

    def _next_long_context_chunk_end(self, seq: SchedulerSequence, max_prefill_num: int | None = None):
        """Return the exclusive absolute token end for the next chunk."""
        if max_prefill_num is None:
            max_prefill_num = self._long_context_chunk_limit(seq)
        plan = plan_long_context_chunk(seq, max_prefill_num, include_multimodals=False)
        return plan.chunk_end

    def _prefill_kv_token_limit(self, seq: SchedulerSequence):
        """Limit KV allocation for a non-final long-context prefill chunk."""
        max_prefill_num = self._long_context_chunk_limit(seq)
        if seq.num_token_ids <= max_prefill_num:
            return None
        return self._next_long_context_chunk_end(seq, max_prefill_num)

    def _prefill_admission_token_count(self, seq: SchedulerSequence):
        """Return token budget cost for the next prefill or chunk."""
        kv_token_limit = self._prefill_kv_token_limit(seq)
        if kv_token_limit is None:
            return seq.num_token_ids
        return max(0, kv_token_limit - seq.num_history_ids)

    def has_waiting_long_prefill(self):
        """Whether a waiting request would need a non-final prefill chunk."""
        return any(self._prefill_kv_token_limit(seq) is not None for seq in self.waiting)

    def _prepare_prefill_allocation(self, seq: SchedulerSequence, prealloc_size: int):
        """Apply chunk KV limit and return the effective prealloc size."""
        kv_token_limit = self._prefill_kv_token_limit(seq)
        if kv_token_limit is None:
            seq.kv_token_limit = None
            return prealloc_size

        seq.kv_token_limit = kv_token_limit
        return 0

    def reserve_long_context_chunk(self,
                                   seq: SchedulerSequence,
                                   chunk_size: int,
                                   prealloc_size: int = 0,
                                   is_last_chunk: bool = False):
        """Reserve KV blocks for the next chunk of a running long prefill."""
        old_kv_token_limit = seq.kv_token_limit
        if is_last_chunk:
            seq.kv_token_limit = None
        else:
            seq.kv_token_limit = seq.num_history_ids + chunk_size
            prealloc_size = 0

        evictable = self._exclude_remote_prefill_victims(
            self.hanging + self.waiting)
        reserved_blocks = self._remote_prefill_reserved_blocks(
            exclude_req_id=int(seq.seq_id))
        eviction_prealloc_size = (
            prealloc_size + reserved_blocks * seq.block_size)
        if not self.eviction_helper.evict_for_seq(
                seq, evictable, eviction_prealloc_size):
            seq.kv_token_limit = old_kv_token_limit
            return False

        self.block_manager.allocate(seq, prealloc_size)
        self.block_trie.allocate(seq)
        return True

    @staticmethod
    def create_status_list_property(status: MessageStatus):
        """Create status list property."""

        def _get_status_list(self):
            seq_map = self.seq_manager.get_sequences(status)
            return list(seq_map.values())

        return property(_get_status_list)

    @staticmethod
    def create_num_status_method(status: MessageStatus):
        """Create num status method."""

        def _num_status(self):
            return self.seq_manager.num_sequences(status)

        return _num_status

    @staticmethod
    def create_has_status_method(status: MessageStatus):
        """Create has status method."""

        def _has_status(self):
            return self.seq_manager.num_sequences(status) > 0

        return _has_status

    # status list properties
    waiting = create_status_list_property(MessageStatus.WAITING)
    remote_loading = create_status_list_property(MessageStatus.WAITING_FOR_REMOTE_KVS)
    ready = create_status_list_property(MessageStatus.READY)
    hanging = create_status_list_property(MessageStatus.STOPPED)
    running = create_status_list_property(MessageStatus.RUNNING)
    migration_waiting = create_status_list_property(MessageStatus.MIGRATION_WAITING)
    migration_done = create_status_list_property(MessageStatus.MIGRATION_DONE)

    # num status methods
    num_waiting = create_num_status_method(MessageStatus.WAITING)
    num_remote_loading = create_num_status_method(MessageStatus.WAITING_FOR_REMOTE_KVS)
    num_ready = create_num_status_method(MessageStatus.READY)
    num_running = create_num_status_method(MessageStatus.RUNNING)
    num_migration_waiting = create_num_status_method(MessageStatus.MIGRATION_WAITING)
    num_migration_done = create_num_status_method(MessageStatus.MIGRATION_DONE)

    # has status methods
    has_waiting = create_has_status_method(MessageStatus.WAITING)
    has_remote_loading = create_has_status_method(MessageStatus.WAITING_FOR_REMOTE_KVS)
    has_ready = create_has_status_method(MessageStatus.READY)
    has_migration_waiting = create_has_status_method(MessageStatus.MIGRATION_WAITING)
    has_migration_done = create_has_status_method(MessageStatus.MIGRATION_DONE)

    def add_session(self, session_id: int):
        """Add new session.

        Args:
            session_id (int): New session id.
        """
        assert session_id not in self.sessions
        session = SchedulerSession(session_id, seq_manager=self.seq_manager, scheduler=self)
        self.sessions[session_id] = session
        return session

    def _schedule_migration(self):
        migration_ready: SeqList = []
        migrating_token_count = 0

        def _to_running(seq: SchedulerSequence):
            """Activate a migrated sequence and count its tokens."""
            seq.state.activate()
            migration_ready.append(seq)
            nonlocal migrating_token_count
            migrating_token_count += seq.num_token_ids

        def __evict_for_seq(seq: SchedulerSequence, waiting):
            """Evict until can append."""
            from itertools import chain

            hanging = reversed(self.hanging)
            waiting = reversed(waiting)
            evictable = self._exclude_remote_prefill_victims(
                list(chain(hanging, waiting)))
            reserved_blocks = self._remote_prefill_reserved_blocks(
                exclude_req_id=int(seq.seq_id))
            return self.eviction_helper.evict_for_seq(
                seq,
                evictable,
                reserved_blocks * seq.block_size,
            )

        def _reorder_migrating():
            """Reorder waiting."""
            return sorted(self.migration_waiting, key=lambda seq: seq.arrive_time)

        migration_waiting = _reorder_migrating()

        max_batches = (
            self.scheduler_config.max_batches
            - self.num_ready()
            - self.num_running()
            - self.num_remote_loading()
        )
        while len(migration_waiting) > 0 and len(migration_ready) < max_batches:
            seq = migration_waiting.pop(0)
            self.block_trie.match(seq)
            if not __evict_for_seq(seq, migration_waiting):
                break

            # allocate session memory
            self.block_manager.allocate(seq)
            self._finish_prefix_cache_schedule(seq)
            _to_running(seq)

        return migration_ready

    @record_function('schedule_prefill')
    def _schedule_prefill(self,
                          prealloc_size: int = 0,
                          allow_long_prefill: bool = True,
                          prefer_long_prefill: bool = False):
        """Schedule for prefilling."""

        max_batches = (
            self.scheduler_config.max_batches
            - self.num_ready()
            - self.num_running()
            - self.num_remote_loading()
        )
        swap_out_map: MapType = dict()
        swap_in_map: MapType = dict()
        copy_map: MapType = dict()
        running: SeqList = []
        admitted_slots = 0
        token_count = 0

        def _to_running(seq: SchedulerSequence, prefill_token_count: int):
            """Activate an admitted sequence and count its prefill tokens."""
            nonlocal admitted_slots, token_count
            seq.state.activate()
            running.append(seq)
            admitted_slots += 1
            token_count += prefill_token_count

        num_waiting = self.seq_manager.num_sequences(MessageStatus.WAITING)
        if (admitted_slots >= max_batches or num_waiting == 0):
            return running, swap_in_map, swap_out_map, copy_map

        waiting = _PrefillReorderer(self).reorder(self.waiting,
                                                 allow_long_prefill=allow_long_prefill,
                                                 prefer_long_prefill=prefer_long_prefill)
        skipped_waiting: SeqList = []
        while len(waiting) > 0 and admitted_slots < max_batches:
            seq = waiting.pop(0)
            evictable_waiting = skipped_waiting + waiting
            admission = _PrefillAdmissionAttempt(
                self,
                seq,
                evictable_waiting=evictable_waiting,
                prealloc_size=prealloc_size,
                token_count=token_count,
                has_admitted=len(running) > 0,
                allow_long_prefill=allow_long_prefill,
            ).run()

            if admission.remote_loading:
                # Async loads retain a model-runner request slot even though
                # this scheduler call emits no forward for them yet.
                admitted_slots += 1
                seq.record_event(EventType.SCHEDULED)
                continue
            if admission.should_skip:
                skipped_waiting.append(seq)
                continue
            if admission.should_stop:
                break

            _to_running(seq, admission.prefill_token_count)

            seq.record_event(EventType.SCHEDULED)

            if seq.kv_token_limit is not None:
                break

        return running, swap_in_map, swap_out_map, copy_map

    @record_function('schedule_decoding')
    def _schedule_decoding(self, prealloc_size: int = 0):
        """Schedule decoding."""

        def _reorder_running():
            """Reorder running."""
            return sorted(self.ready, key=lambda seq: seq.arrive_time)

        running = _reorder_running()
        assert len(running) != 0
        self._release_completed_prefill_reservations(running)

        eviction_helper = self.eviction_helper
        swap_out_map: MapType = dict()
        swap_in_map: MapType = dict()
        copy_map: MapType = dict()

        def __evict_for_seq(seq: SchedulerSequence, num_required_blocks: int):
            """Evict until can append."""
            if num_required_blocks == 0:
                # This request does not spend reserved headroom.
                return True
            reserved_blocks = self._remote_prefill_reserved_blocks(
                exclude_req_id=int(seq.seq_id))
            total_required_blocks = num_required_blocks + reserved_blocks
            if total_required_blocks <= self.block_manager.get_num_free_gpu_blocks():
                # Enough free blocks, just return True.
                return True

            from itertools import chain
            hanging = reversed(self.hanging)
            waiting = reversed(self.waiting)
            evictable = self._exclude_remote_prefill_victims(
                list(chain(hanging, waiting)))
            eviction_prealloc_size = (
                prealloc_size + reserved_blocks * seq.block_size)
            return eviction_helper.evict_for_seq(
                seq, evictable, eviction_prealloc_size)

        # 1. running
        while len(running) > 0:
            # token + n
            seq = running.pop(0)
            num_required_blocks = self.block_manager.num_required_blocks(seq, prealloc_size)
            assert seq.num_blocks + num_required_blocks <= self.block_manager.num_gpu_blocks, (
                'Sequence requires more blocks than total gpu blocks.')

            while not __evict_for_seq(seq, num_required_blocks):
                if len(running) == 0:
                    break
                seq_preempted = running.pop(-1)
                self.mark_kv_connector_preempted(seq_preempted)
                seq_preempted.state.evict()

            reserved_blocks = self._remote_prefill_reserved_blocks(
                exclude_req_id=int(seq.seq_id))
            if self.block_manager.get_num_free_gpu_blocks() < (
                    num_required_blocks + reserved_blocks):
                self.mark_kv_connector_preempted(seq)
                seq.state.evict()
                continue

            self.block_manager.allocate(seq, prealloc_size)
            self.block_trie.allocate(seq)

        return self.ready[:self.scheduler_config.max_batches], swap_in_map, swap_out_map, copy_map

    def schedule(self,
                 is_prefill: bool,
                 prealloc_size: int = 0,
                 allow_long_prefill: bool = True,
                 prefer_long_prefill: bool = False):
        """Schedule inputs for next steps."""
        if is_prefill:
            output = self._schedule_prefill(prealloc_size, allow_long_prefill, prefer_long_prefill)
        else:
            output = self._schedule_decoding(prealloc_size)
        running, swap_in_map, swap_out_map, copy_map = output

        return SchedulerOutput(running=running, swap_in_map=swap_in_map, swap_out_map=swap_out_map, copy_map=copy_map)

    @record_function('schedule_running')
    def schedule_running(self, running: SeqList, num_required_tokens: int = 1, prealloc_size: int = 1):
        """Schedule running sequences.

        This function is used to add blocks for running sequences request would be marked as invalid if not enough
        blocks can be allocated.
        """
        assert len(running) > 0
        eviction_helper = self.eviction_helper
        self._release_completed_prefill_reservations(running)

        valid_mask = [True for _ in running]

        # loop over reverse running
        rev_running = reversed(running)
        for idx, seq in enumerate(rev_running):
            if not seq.status == MessageStatus.RUNNING:
                valid_mask[idx] = False
                continue
            num_required_blocks = self.block_manager.num_required_blocks(seq, num_required_tokens)
            if num_required_blocks == 0:
                continue

            evictable = self._exclude_remote_prefill_victims(
                self.hanging + self.waiting)
            reserved_blocks = self._remote_prefill_reserved_blocks(
                exclude_req_id=int(seq.seq_id))
            eviction_prealloc_size = (
                prealloc_size + reserved_blocks * seq.block_size)
            if eviction_helper.evict_for_seq(
                    seq, evictable, eviction_prealloc_size):
                self.block_manager.allocate(seq, prealloc_size)
                self.block_trie.allocate(seq)
                continue

            # running to ready
            seq.state.deactivate()
            # ready to waiting
            self.mark_kv_connector_preempted(seq)
            seq.state.evict()
            valid_mask[idx] = False
        valid_mask = list(reversed(valid_mask))
        return valid_mask

    def stop_session(self, session_id: int):
        """Stop session.

        Args:
            session_id (int): The session id.
        """
        assert session_id in self.sessions
        session = self.sessions[session_id]
        for seq in session.sequences.values():
            connector = self.kv_connector
            cancel_lookup = getattr(connector, 'cancel_lookup', None)
            if cancel_lookup is not None:
                cancel_lookup(int(seq.seq_id))
            self._release_prefill_reservation(seq)
            seq.state.stop()

    def end_session(self, session_id: int):
        """End session.

        Args:
            session_id (int): The session id.
        """
        if self.seq_meta.sampling_strategy is not None:
            self.seq_meta.sampling_strategy.on_session_end(session_id)
        session = self.sessions[session_id]
        seqs = list(session.sequences.values())
        for seq in seqs:
            # stop session so it won't get scheduled again
            seq.state.stop()
            session.remove_sequence(seq)
        self.sessions.pop(session_id)

    def has_unfinished(self):
        """Check if there are any unfinished message."""
        return (
            self.has_ready()
            or self.has_waiting()
            or self.has_remote_loading()
            or self.has_migration_done()
        )

    def get_block_tables(self, seqs: SeqList):
        """Get block tables for the sequences."""
        return [self.block_manager.get_block_table(seq) for seq in seqs]

    def resolve_gpu_block_offsets(self, logical_block_ids):
        """Resolve paging-owned logical ids for a forward cache-copy plan."""
        return self.block_manager.resolve_gpu_block_offsets(logical_block_ids)

    def evict_seqs(self, running: SeqList):
        """Evict running sequences."""
        for seq in running:
            self.mark_kv_connector_preempted(seq)
            seq.state.evict()

    def activate_seqs(self, running: SeqList, filter_status: MessageStatus = MessageStatus.READY):
        """Lock running sequence."""
        for seq in running:
            if seq.status == filter_status:
                seq.state.activate()

    def deactivate_seqs(self, running: SeqList, filter_status: MessageStatus = MessageStatus.RUNNING):
        for seq in running:
            if seq.status == filter_status:
                seq.state.deactivate()

    @contextmanager
    def seqs_activation(self, running: SeqList):
        """Context manager to activate and deactivate sequences."""
        self.activate_seqs(running, MessageStatus.READY)
        try:
            yield running
        finally:
            self.deactivate_seqs(running, MessageStatus.RUNNING)

    def activate_migration_seqs(self, running: SeqList):
        """Lock running sequence."""
        return self.activate_seqs(running, filter_status=MessageStatus.MIGRATION_READY)

    def deactivate_migration_seqs(self, running: SeqList):
        """Unlock running migration."""
        return self.deactivate_seqs(running, filter_status=MessageStatus.MIGRATION_RUNNING)

    @contextmanager
    def seqs_migration_activation(self, running: SeqList):
        """Context manager to activate and deactivate sequences."""
        self.activate_migration_seqs(running)
        try:
            yield running
        finally:
            self.deactivate_migration_seqs(running)

    def collect_migration_done(self):
        for seq in self.migration_done:
            seq.state.activate()

    @property
    def schedule_metrics(self):
        total_blocks = self.block_manager.num_gpu_blocks
        free_blocks = self.block_manager.get_num_free_gpu_blocks()
        cache_usage = 1.0 - free_blocks / total_blocks if total_blocks else 0.0
        return ScheduleMetrics(
            active_seqs=self.num_running(),
            waiting_seqs=(
                self.num_waiting()
                + self.num_remote_loading()
                + self.num_ready()
            ),
            cache_usage=cache_usage,
            prefix_cache_hit_rate=self.block_trie.stats.hit_rate(),
            scheduler_tick=self.scheduler_tick,
        )
