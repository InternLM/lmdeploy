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

* ``block_trie.match(seq)`` may find a ready checkpoint and record
  ``seq.prefix_cache.restore_state`` before the request owns a runtime state.
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

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

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


_PREFILL_GATE_SKIP = 'skip'
_PREFILL_GATE_BREAK = 'break'


@dataclass
class _PrefixMatchForPrefillGate:
    """Tentative prefix match kept only because it passes a prefill gate."""

    stats_snapshot: object
    prefill_token_count: int
    is_nonfinal_long_prefill: bool


@dataclass
class _PrefillGateCheck:
    """Result of prefill-gate checks before final resource admission."""

    prefix_match: _PrefixMatchForPrefillGate | None = None
    rollback_action: str | None = None
    reject_action: str | None = None


@dataclass(frozen=True)
class _PrefillReorderInfo:
    """Immutable pre-admission metadata used only for waiting-list ordering."""

    prefill_token_count: int
    is_nonfinal_long_prefill: bool
    estimated_long_chunks: int


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
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.sessions: dict[int, SchedulerSession] = OrderedDict()

        # For Disaggregation
        self.locked_sessions: dict[int, SchedulerSession] = OrderedDict()

        self.state_manager = build_state_manager(self.cache_config)
        self.block_manager = build_block_manager(cache_config)
        self.block_trie = BlockTrie(self.cache_config, self.block_manager, self.state_manager)
        self.is_ssm = len(self.cache_config.states_shapes) > 0

        self.eviction_helper = build_eviction_helper(self, self.scheduler_config.eviction_type)

        seq_meta = seq_meta or SequenceMeta(self.cache_config.block_size)
        self.seq_meta = seq_meta
        self.seq_manager = SequenceManager(seq_meta)
        self.scheduler_tick = 0
        self._long_prefill_policy = _envs.opt_ttft_policy
        self._long_prefill_aging_seconds_per_chunk = max(0.001, _envs.opt_ttft_aging_sec)

    def tick(self):
        """Mark one scheduler progress step (once per forward dispatch)."""
        self.scheduler_tick += 1

    def _ensure_runtime_state_available(self):
        """Make one state-cache slot available for an SSM runtime state.

        Runtime states and frozen checkpoints share the same state-cache pool. Scheduling a request is more important
        than keeping an old checkpoint, so unpinned checkpoints are evicted before we give up.
        """
        if not self.is_ssm:
            return True
        if self.state_manager.get_num_free_runtime() > 0:
            return True
        self.block_trie.evict_state_checkpoints(1)
        return self.state_manager.get_num_free_runtime() > 0

    def _acquire_ssm_restore_if_needed(self, seq: SchedulerSequence):
        """Pin a matched SSM checkpoint before scheduler-side eviction."""
        if not self.is_ssm or seq.prefix_cache.restore_state < 0:
            return True
        return self.block_trie.acquire_state_checkpoint_restore_for_seq(seq)

    def _rollback_unscheduled_prefix_match(self, seq: SchedulerSequence, stats_snapshot=None):
        """Drop a tentative prefix match that will not be used now.

        ``block_trie.match()`` mutates sequence state immediately: it advances
        the history step, appends shared blocks, and may pin a restore node.
        If later eviction or state allocation fails, undo those side effects so
        the waiting sequence can be scheduled cleanly in a later round.
        """
        self.block_trie.restore_stats(stats_snapshot)
        if self.is_ssm:
            self.block_trie.release_state_checkpoint_restore_for_seq(seq)
        if seq.num_blocks > 0 or seq.logical_state >= 0:
            seq.state.free()
        elif seq.num_history_ids > 0:
            seq.set_step(0)
        seq.kv_token_limit = None
        prefix_cache = seq.prefix_cache
        prefix_cache.last_shared_node = None
        prefix_cache.restore_state = -1
        prefix_cache.restore_node = None
        prefix_cache.restore_state_acquired = False
        prefix_cache.match_start_step = -1
        prefix_cache.private_recompute_start_step = -1
        prefix_cache.private_recompute_end_step = -1
        seq.cached_tokens = 0

    def _rollback_prefix_match_for_prefill_gate(self, seq: SchedulerSequence, stats_snapshot, reason: str):
        """Rollback a prefix match tried only to re-check prefill gates."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Rollback tentative prefix-cache gate match: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} reason={reason} num_history_ids={seq.num_history_ids} '
                         f'restore_state={seq.prefix_cache.restore_state}')
        self._rollback_unscheduled_prefix_match(seq, stats_snapshot)

    def _try_prefix_match_for_prefill_gate(
        self,
        seq: SchedulerSequence,
        accept_match: Callable[[_PrefixMatchForPrefillGate], bool],
        rollback_reason: str,
    ):
        """Tentatively match prefix cache before rejecting a prefill candidate.

        This helper is intentionally limited to pre-admission gates.  It does not evict, allocate, acquire SSM restore
        state, or publish cache state. The caller either continues into the normal admission path with the returned
        match, or the helper rolls every match side effect back.
        """
        if not self.block_trie.enable:
            return None

        stats_snapshot = self.block_trie.snapshot_stats()
        self.block_trie.match(seq)

        prefix_match = _PrefixMatchForPrefillGate(
            stats_snapshot=stats_snapshot,
            prefill_token_count=self._prefill_admission_token_count(seq),
            is_nonfinal_long_prefill=self._prefill_kv_token_limit(seq) is not None,
        )
        if accept_match(prefix_match):
            return prefix_match

        self._rollback_prefix_match_for_prefill_gate(seq, stats_snapshot, rollback_reason)
        return None

    def _check_prefill_admission_gates(self, seq: SchedulerSequence, token_count: int, has_admitted: bool,
                                       allow_long_prefill: bool):
        """Check prefill policy gates before resource admission.

        A prefix-cache hit can shrink a request enough to pass a short-turn or
        token-budget gate.  When that happens, the returned prefix match is
        still tentative; if later resource admission rolls it back, the caller
        must reject this candidate with ``rollback_action`` for the current
        scheduler turn.
        """
        prefill_token_count = self._prefill_admission_token_count(seq)
        is_nonfinal_long_prefill = self._prefill_kv_token_limit(seq) is not None
        prefix_match = None
        rollback_action = None

        if is_nonfinal_long_prefill and not allow_long_prefill:
            prefix_match = self._try_prefix_match_for_prefill_gate(
                seq,
                accept_match=lambda match: not match.is_nonfinal_long_prefill,
                rollback_reason='still non-final long prefill on short turn')
            if prefix_match is None:
                return _PrefillGateCheck(reject_action=_PREFILL_GATE_SKIP)
            prefill_token_count = prefix_match.prefill_token_count
            rollback_action = _PREFILL_GATE_SKIP

        exceeds_token_budget = (has_admitted
                                and token_count + prefill_token_count > self.cache_config.max_prefill_token_num)
        if exceeds_token_budget:
            if prefix_match is None:
                prefix_match = self._try_prefix_match_for_prefill_gate(
                    seq,
                    accept_match=lambda match: token_count +
                    match.prefill_token_count <= self.cache_config.max_prefill_token_num,
                    rollback_reason='still exceeds prefill token budget')
                if prefix_match is not None:
                    prefill_token_count = prefix_match.prefill_token_count
                    rollback_action = _PREFILL_GATE_SKIP if not allow_long_prefill else _PREFILL_GATE_BREAK

            still_exceeds_token_budget = token_count + prefill_token_count > self.cache_config.max_prefill_token_num
            if prefix_match is None or still_exceeds_token_budget:
                if prefix_match is not None:
                    self._rollback_prefix_match_for_prefill_gate(seq, prefix_match.stats_snapshot,
                                                                 'still exceeds prefill token budget')
                reject_action = _PREFILL_GATE_SKIP if not allow_long_prefill else _PREFILL_GATE_BREAK
                return _PrefillGateCheck(reject_action=reject_action)

        return _PrefillGateCheck(prefix_match=prefix_match,
                                 rollback_action=rollback_action)

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

        evictable = self.hanging + self.waiting
        if not self.eviction_helper.evict_for_seq(seq, evictable, prealloc_size):
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
    ready = create_status_list_property(MessageStatus.READY)
    hanging = create_status_list_property(MessageStatus.STOPPED)
    running = create_status_list_property(MessageStatus.RUNNING)
    migration_waiting = create_status_list_property(MessageStatus.MIGRATION_WAITING)
    migration_done = create_status_list_property(MessageStatus.MIGRATION_DONE)

    # num status methods
    num_waiting = create_num_status_method(MessageStatus.WAITING)
    num_ready = create_num_status_method(MessageStatus.READY)
    num_running = create_num_status_method(MessageStatus.RUNNING)
    num_migration_waiting = create_num_status_method(MessageStatus.MIGRATION_WAITING)
    num_migration_done = create_num_status_method(MessageStatus.MIGRATION_DONE)

    # has status methods
    has_waiting = create_has_status_method(MessageStatus.WAITING)
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
            """To running."""
            seq.state.activate()
            migration_ready.append(seq)
            nonlocal migrating_token_count
            migrating_token_count += seq.num_token_ids

        def __evict_for_seq(seq: SchedulerSequence, waiting):
            """Evict until can append."""
            from itertools import chain

            hanging = reversed(self.hanging)
            waiting = reversed(waiting)
            evictable = list(chain(hanging, waiting))
            return self.eviction_helper.evict_for_seq(seq, evictable, 0)

        def _reorder_migrating():
            """Reorder waiting."""
            return sorted(self.migration_waiting, key=lambda seq: seq.arrive_time)

        migration_waiting = _reorder_migrating()

        max_batches = self.scheduler_config.max_batches - self.num_ready() - self.num_running()
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

        max_batches = self.scheduler_config.max_batches - self.num_ready() - self.num_running()
        eviction_helper = self.eviction_helper
        swap_out_map: MapType = dict()
        swap_in_map: MapType = dict()
        copy_map: MapType = dict()
        running: SeqList = []
        token_count = 0

        def _to_running(seq: SchedulerSequence, prefill_token_count: int):
            """To running."""
            seq.state.activate()
            running.append(seq)
            nonlocal token_count
            token_count += prefill_token_count

        def __evict_for_seq(seq: SchedulerSequence, waiting, evict_prealloc_size: int):
            """Evict until can append."""
            from itertools import chain
            hanging = reversed(self.hanging)
            waiting = reversed(waiting)
            evictable = list(chain(hanging, waiting))
            return eviction_helper.evict_for_seq(seq, evictable, evict_prealloc_size)

        def __prepare_and_evict(seq: SchedulerSequence, waiting):
            """Apply chunk allocation limits and evict for this prefill."""
            alloc_prealloc_size = self._prepare_prefill_allocation(seq, prealloc_size)
            if __evict_for_seq(seq, waiting, alloc_prealloc_size):
                return True, alloc_prealloc_size
            seq.kv_token_limit = None
            return False, alloc_prealloc_size

        reorder_info_cache: dict[int, _PrefillReorderInfo] = {}

        def _get_prefill_reorder_info(seq: SchedulerSequence):
            """Return reorder-only info before prefix-cache side effects.

            Prefix-cache match/rollback mutates the remaining prompt.  Keep this cache confined to waiting-list ordering
            and recompute fresh values in the admission path below.
            """
            seq_key = id(seq)
            info = reorder_info_cache.get(seq_key)
            if info is not None:
                return info

            chunk_limit = self._long_context_chunk_limit(seq)
            if seq.num_token_ids <= chunk_limit:
                info = _PrefillReorderInfo(prefill_token_count=seq.num_token_ids,
                                           is_nonfinal_long_prefill=False,
                                           estimated_long_chunks=1)
            else:
                kv_token_limit = self._next_long_context_chunk_end(seq, chunk_limit)
                safe_chunk_limit = max(1, chunk_limit)
                info = _PrefillReorderInfo(
                    prefill_token_count=max(0, kv_token_limit - seq.num_history_ids),
                    is_nonfinal_long_prefill=True,
                    estimated_long_chunks=max(1, (seq.num_token_ids + safe_chunk_limit - 1) // safe_chunk_limit),
                )
            reorder_info_cache[seq_key] = info
            return info

        def _long_prefill_priority_key_for_reorder(seq: SchedulerSequence, now: float):
            """Prefer smaller long prompts, with age credit to avoid
            starvation."""
            info = _get_prefill_reorder_info(seq)
            wait_age = max(0.0, now - seq.arrive_time)
            age_credit = int(wait_age // self._long_prefill_aging_seconds_per_chunk)
            age_adjusted_chunks = info.estimated_long_chunks - age_credit
            return age_adjusted_chunks, info.estimated_long_chunks, seq.arrive_time

        def _split_waiting_by_prefill_kind(waiting: SeqList):
            """Split waiting requests into normal/final and non-final long
            prefill."""
            normal_waiting: SeqList = []
            long_waiting: SeqList = []
            for seq in waiting:
                if _get_prefill_reorder_info(seq).is_nonfinal_long_prefill:
                    long_waiting.append(seq)
                else:
                    normal_waiting.append(seq)
            return normal_waiting, long_waiting

        def _sort_normal_prefills(waiting: SeqList):
            return sorted(waiting, key=lambda seq: (_get_prefill_reorder_info(seq).prefill_token_count,
                                                   seq.arrive_time))

        def _sort_long_prefills_for_long_turn(waiting: SeqList):
            if self._long_prefill_policy != 'size':
                return waiting
            now = time.perf_counter()
            return sorted(waiting, key=lambda seq: _long_prefill_priority_key_for_reorder(seq, now))

        def _reorder_waiting_for_long_turn(waiting: SeqList):
            """Choose one long waiter, then fill the turn with normal
            prefills."""
            normal_waiting, long_waiting = _split_waiting_by_prefill_kind(waiting)
            if len(long_waiting) == 0:
                return None

            long_waiting = _sort_long_prefills_for_long_turn(long_waiting)
            normal_waiting = _sort_normal_prefills(normal_waiting)
            return [long_waiting[0]] + normal_waiting + long_waiting[1:]

        def _reorder_waiting_for_short_turn(waiting: SeqList):
            """Prioritize normal/final prefills while preserving long
            waiters."""
            normal_waiting, long_waiting = _split_waiting_by_prefill_kind(waiting)
            return _sort_normal_prefills(normal_waiting) + long_waiting

        def _reorder_waiting():
            """Reorder waiting."""
            waiting = sorted(self.waiting, key=lambda seq: seq.arrive_time)
            if prefer_long_prefill:
                # Long-work turns choose one long waiter first. The size policy
                # only reorders this long lane; it is not global
                # shortest-prefill-first admission.
                long_turn_waiting = _reorder_waiting_for_long_turn(waiting)
                if long_turn_waiting is not None:
                    return long_turn_waiting

            if allow_long_prefill:
                return waiting

            return _reorder_waiting_for_short_turn(waiting)

        num_waiting = self.seq_manager.num_sequences(MessageStatus.WAITING)
        if (len(running) >= max_batches or num_waiting == 0):
            return running, swap_in_map, swap_out_map, copy_map

        waiting = _reorder_waiting()
        skipped_waiting: SeqList = []
        while len(waiting) > 0 and len(running) < max_batches:
            seq = waiting.pop(0)
            gate_check = self._check_prefill_admission_gates(seq,
                                                             token_count=token_count,
                                                             has_admitted=len(running) > 0,
                                                             allow_long_prefill=allow_long_prefill)

            def __reject_after_prefill_gate_match_rollback():
                """Reject if resource admission rolled back a gate-only hit."""
                if gate_check.prefix_match is None:
                    return False
                if gate_check.rollback_action == _PREFILL_GATE_SKIP:
                    skipped_waiting.append(seq)
                    return True
                return False

            if gate_check.reject_action is not None:
                if gate_check.reject_action == _PREFILL_GATE_SKIP:
                    skipped_waiting.append(seq)
                    continue
                break

            evictable_waiting = skipped_waiting + waiting

            if self.block_trie.enable:
                if gate_check.prefix_match is None:
                    stats_snapshot = self.block_trie.snapshot_stats()
                else:
                    stats_snapshot = gate_check.prefix_match.stats_snapshot

                def __rollback_prefix_match(reason: str):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'Rollback tentative prefix-cache match: session_id={seq.session_id} '
                                     f'seq_id={seq.seq_id} reason={reason} num_history_ids={seq.num_history_ids} '
                                     f'restore_state={seq.prefix_cache.restore_state}')
                    self._rollback_unscheduled_prefix_match(seq, stats_snapshot)

                if gate_check.prefix_match is None:
                    self.block_trie.match(seq)

                had_ssm_restore = self.is_ssm and seq.prefix_cache.restore_state >= 0
                if not self._acquire_ssm_restore_if_needed(seq):
                    __rollback_prefix_match('failed to acquire SSM restore checkpoint')
                    if gate_check.prefix_match is not None:
                        if __reject_after_prefill_gate_match_rollback():
                            continue
                        break

                evicted, alloc_prealloc_size = __prepare_and_evict(seq, evictable_waiting)
                if not evicted:
                    if not had_ssm_restore:
                        __rollback_prefix_match('eviction failed')
                        if __reject_after_prefill_gate_match_rollback():
                            continue
                        break
                    # A matched SSM restore may be pinning the only checkpoint
                    # state that eviction would otherwise free.  Roll it back once
                    # and retry eviction before declaring the sequence unschedulable.
                    __rollback_prefix_match('eviction failed with pinned SSM restore')
                    if __reject_after_prefill_gate_match_rollback():
                        continue
                    if gate_check.prefix_match is not None:
                        break
                    evicted, alloc_prealloc_size = __prepare_and_evict(seq, evictable_waiting)
                    if not evicted:
                        break

                # allocate session memory
                if self.is_ssm and not self._ensure_runtime_state_available():
                    __rollback_prefix_match('no runtime SSM state available')
                    if __reject_after_prefill_gate_match_rollback():
                        continue
                    if gate_check.prefix_match is not None:
                        break
                    evicted, alloc_prealloc_size = __prepare_and_evict(seq, evictable_waiting)
                    if not evicted:
                        break
                    if not self._ensure_runtime_state_available():
                        seq.kv_token_limit = None
                        break
            else:
                evicted, alloc_prealloc_size = __prepare_and_evict(seq, evictable_waiting)
                if not evicted:
                    break
            # Prefix-cache matching can advance the sequence step and shrink
            # the remaining prefill tail.  Charge the admitted batch with the
            # post-match/post-rollback cost, not the conservative pre-match
            # estimate used to decide whether this sequence is worth trying.
            prefill_token_count = self._prefill_admission_token_count(seq)
            self.block_manager.allocate(seq, alloc_prealloc_size)
            if self.block_trie.enable:
                self.block_trie.allocate(seq)
            if self.is_ssm:
                self.state_manager.allocate(seq)
            if self.block_trie.enable:
                self._finish_prefix_cache_schedule(seq)
            _to_running(seq, prefill_token_count)

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

        eviction_helper = self.eviction_helper
        swap_out_map: MapType = dict()
        swap_in_map: MapType = dict()
        copy_map: MapType = dict()

        def __evict_for_seq(seq: SchedulerSequence, num_required_blocks: int):
            """Evict until can append."""
            if num_required_blocks == 0:
                # No need to evict, just return True.
                return True
            elif num_required_blocks < self.block_manager.get_num_free_gpu_blocks():
                # Enough free blocks, just return True.
                return True

            from itertools import chain
            hanging = reversed(self.hanging)
            waiting = reversed(self.waiting)
            evictable = list(chain(hanging, waiting))
            return eviction_helper.evict_for_seq(seq, evictable, prealloc_size)

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
                seq_preempted.state.evict()

            if self.block_manager.get_num_free_gpu_blocks() < num_required_blocks:
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

            if eviction_helper.evict_for_seq(seq, self.hanging + self.waiting, prealloc_size):
                self.block_manager.allocate(seq, prealloc_size)
                self.block_trie.allocate(seq)
                continue

            # running to ready
            seq.state.deactivate()
            # ready to waiting
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
        return self.has_ready() or self.has_waiting() or self.has_migration_done()

    def get_block_tables(self, seqs: SeqList):
        """Get block tables for the sequences."""
        return [self.block_manager.get_block_table(seq) for seq in seqs]

    def evict_seqs(self, running: SeqList):
        """Evict running sequences."""
        for seq in running:
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
        return ScheduleMetrics(
            active_seqs=self.num_running(),
            waiting_seqs=self.num_waiting() + self.num_ready(),
            total_blocks=self.block_manager.num_gpu_blocks,
            free_blocks=self.block_manager.get_num_free_gpu_blocks(),
            prefix_cache_hit_rate=self.block_trie.hit_rate(),
            scheduler_tick=self.scheduler_tick,
        )
