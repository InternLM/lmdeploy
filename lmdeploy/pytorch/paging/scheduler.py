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

External KV scheduling detail:

* External lookup is enabled only for a KV consumer with a connector, and is
  kept separate from the SSM checkpoint path. Local ``BlockTrie.match()`` runs
  first so the connector searches only beyond KV already resident on this node.
* Lookup is asynchronous. A pending result must leave the request schedulable
  for a later tick without retaining a tentative local match, so multi-turn
  sequence state is snapshotted and restored exactly.
* A positive hit is block-aligned, allocated, and handed to
  ``KVLoadCoordinator``. While workers may write those blocks, the sequence is
  in ``WAITING_FOR_REMOTE_KVS`` and paging cleanup is deferred.
* A successful load is published into the local trie and prioritized for its
  remaining prefill. A failed or cancelled load returns to the last safe
  block-aligned prefix because partially written destinations are untrusted.
* Prefill saves take a physical block snapshot for workers and a logical block
  lease for paging. ``KVSaveCoordinator`` keeps those blocks alive until every
  TP rank reports terminal progress or worker queues are drained.
"""

import enum
import time
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
from .kv_load_coordinator import KVLoadAdmission, KVLoadCoordinator
from .kv_save_coordinator import KVSaveCoordinator
from .state_manager import StateManager, build_state_manager

if TYPE_CHECKING:
    from lmdeploy.pytorch.kv_connector.base import KVConnectorBase

    from .block_manager.base_block_manager import BaseBlockManager
    from .eviction_helper.base_eviction_helper import BaseEvictionHelper

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
    # Absolute post-forward token boundary for each request. The connector
    # rounds it down to full blocks and saves only the suffix not saved before.
    connector_token_lens: tuple[int, ...] = ()
    # Current physical GPU block table for each request. Workers use these IDs
    # to locate KV tensors, but paging may reuse them after ownership is lost.
    connector_block_ids: tuple[tuple[int, ...], ...] = ()
    # Logical paging IDs corresponding to connector_block_ids. Save leases pin
    # these stable ownership handles until all TP ranks finish asynchronous I/O,
    # preventing their physical blocks from being reassigned too early.
    connector_logical_block_ids: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class _PrefillReorderInfo:
    """Immutable pre-admission metadata used only for waiting-list ordering."""

    prefill_token_count: int
    is_nonfinal_long_prefill: bool
    estimated_long_chunks: int


class _PrefillReorderer:
    """Order waiting prefills without applying scheduler side effects."""

    def __init__(self, prefill_scheduler: '_PrefillScheduler'):
        self.prefill_scheduler = prefill_scheduler
        self._info_cache: dict[int, _PrefillReorderInfo] = {}

    def reorder(self,
                waiting: SeqList,
                allow_long_prefill: bool,
                prefer_long_prefill: bool):
        """Return waiting requests in the order the prefill loop should try."""
        waiting = sorted(waiting, key=lambda seq: seq.arrive_time)
        # A completed load already owns destination blocks and a soft prefill
        # reservation. Admit it first to shorten that ownership window.
        remote_ready = [
            seq for seq in waiting
            if self.prefill_scheduler.load_coordinator.is_remote_ready(seq)
        ]
        waiting = [
            seq for seq in waiting
            if not self.prefill_scheduler.load_coordinator.is_remote_ready(seq)
        ]
        if prefer_long_prefill:
            # Long-work turns choose one long waiter first. The size policy only
            # reorders this long lane; it is not global shortest-prefill-first
            # admission.
            long_turn_order = self._reorder_for_long_turn(waiting)
            if long_turn_order is not None:
                return remote_ready + self._warn_if_not_permutation(waiting, long_turn_order)

        if allow_long_prefill:
            return remote_ready + self._warn_if_not_permutation(waiting, waiting)

        reordered = self._reorder_for_short_turn(waiting)
        return remote_ready + self._warn_if_not_permutation(waiting, reordered)

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

        prefill = self.prefill_scheduler
        chunk_limit = prefill._long_context_chunk_limit(seq)
        if seq.num_token_ids <= chunk_limit:
            info = _PrefillReorderInfo(prefill_token_count=seq.num_token_ids,
                                       is_nonfinal_long_prefill=False,
                                       estimated_long_chunks=1)
        else:
            kv_token_limit = prefill._next_long_context_chunk_end(seq, chunk_limit)
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
        prefill = self.prefill_scheduler
        info = self._get_reorder_info(seq)
        wait_age = max(0.0, now - seq.arrive_time)
        age_credit = int(wait_age // prefill._long_prefill_aging_seconds_per_chunk)
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
        prefill = self.prefill_scheduler
        if prefill._long_prefill_policy != 'size':
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


class _PrefillAdmissionAction(enum.Enum):
    ADMIT = enum.auto()
    SKIP = enum.auto()
    STOP = enum.auto()
    LOAD_STARTED = enum.auto()


@dataclass(frozen=True)
class _PrefillAdmissionResult:
    """Outcome from trying to admit one waiting prefill request.

    The outer loop distinguishes four outcomes:

    * ``ADMIT``: include the request in this tick's model batch.
    * ``SKIP``: leave it waiting but continue trying later candidates.
    * ``STOP``: resource pressure ends this prefill admission turn.
    * ``LOAD_STARTED``: no model work was selected, but the request left the
      waiting queue for asynchronous KV load.
    """

    action: _PrefillAdmissionAction
    prefill_token_count: int = 0

    @classmethod
    def admit(cls, prefill_token_count: int):
        return cls(action=_PrefillAdmissionAction.ADMIT,
                   prefill_token_count=prefill_token_count)

    @classmethod
    def skip(cls):
        return cls(action=_PrefillAdmissionAction.SKIP)

    @classmethod
    def stop(cls):
        return cls(action=_PrefillAdmissionAction.STOP)

    @classmethod
    def load(cls):
        return cls(action=_PrefillAdmissionAction.LOAD_STARTED)


@dataclass(frozen=True, slots=True)
class _PrefixMatchStateSnapshot:
    """Exact sequence state captured before a tentative local trie match.

    External lookup itself does not mutate sequence paging state. The scheduler
    may, however, run ``block_trie.match`` first so the connector queries only
    beyond the locally resident prefix. If that non-blocking lookup returns
    pending, or a positive hit cannot be admitted before worker writes start,
    the request will not run this tick and the tentative local match must be
    undone.

    A multi-turn request may already own valid history, blocks, and model
    metadata before this attempt. Restoring this baseline preserves that exact
    committed state; the legacy new-request rollback to step zero would discard
    it. This snapshot is not used after an asynchronous load starts--load
    failure then rolls back to its block-aligned ``fallback_step`` through
    ``KVLoadCoordinator`` because workers may have partially written KV.
    """

    # Committed sequence progress and block ownership before tentative match.
    num_history_ids: int
    num_blocks: int
    # Prefix-cache cursor, public hit accounting, and temporary overlap state.
    trie_cursor: Any
    match_start_step: int
    cached_tokens: int
    # Request-local allocation limit that a multi-turn attempt may carry.
    kv_token_limit: int | None
    # Temporary recompute-overlap identities created by local trie matching.
    fresh_block_range: range | None
    trie_block_map: dict[int, int]
    # Model state that must remain aligned with the committed history step.
    model_meta: Any

    @classmethod
    def capture(cls, seq: SchedulerSequence):
        overlap = seq.prefix_cache.recompute_overlap
        return cls(
            num_history_ids=int(seq.num_history_ids),
            num_blocks=int(seq.num_blocks),
            trie_cursor=seq.prefix_cache.trie_cursor,
            match_start_step=int(seq.prefix_cache.match_start_step),
            cached_tokens=int(seq.cached_tokens),
            kv_token_limit=seq.kv_token_limit,
            fresh_block_range=overlap.fresh_block_range,
            trie_block_map=dict(overlap.trie_block_map),
            model_meta=seq.model_meta,
        )


class _TentativePrefixMatch:
    """Request-local transaction around ``BlockTrie.match`` side effects.

    Ordinary and SSM admission preserve the historical fallback to an unmatched request. External lookup instead needs
    an exact pre-match snapshot because a multi-turn request may already own committed progress. Both contracts share
    one stats snapshot, restore-pin boundary, and explicit commit/rollback lifecycle without changing their rollback
    semantics.
    """

    __slots__ = (
        'seq',
        'block_trie',
        'block_manager',
        'is_ssm',
        '_preserve_existing_state',
        '_stats_snapshot',
        '_state_snapshot',
        '_rejection_on_rollback',
        '_started',
        'matched',
    )

    def __init__(self,
                 seq: SchedulerSequence,
                 block_trie: BlockTrie,
                 block_manager,
                 *,
                 is_ssm: bool,
                 preserve_existing_state: bool):
        self.seq = seq
        self.block_trie = block_trie
        self.block_manager = block_manager
        self.is_ssm = is_ssm
        self._preserve_existing_state = preserve_existing_state
        self._stats_snapshot = None
        self._state_snapshot: _PrefixMatchStateSnapshot | None = None
        self._rejection_on_rollback: _PrefillAdmissionResult | None = None
        self._started = False
        self.matched = False

    def begin(self) -> None:
        """Start the transaction before gates can mutate exact external state.

        Ordinary admission starts lazily from ``match``. External admission
        starts before gates so rollback can restore existing request state even
        when a private partial block prevents another trie match.
        """
        if self._started or not self.block_trie.enabled:
            return
        self._stats_snapshot = self.block_trie.stats.snapshot()
        if self._preserve_existing_state:
            self._state_snapshot = _PrefixMatchStateSnapshot.capture(self.seq)
        self._started = True

    def match(self) -> None:
        """Apply one tentative match after capturing its rollback boundary."""
        assert not self.matched
        self.begin()
        self.block_trie.match(self.seq)
        self.matched = True

    def retain_for_admission(self, rejection_on_rollback: _PrefillAdmissionResult) -> None:
        """Keep a gate-enabling match and remember its original rejection."""
        assert self.matched
        self._rejection_on_rollback = rejection_on_rollback

    def pin_restore(self) -> bool:
        """Pin an SSM restore selected by this tentative match."""
        restore = self.seq.prefix_cache.restore
        if not self.is_ssm or not restore.is_selected:
            return True
        return self.block_trie.state_checkpoints.pin_restore(self.seq)

    def commit(self) -> None:
        """Accept the match and discard request-local rollback state."""
        self._clear()

    def rollback(self, reason: str):
        """Undo the transaction and return any gate-defined rejection."""
        rejection = self._rejection_on_rollback
        if not self._started:
            return rejection

        seq = self.seq
        logger.debug('Rollback tentative prefix-cache match: session_id=%s seq_id=%s reason=%s '
                     'num_history_ids=%s restore_state=%s', seq.session_id, seq.seq_id, reason, seq.num_history_ids,
                     seq.prefix_cache.restore.slot)
        self.block_trie.stats.restore(self._stats_snapshot)
        snapshot = self._state_snapshot
        if snapshot is None:
            self._reset_to_unmatched()
        else:
            self._restore_snapshot(snapshot)
        self._clear()
        return rejection

    def _restore_snapshot(self, snapshot: _PrefixMatchStateSnapshot) -> None:
        seq = self.seq
        if seq.num_blocks < snapshot.num_blocks:
            raise RuntimeError(
                'tentative prefix match removed sequence-owned baseline blocks')
        if seq.num_blocks > snapshot.num_blocks:
            self.block_manager.truncate(seq, snapshot.num_blocks)
        seq.set_step(snapshot.num_history_ids)
        seq.model_meta = snapshot.model_meta
        seq.kv_token_limit = snapshot.kv_token_limit
        prefix_cache = seq.prefix_cache
        prefix_cache.trie_cursor = snapshot.trie_cursor
        prefix_cache.match_start_step = snapshot.match_start_step
        overlap = prefix_cache.recompute_overlap
        overlap.fresh_block_range = snapshot.fresh_block_range
        overlap.trie_block_map.clear()
        overlap.trie_block_map.update(snapshot.trie_block_map)
        seq.cached_tokens = snapshot.cached_tokens

    def _reset_to_unmatched(self) -> None:
        seq = self.seq
        if self.is_ssm:
            self.block_trie.state_checkpoints.unpin_restore(seq)
        if seq.num_blocks > 0 or seq.logical_state >= 0:
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

    def _clear(self) -> None:
        self._stats_snapshot = None
        self._state_snapshot = None
        self._rejection_on_rollback = None
        self._started = False
        self.matched = False


class _PrefillAdmissionAttempt:
    """Try to admit one waiting prefill sequence.

    The attempt owns all tentative prefix-cache side effects for the sequence:
    match, SSM restore pinning, eviction, runtime-state checks, allocation, and
    rollback. The outer prefill loop still owns queue traversal and decides
    whether a rejected candidate is skipped or ends the current prefill turn.
    """

    def __init__(self,
                 prefill_scheduler: '_PrefillScheduler',
                 seq: SchedulerSequence,
                 hanging: SeqList,
                 evictable_waiting: SeqList,
                 prealloc_size: int,
                 token_count: int,
                 has_admitted: bool,
                 allow_long_prefill: bool):
        self.prefill_scheduler = prefill_scheduler
        self.seq = seq
        self.hanging = hanging
        self.evictable_waiting = evictable_waiting
        self.prealloc_size = prealloc_size
        self.token_count = token_count
        self.has_admitted = has_admitted
        self.load_coordinator = prefill_scheduler.load_coordinator
        self._remote_ready = self.load_coordinator.is_remote_ready(seq)
        self.allow_long_prefill = allow_long_prefill
        self._alloc_size = prealloc_size
        self._prefix_match = _TentativePrefixMatch(
            seq,
            prefill_scheduler.block_trie,
            prefill_scheduler.block_manager,
            is_ssm=prefill_scheduler.is_ssm,
            preserve_existing_state=(
                self.load_coordinator.lookup_enabled and not self._remote_ready),
        )

    def run(self):
        """Run the admission route for one waiting prefill.

        1. If a previous external lookup is pending, skip without applying new
           local prefix-cache side effects.
        2. Snapshot multi-turn state before a local match may become tentative.
        3. Apply long-prefill and token-budget gates.
        4. Prefer a local trie hit, then query/load only its remote extension.
        5. Admit KV/state resources or roll the tentative match back precisely.
        6. On success, allocate blocks/states and publish the accepted hit.
        """
        if self.load_coordinator.lookup_enabled and not self._remote_ready:
            if self._lookup_is_pending():
                return _PrefillAdmissionResult.skip()
            self._prefix_match.begin()

        gate_result = self._check_prefill_admission_gates()
        if gate_result is not None:
            return gate_result

        resource_result = self._admit_resources()
        if resource_result is not None:
            return resource_result

        return self._finish_admission()

    def _lookup_is_pending(self) -> bool:
        """Skip without touching local prefix state while lookup is running.

        The connector owns the Future and deduplicates polls. Marking this turn lets EngineLoop use a short I/O poll
        delay instead of diagnosing an empty batch as GPU-cache pressure.
        """
        prefill = self.prefill_scheduler
        if not self.load_coordinator.is_lookup_pending(self.seq):
            return False
        prefill.last_schedule_had_pending_lookup = True
        return True

    def _admit_resources(self):
        if self.prefill_scheduler.block_trie.enabled:
            return self._admit_prefix_cache_resources()
        if self.load_coordinator.lookup_enabled:
            lookup_result = self._query_external_prefix()
            if lookup_result is not None:
                return lookup_result
        if not self._prepare_and_evict():
            return _PrefillAdmissionResult.stop()
        return None

    def _admit_prefix_cache_resources(self):
        """Admit resources for prefix-cache scheduling.

        Route map:
        1. Use or create the tentative prefix-cache match.
        2. For external consumers, query only beyond that local match.
        3. Pin any SSM restore state required by the match.
        4. Prepare allocation limits and evict KV/state resources.
        5. For SSM, verify a runtime state slot is still available.

        Any failure rolls the tentative match back. A match created only to pass
        a prefill gate returns that gate's skip/stop result after rollback;
        normal resource failures keep their local retry/stop behavior here.
        """
        prefill = self.prefill_scheduler
        seq = self.seq
        if not self._prefix_match.matched:
            # A completed external load has already published the accepted
            # prefix interval. Matching again would restart accounting at the
            # remote step and drop the restored tokens from request metrics.
            if not self._remote_ready and not self._has_private_local_tail():
                self._prefix_match.match()

        if self.load_coordinator.lookup_enabled:
            lookup_result = self._query_external_prefix()
            if lookup_result is not None:
                return lookup_result

        had_ssm_restore = prefill.is_ssm and seq.prefix_cache.restore.is_selected
        if not self._prefix_match.pin_restore():
            result = self._prefix_match.rollback(
                'failed to pin SSM restore checkpoint')
            if result is not None:
                return result

        if not self._prepare_and_evict():
            if not had_ssm_restore:
                result = self._prefix_match.rollback('eviction failed')
                if result is not None:
                    return result
                return _PrefillAdmissionResult.stop()

            # A matched SSM restore may be pinning the only checkpoint state
            # that eviction would otherwise free. Roll it back once and retry
            # eviction before declaring the sequence unschedulable.
            result = self._prefix_match.rollback(
                'eviction failed with pinned SSM restore')
            if result is not None:
                return result
            if not self._prepare_and_evict():
                return _PrefillAdmissionResult.stop()

        if prefill.is_ssm and not prefill._ensure_runtime_state_available():
            result = self._prefix_match.rollback(
                'no runtime SSM state available')
            if result is not None:
                return result
            if not self._prepare_and_evict():
                return _PrefillAdmissionResult.stop()
            if not prefill._ensure_runtime_state_available():
                seq.kv_token_limit = None
                return _PrefillAdmissionResult.stop()

        return None

    def _query_external_prefix(self):
        """Map connector/paging admission to prefill queue policy."""
        prefill = self.prefill_scheduler
        if self._remote_ready:
            return None
        admission = self.load_coordinator.try_load(
            self.seq,
            prealloc_size=self.prealloc_size,
            evictable_seqs=self._evictable_sequences(),
        )
        if admission is KVLoadAdmission.NO_LOAD:
            return None
        if admission is KVLoadAdmission.PENDING:
            self._prefix_match.rollback('external lookup pending')
            prefill.last_schedule_had_pending_lookup = True
            return _PrefillAdmissionResult.skip()
        if admission is KVLoadAdmission.STARTED:
            self._prefix_match.commit()
            return _PrefillAdmissionResult.load()
        if admission is KVLoadAdmission.FULL_PREFILL_UNAVAILABLE:
            reason = 'full prefill capacity unavailable'
        else:
            assert admission is KVLoadAdmission.SOFT_BUDGET_UNAVAILABLE
            reason = 'soft prefill budget unavailable'
        # No worker has seen a destination on rejected admission, so the
        # request-local prefix transaction remains exactly reversible.
        self._prefix_match.rollback(reason)
        return _PrefillAdmissionResult.stop()

    def _evictable_sequences(self):
        """Iterate queue-owned eviction candidates in historical order."""
        yield from reversed(self.hanging)
        yield from reversed(self.evictable_waiting)

    def _match_prefix_for_prefill_gate(self):
        """Tentatively match once so a request can be rechecked by a gate."""
        prefill = self.prefill_scheduler
        if (self._remote_ready or not prefill.block_trie.enabled
                or self._has_private_local_tail()):
            return None
        self._prefix_match.match()
        return True

    def _has_private_local_tail(self) -> bool:
        """Whether blocks exist beyond the full-block part of local history.

        ``num_history_ids // block_size`` counts the completely computed
        blocks before the current step.  A larger ``num_blocks`` means that the
        sequence also owns the block containing a non-aligned current step, or
        blocks preallocated after it.  Those blocks are private to this
        sequence because their KV is partial or not computed yet, so they
        cannot be published as complete reusable trie blocks.

        For example, with block size 4, a chunked prefill may stop at step 5
        with block table ``[P0, P1]``::

            P0 -> tokens [0, 4), complete
            P1 -> tokens [4, 8), only the KV at token 4 is valid

        The trie cursor is at step 4 while ``P1`` already occupies logical
        block index 1.  If another ``block_trie.match`` finds a shared block
        ``S1`` for tokens [4, 8), matching appends it after ``P1`` instead of
        filling ``P1``.  The resulting table ``[P0, P1, S1]`` is misaligned:
        ``S1`` describes logical block 1 but resides at block-table index 2.

        External lookup starts at the exact local step 5, but a block-granular
        transfer rounds its start down to step 4.  It must therefore reuse
        ``P1`` at index 1 as the first destination and overwrite the incomplete
        KV there.  Skipping trie rematch keeps that destination stable until
        the load is bound.  Without this guard, lookup may start from an
        incorrectly advanced step or the load/model may address a block table
        whose logical token ranges no longer match its indices.

        This state means that a sequence retains local progress across
        scheduling attempts.  It can result from chunked prefill, repeated
        model forwards, preemption/resume, or a continued chat session; it is
        not specific to multi-turn conversation.
        """
        if not self.load_coordinator.lookup_enabled:
            return False
        seq = self.seq
        return seq.num_blocks > int(seq.num_history_ids) // seq.block_size

    def _token_budget_rejection(self):
        if self.allow_long_prefill:
            return _PrefillAdmissionResult.stop()
        return _PrefillAdmissionResult.skip()

    def _check_prefill_admission_gates(self):
        """Apply prefill gates, tentatively matching only when it may help."""
        prefill = self.prefill_scheduler
        seq = self.seq
        token_budget = prefill.cache_config.max_prefill_token_num
        prefill_token_count = prefill._prefill_admission_token_count(seq)
        is_nonfinal_long_prefill = prefill._prefill_kv_token_limit(seq) is not None

        if is_nonfinal_long_prefill and not self.allow_long_prefill:
            matched = self._match_prefix_for_prefill_gate()
            if matched is None:
                return _PrefillAdmissionResult.skip()
            if prefill._prefill_kv_token_limit(seq) is not None:
                self._prefix_match.rollback('still non-final long prefill on short turn')
                return _PrefillAdmissionResult.skip()
            self._prefix_match.retain_for_admission(
                _PrefillAdmissionResult.skip())
            prefill_token_count = prefill._prefill_admission_token_count(seq)

        exceeds_token_budget = self.has_admitted and self.token_count + prefill_token_count > token_budget
        if not exceeds_token_budget:
            return None

        if not self._prefix_match.matched:
            matched = self._match_prefix_for_prefill_gate()
            if matched is not None:
                prefill_token_count = prefill._prefill_admission_token_count(seq)
                if self.token_count + prefill_token_count <= token_budget:
                    self._prefix_match.retain_for_admission(
                        self._token_budget_rejection())
                    return None
                self._prefix_match.rollback('still exceeds prefill token budget')
        else:
            self._prefix_match.rollback('still exceeds prefill token budget')
        return self._token_budget_rejection()

    def _prepare_and_evict(self):
        """Apply chunk allocation limits and evict for this prefill."""
        prefill = self.prefill_scheduler
        seq = self.seq
        alloc_size = prefill._prepare_prefill_allocation(seq, self.prealloc_size)
        self._alloc_size = alloc_size
        if self._evict_for_seq(alloc_size):
            return True
        seq.kv_token_limit = None
        return False

    def _evict_for_seq(self, alloc_size: int):
        """Evict stopped or skipped waiters until this sequence can run."""
        prefill = self.prefill_scheduler
        return prefill.eviction_helper.evict_for_seq(
            self.seq,
            list(self._evictable_sequences()),
            alloc_size,
        )

    def _finish_admission(self):
        prefill = self.prefill_scheduler
        seq = self.seq
        # Prefix-cache matching can advance the sequence step and shrink the
        # remaining prefill tail. Charge the admitted batch with the
        # post-match/post-rollback cost, not the conservative pre-match
        # estimate used to decide whether this sequence is worth trying.
        prefill_token_count = prefill._prefill_admission_token_count(seq)
        prefill.block_manager.allocate(seq, self._alloc_size)
        if prefill.block_trie.enabled:
            prefill.block_trie.allocate(seq)
        if prefill.is_ssm:
            prefill.state_manager.allocate(seq)
        if prefill.block_trie.enabled:
            prefill.block_trie.finalize_match(seq)
        self.load_coordinator.track_prefill(
            seq,
            prealloc_size=self.prealloc_size,
        )
        if self._remote_ready:
            # Preserve the load record through the remaining prefill so its
            # reservation can be released only after model output advances the
            # sequence to input_end_pos.
            self.load_coordinator.mark_scheduled(seq)
        self._prefix_match.commit()
        return _PrefillAdmissionResult.admit(prefill_token_count)


class _PrefillScheduler:
    """Own prefill ordering, admission, and long-context reservation.

    Long-lived dependencies are the resource owners used by prefill. Queue
    contents and active-batch counts remain request-local inputs supplied by
    the public :class:`Scheduler` facade for each scheduling turn.
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        *,
        is_ssm: bool,
        block_manager: 'BaseBlockManager',
        block_trie: BlockTrie,
        state_manager: StateManager,
        eviction_helper: 'BaseEvictionHelper',
        load_coordinator: KVLoadCoordinator,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.is_ssm = is_ssm
        self.block_manager = block_manager
        self.block_trie = block_trie
        self.state_manager = state_manager
        self.eviction_helper = eviction_helper
        self.load_coordinator = load_coordinator
        self.last_schedule_had_pending_lookup = False
        self._long_prefill_policy = _envs.opt_ttft_policy
        self._long_prefill_aging_seconds_per_chunk = max(
            0.001,
            _envs.opt_ttft_aging_sec,
        )

    def _ensure_runtime_state_available(self):
        """Make one state-cache slot available for an SSM runtime state."""
        if not self.is_ssm:
            return True
        if self.state_manager.get_num_free_runtime() > 0:
            return True
        self.block_trie.state_checkpoints.evict(1)
        return self.state_manager.get_num_free_runtime() > 0

    def _long_context_chunk_limit(self, seq: SchedulerSequence):
        """Return the token budget for one long-context chunk."""
        return get_long_context_chunk_limit(
            seq,
            self.cache_config.max_prefill_token_num,
        )

    def _next_long_context_chunk_end(
        self,
        seq: SchedulerSequence,
        max_prefill_num: int | None = None,
    ):
        """Return the exclusive absolute token end for the next chunk."""
        if max_prefill_num is None:
            max_prefill_num = self._long_context_chunk_limit(seq)
        plan = plan_long_context_chunk(
            seq,
            max_prefill_num,
            include_multimodals=False,
        )
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

    def _prepare_prefill_allocation(
        self,
        seq: SchedulerSequence,
        prealloc_size: int,
    ):
        """Apply chunk KV limit and return the effective prealloc size."""
        kv_token_limit = self._prefill_kv_token_limit(seq)
        if kv_token_limit is None:
            seq.kv_token_limit = None
            return prealloc_size

        seq.kv_token_limit = kv_token_limit
        return 0

    def has_waiting_long_prefill(self, waiting: SeqList):
        """Whether a waiting request needs a non-final prefill chunk."""
        return any(
            self._prefill_kv_token_limit(seq) is not None
            for seq in waiting
        )

    def reserve_long_context_chunk(
        self,
        seq: SchedulerSequence,
        *,
        hanging: SeqList,
        waiting: SeqList,
        chunk_size: int,
        prealloc_size: int = 0,
        is_last_chunk: bool = False,
    ):
        """Reserve KV blocks for the next chunk of a running long prefill."""
        old_kv_token_limit = seq.kv_token_limit
        if is_last_chunk:
            seq.kv_token_limit = None
        else:
            seq.kv_token_limit = seq.num_history_ids + chunk_size
            prealloc_size = 0

        evictable = hanging + waiting
        if not self.eviction_helper.evict_for_seq(
            seq,
            evictable,
            prealloc_size,
        ):
            seq.kv_token_limit = old_kv_token_limit
            return False

        self.block_manager.allocate(seq, prealloc_size)
        self.block_trie.allocate(seq)
        return True

    @record_function('schedule_prefill')
    def schedule(
        self,
        *,
        waiting: SeqList,
        hanging: SeqList,
        num_ready: int,
        num_running: int,
        prealloc_size: int = 0,
        allow_long_prefill: bool = True,
        prefer_long_prefill: bool = False,
    ):
        """Select and activate one prefill batch."""
        self.last_schedule_had_pending_lookup = False
        max_batches = self.scheduler_config.max_batches - num_ready - num_running
        running: SeqList = []
        token_count = 0

        def _to_running(
            seq: SchedulerSequence,
            prefill_token_count: int,
        ):
            """Activate an admitted sequence and count its prefill tokens."""
            seq.state.activate()
            running.append(seq)
            nonlocal token_count
            token_count += prefill_token_count

        if len(running) >= max_batches or len(waiting) == 0:
            return running

        waiting = _PrefillReorderer(self).reorder(
            waiting,
            allow_long_prefill=allow_long_prefill,
            prefer_long_prefill=prefer_long_prefill,
        )
        skipped_waiting: SeqList = []
        while len(waiting) > 0 and len(running) < max_batches:
            seq = waiting.pop(0)
            evictable_waiting = skipped_waiting + waiting
            admission = _PrefillAdmissionAttempt(
                self,
                seq,
                hanging=hanging,
                evictable_waiting=evictable_waiting,
                prealloc_size=prealloc_size,
                token_count=token_count,
                has_admitted=len(running) > 0,
                allow_long_prefill=allow_long_prefill,
            ).run()

            if admission.action is _PrefillAdmissionAction.LOAD_STARTED:
                # The request left WAITING for asynchronous load without using
                # a model-batch slot or prefill token budget.
                continue
            if admission.action is _PrefillAdmissionAction.SKIP:
                skipped_waiting.append(seq)
                continue
            if admission.action is _PrefillAdmissionAction.STOP:
                break

            assert admission.action is _PrefillAdmissionAction.ADMIT
            _to_running(seq, admission.prefill_token_count)
            seq.record_event(EventType.SCHEDULED)

            if seq.kv_token_limit is not None:
                break

        return running


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
        transfer_config = cache_config.kv_transfer_config
        # A producer-only connector still needs the save path below, but must
        # not issue lookups. SSM restore owns a different state-cache protocol
        # and is deliberately excluded from external KV load admission.
        self._external_lookup_enabled = (
            kv_connector is not None
            and transfer_config is not None
            and transfer_config.is_kv_consumer
            and not self.is_ssm
        )
        checkpoint_state_manager = self.state_manager if self.is_ssm else None
        self.block_trie = BlockTrie(allocator=self.block_manager.allocator,
                                   block_size=self.cache_config.block_size,
                                   enabled=self.cache_config.enable_prefix_caching,
                                   checkpoint_state_manager=checkpoint_state_manager)

        self.eviction_helper = build_eviction_helper(self, self.scheduler_config.eviction_type)
        # Load admission receives only paging owners plus request-local queue
        # candidates from its caller; it does not reach back through Scheduler.
        self.kv_load_coordinator = KVLoadCoordinator(
            lookup_enabled=self._external_lookup_enabled,
            connector=kv_connector,
            block_manager=self.block_manager,
            block_trie=self.block_trie,
            eviction_helper=self.eviction_helper,
            sessions=self.sessions,
        )
        self._prefill_scheduler = _PrefillScheduler(
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config,
            is_ssm=self.is_ssm,
            block_manager=self.block_manager,
            block_trie=self.block_trie,
            state_manager=self.state_manager,
            eviction_helper=self.eviction_helper,
            load_coordinator=self.kv_load_coordinator,
        )
        # Keep save call sites uniform even when the producer role is disabled.
        self.kv_save_coordinator = KVSaveCoordinator(self)
        # Per-tick signal consumed by EngineLoop to distinguish asynchronous
        # lookup latency from actual cache-allocation pressure.
        self.last_schedule_had_pending_lookup = False

        seq_meta = seq_meta or SequenceMeta(self.cache_config.block_size)
        self.seq_meta = seq_meta
        self.seq_manager = SequenceManager(seq_meta)
        self.scheduler_tick = 0

    def tick(self):
        """Mark one scheduler progress step (once per forward dispatch)."""
        self.scheduler_tick += 1

    def shutdown(self) -> None:
        """Release scheduler-side connector resources exactly once.

        Engine shutdown normally drains worker queues first. Clearing scheduler ownership here makes repeated shutdown
        harmless and prevents any later scheduling path from starting new external work.
        """
        connector = self.kv_connector
        self.kv_connector = None
        self._external_lookup_enabled = False
        self.kv_load_coordinator.disable()
        self.kv_save_coordinator.clear()
        if connector is not None:
            connector.shutdown()

    def has_waiting_long_prefill(self):
        """Whether a waiting request would need a non-final prefill chunk."""
        return self._prefill_scheduler.has_waiting_long_prefill(self.waiting)

    def reserve_long_context_chunk(self,
                                   seq: SchedulerSequence,
                                   chunk_size: int,
                                   prealloc_size: int = 0,
                                   is_last_chunk: bool = False):
        """Reserve KV blocks for the next chunk of a running long prefill."""
        return self._prefill_scheduler.reserve_long_context_chunk(
            seq,
            hanging=self.hanging,
            waiting=self.waiting,
            chunk_size=chunk_size,
            prealloc_size=prealloc_size,
            is_last_chunk=is_last_chunk,
        )

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

    # Remote-loading sequences are intentionally separate from WAITING: workers
    # may address their destination blocks, so ordinary scheduling/eviction
    # must not treat them as candidates until the coordinator publishes them.
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
            self.block_trie.finalize_match(seq)
            _to_running(seq)

        return migration_ready

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
            elif num_required_blocks <= self.block_manager.get_num_free_gpu_blocks():
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
                # Preemption abandons the tracked full-prefill target. Keeping
                # it would reserve blocks for work no longer admitted.
                self.kv_load_coordinator.release(seq_preempted)
                seq_preempted.state.evict()

            if self.block_manager.get_num_free_gpu_blocks() < num_required_blocks:
                self.kv_load_coordinator.release(seq)
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
        self.last_schedule_had_pending_lookup = False
        if is_prefill:
            running = self._prefill_scheduler.schedule(
                waiting=self.waiting,
                hanging=self.hanging,
                num_ready=self.num_ready(),
                num_running=self.num_running(),
                prealloc_size=prealloc_size,
                allow_long_prefill=allow_long_prefill,
                prefer_long_prefill=prefer_long_prefill,
            )
            self.last_schedule_had_pending_lookup = (
                self._prefill_scheduler.last_schedule_had_pending_lookup)
            swap_in_map: MapType = {}
            swap_out_map: MapType = {}
            copy_map: MapType = {}
        else:
            running, swap_in_map, swap_out_map, copy_map = self._schedule_decoding(
                prealloc_size)

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
            self.kv_load_coordinator.release(seq)
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
        connector = self.kv_connector
        for seq in session.sequences.values():
            # A lookup owns no GPU destinations and can be cancelled directly.
            if connector is not None:
                connector.cancel_lookup(seq.seq_id)
            # An active load may still write GPU memory. Defer the state change
            # until all ranks terminate instead of making its blocks evictable.
            if self.kv_load_coordinator.request_stop(seq):
                continue
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
        connector = self.kv_connector
        for seq in seqs:
            if connector is not None:
                connector.cancel_lookup(seq.seq_id)
            # Session removal also frees sequence blocks, so it must be deferred
            # while a worker may still address an in-flight load destination.
            if self.kv_load_coordinator.request_end(seq):
                continue
            # stop session so it won't get scheduled again
            seq.state.stop()
            if connector is not None:
                connector.request_finished(seq)
            session.remove_sequence(seq)
        if not session.sequences:
            self.sessions.pop(session_id)

    def has_unfinished(self):
        """Whether model, migration, load, or save ownership is outstanding.

        Remote-loading requests are outside the normal waiting queue, and save leases may outlive their request. Both
        must keep the engine alive until workers stop accessing paging-owned memory.
        """
        return (
            self.has_ready()
            or self.has_waiting()
            or self.has_remote_loading()
            or self.has_migration_done()
            or self.kv_save_coordinator.has_pending()
        )

    def build_connector_meta(
        self,
        running: SeqList,
        swap_in_map: MapType | None = None,
        swap_out_map: MapType | None = None,
        connector_token_lens: tuple[int, ...] = (),
    ):
        """Build and lease one connector payload after work selection.

        This is called even when ``running`` is empty because connector-only
        executor steps submit pending transfers and poll completions. For a
        prefill save, block tables are snapshotted only after the model batch is
        fixed, then logical leases are acquired before metadata reaches workers.
        """
        connector = self.kv_connector
        if connector is None:
            return None
        if connector_token_lens:
            # Workers address the current physical cache slots, while save
            # leases pin logical block ownership across asynchronous I/O.
            logical_block_ids = tuple(
                tuple(int(block_id) for block_id in seq.logical_blocks.get_real_blocks())
                for seq in running
            )
            block_ids = tuple(
                tuple(int(block_id) for block_id in self.block_manager.get_block_table(seq))
                for seq in running
            )
        else:
            logical_block_ids = ()
            block_ids = ()
        scheduler_output = SchedulerOutput(
            running=running,
            swap_in_map=swap_in_map or {},
            swap_out_map=swap_out_map or {},
            copy_map={},
            connector_token_lens=connector_token_lens,
            connector_block_ids=block_ids,
            connector_logical_block_ids=logical_block_ids,
        )
        metadata = connector.build_connector_meta(scheduler_output)
        if metadata is not None:
            # Acquire before the caller queues metadata. Sequence cleanup may
            # otherwise release the last block reference before save starts.
            self.kv_save_coordinator.acquire(metadata)
        return metadata

    def update_connector_output(self, connector_output) -> None:
        """Convert all-TP worker progress into paging state transitions.

        The connector filters/aggregates rank-local output first. Paging then publishes or rolls back loads and releases
        only save operations known to be terminal across all ranks.
        """
        if connector_output is None or self.kv_connector is None:
            return
        result = self.kv_connector.update_connector_output(connector_output)
        self.kv_load_coordinator.update(result.load_results)
        self.kv_save_coordinator.update(result.completed_save_ids)

    def release_completed_prefill_reservations(self, seqs: SeqList) -> None:
        """Release soft targets only after forward output advanced history."""
        self.kv_load_coordinator.release_completed_prefills(seqs)

    def finish_deferred_kv_transfers_after_worker_drain(self) -> None:
        """Release paging ownership after worker transfer queues have drained.

        Engine sleep may discard prefetched completion outputs. Worker drain is
        the alternate terminal proof: deferred ended loads can be removed and
        save leases can be released even without their normal output events.
        """
        self.kv_load_coordinator.finish_deferred_loads_after_worker_drain()
        if self.kv_connector is not None:
            self.kv_connector.finish_transfers_after_worker_drain()
        self.kv_save_coordinator.clear()

    def get_block_tables(self, seqs: SeqList):
        """Get block tables for the sequences."""
        return [self.block_manager.get_block_table(seq) for seq in seqs]

    def resolve_gpu_block_offsets(self, logical_block_ids):
        """Resolve paging-owned logical ids for a forward cache-copy plan."""
        return self.block_manager.resolve_gpu_block_offsets(logical_block_ids)

    def evict_seqs(self, running: SeqList):
        """Evict running sequences."""
        for seq in running:
            self.kv_load_coordinator.release(seq)
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
            waiting_seqs=self.num_waiting() + self.num_ready() + self.num_remote_loading(),
            cache_usage=cache_usage,
            prefix_cache_hit_rate=self.block_trie.stats.hit_rate(),
            scheduler_tick=self.scheduler_tick,
        )
