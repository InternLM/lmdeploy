# Copyright (c) OpenMMLab. All rights reserved.
"""Prefill ordering and prefix-cache admission.

Each candidate is checked against long-context and token-budget policy before acquiring KV or runtime-state resources.
Local trie matches remain tentative until those resources are available; rejected attempts restore the sequence's exact
pre-match state. External loads use a later rollback boundary once a worker may have written block-aligned destinations.
"""

import enum
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch.profiler import record_function

from lmdeploy.messages import EventType
from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.long_context import get_long_context_chunk_limit, plan_long_context_chunk
from lmdeploy.utils import get_logger

from ..config import CacheConfig, SchedulerConfig
from ..messages import SchedulerSequence
from .block_trie import BlockTrie
from .kv_load_coordinator import KVLoadAdmission, KVLoadCoordinator
from .state_manager import StateManager

if TYPE_CHECKING:
    from .block_manager.base_block_manager import BaseBlockManager
    from .eviction_helper.base_eviction_helper import BaseEvictionHelper

logger = get_logger('lmdeploy')

SeqList = list[SchedulerSequence]


@dataclass(frozen=True)
class _PrefillReorderInfo:
    """Immutable pre-admission metadata used only for waiting-list ordering."""

    prefill_token_count: int
    is_nonfinal_long_prefill: bool
    estimated_long_chunks: int


class _PrefillTurnPolicy(enum.Enum):
    """Complete internal policy derived from the public long-prefill flags."""

    STANDARD = (True, False)
    SHORT_ONLY = (False, False)
    LONG_FIRST = (True, True)
    # Preserve the fourth public flag combination: try a long-looking request
    # first, but admit it only if prefix matching makes this its final prefill.
    LONG_FIRST_IF_FINAL = (False, True)

    def __init__(self, allows_nonfinal_long_prefill: bool,
                 prefers_long_prefill: bool):
        self.allows_nonfinal_long_prefill = allows_nonfinal_long_prefill
        self.prefers_long_prefill = prefers_long_prefill

    @classmethod
    def from_flags(cls, allow_long_prefill: bool, prefer_long_prefill: bool):
        """Normalize the stable public flag pair at the scheduler boundary."""
        return cls((allow_long_prefill, prefer_long_prefill))


class _PrefillReorderer:
    """Order waiting prefills without applying scheduler side effects."""

    def __init__(self, prefill_scheduler: '_PrefillScheduler'):
        self.prefill_scheduler = prefill_scheduler
        self._info_cache: dict[int, _PrefillReorderInfo] = {}

    def reorder(self,
                waiting: SeqList,
                turn_policy: _PrefillTurnPolicy):
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
        if turn_policy.prefers_long_prefill:
            # Long-work turns choose one long waiter first. The size policy only
            # reorders this long lane; it is not global shortest-prefill-first
            # admission.
            long_turn_order = self._reorder_for_long_turn(waiting)
            if long_turn_order is not None:
                return remote_ready + self._warn_if_not_permutation(waiting, long_turn_order)

        if turn_policy.allows_nonfinal_long_prefill:
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
    """Outcome from trying to admit one waiting prefill request."""

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
    def load_started(cls):
        return cls(action=_PrefillAdmissionAction.LOAD_STARTED)


@dataclass(frozen=True, slots=True)
class _PrefixMatchStateSnapshot:
    """Committed state restored when a tentative local match is rejected.

    Load failure after worker writes uses its block-aligned fallback instead.
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


class _PrefixMatchPhase(enum.Enum):
    """Lifecycle of one request-local tentative prefix transaction."""

    IDLE = enum.auto()
    TRACKING = enum.auto()
    MATCHED = enum.auto()


class _TentativePrefixMatch:
    """Manage one request's tentative ``BlockTrie.match`` side effects."""

    __slots__ = (
        'seq',
        'block_trie',
        'block_manager',
        'is_ssm',
        '_preserve_existing_state',
        '_stats_snapshot',
        '_state_snapshot',
        '_phase',
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
        self._phase = _PrefixMatchPhase.IDLE

    @property
    def is_matched(self) -> bool:
        return self._phase is _PrefixMatchPhase.MATCHED

    def begin(self) -> None:
        """Start the transaction before gates can mutate exact external state.

        Ordinary admission starts lazily from ``match``. External admission
        starts before gates so rollback can restore existing request state even
        when a private partial block prevents another trie match.
        """
        if self._phase is not _PrefixMatchPhase.IDLE or not self.block_trie.enabled:
            return
        self._stats_snapshot = self.block_trie.stats.snapshot()
        if self._preserve_existing_state:
            self._state_snapshot = _PrefixMatchStateSnapshot.capture(self.seq)
        self._phase = _PrefixMatchPhase.TRACKING

    def match(self) -> None:
        """Apply one tentative match after capturing its rollback boundary."""
        assert not self.is_matched
        self.begin()
        assert self._phase is _PrefixMatchPhase.TRACKING
        self.block_trie.match(self.seq)
        self._phase = _PrefixMatchPhase.MATCHED

    def pin_restore(self) -> bool:
        """Pin an SSM restore selected by this tentative match."""
        restore = self.seq.prefix_cache.restore
        if not self.is_ssm or not restore.is_selected:
            return True
        return self.block_trie.state_checkpoints.pin_restore(self.seq)

    def commit(self) -> None:
        """Accept the match and discard request-local rollback state."""
        self._clear()

    def rollback(self, reason: str) -> None:
        """Undo the tentative prefix-cache transaction."""
        if self._phase is _PrefixMatchPhase.IDLE:
            return

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
        self._phase = _PrefixMatchPhase.IDLE


class _PrefillAdmissionAttempt:
    """Own one waiting sequence's tentative admission side effects."""

    def __init__(self,
                 prefill_scheduler: '_PrefillScheduler',
                 seq: SchedulerSequence,
                 stopped: SeqList,
                 evictable_waiting: SeqList,
                 prealloc_size: int,
                 batch_prefill_tokens: int,
                 batch_has_prefill: bool,
                 turn_policy: _PrefillTurnPolicy):
        self.prefill_scheduler = prefill_scheduler
        self.seq = seq
        self.stopped = stopped
        self.evictable_waiting = evictable_waiting
        self.prealloc_size = prealloc_size
        self.batch_prefill_tokens = batch_prefill_tokens
        self.batch_has_prefill = batch_has_prefill
        self.load_coordinator = prefill_scheduler.load_coordinator
        self._load_ready = self.load_coordinator.is_remote_ready(seq)
        self.turn_policy = turn_policy
        self._effective_prealloc_size = prealloc_size
        self._prefix_match = _TentativePrefixMatch(
            seq,
            prefill_scheduler.block_trie,
            prefill_scheduler.block_manager,
            is_ssm=prefill_scheduler.is_ssm,
            preserve_existing_state=(
                self.load_coordinator.lookup_enabled and not self._load_ready),
        )
        self._resource_rollback_rejection: _PrefillAdmissionResult | None = None

    def run(self):
        """Apply policy, acquire resources, and commit one admission."""
        if self.load_coordinator.lookup_enabled and not self._load_ready:
            if self.load_coordinator.is_lookup_pending(self.seq):
                self.prefill_scheduler.last_schedule_had_pending_lookup = True
                return _PrefillAdmissionResult.skip()
            self._prefix_match.begin()

        gate_result = self._apply_prefill_admission_gates()
        if gate_result is not None:
            return gate_result

        resource_result = self._admit_resources()
        if resource_result is not None:
            return resource_result

        return self._allocate_and_commit()

    def _admit_resources(self):
        if self.prefill_scheduler.block_trie.enabled:
            return self._admit_prefix_cache_resources()
        if self.load_coordinator.lookup_enabled:
            load_result = self._try_external_load()
            if load_result is not None:
                return load_result
        if not self._prepare_and_evict():
            return _PrefillAdmissionResult.stop()
        return None

    def _admit_prefix_cache_resources(self):
        """Resolve the prefix source, then admit its paging resources."""
        load_result = self._resolve_prefix_source()
        if load_result is not None:
            return load_result
        return self._admit_matched_resources()

    def _resolve_prefix_source(self):
        """Match local cache first, then try loading its remote extension."""
        if not self._prefix_match.is_matched:
            # A completed external load has already published the accepted
            # prefix. Matching again would lose its cached-token accounting.
            if not self._load_ready and not self._has_private_local_tail():
                self._prefix_match.match()

        if self.load_coordinator.lookup_enabled:
            return self._try_external_load()
        return None

    def _admit_matched_resources(self):
        """Pin matched state, evict for KV, and reserve runtime state."""
        prefill = self.prefill_scheduler
        seq = self.seq
        had_ssm_restore = prefill.is_ssm and seq.prefix_cache.restore.is_selected
        if not self._prefix_match.pin_restore():
            gate_rejection = self._rollback_match_after_resource_failure(
                'failed to pin SSM restore checkpoint')
            if gate_rejection is not None:
                return gate_rejection

        kv_result = self._admit_kv_resources(had_ssm_restore)
        if kv_result is not None:
            return kv_result
        return self._admit_runtime_state()

    def _admit_kv_resources(self, had_ssm_restore: bool):
        """Evict for KV, retrying once without a pinned SSM restore."""
        if self._prepare_and_evict():
            return None

        reason = 'eviction failed'
        if had_ssm_restore:
            reason = 'eviction failed with pinned SSM restore'
        gate_rejection = self._rollback_match_after_resource_failure(reason)
        if gate_rejection is not None:
            return gate_rejection

        # The matched restore may pin the only checkpoint state that eviction
        # can free. Retrying after rollback preserves the unmatched fallback.
        if had_ssm_restore and self._prepare_and_evict():
            return None
        return _PrefillAdmissionResult.stop()

    def _admit_runtime_state(self):
        """Ensure an SSM runtime slot, retrying after match rollback."""
        prefill = self.prefill_scheduler
        seq = self.seq
        if not prefill.is_ssm or prefill._make_runtime_state_available():
            return None

        gate_rejection = self._rollback_match_after_resource_failure(
            'no runtime SSM state available')
        if gate_rejection is not None:
            return gate_rejection
        if not self._prepare_and_evict():
            return _PrefillAdmissionResult.stop()
        if not prefill._make_runtime_state_available():
            seq.kv_token_limit = None
            return _PrefillAdmissionResult.stop()
        return None

    def _try_external_load(self):
        """Map connector/paging admission to prefill queue policy."""
        prefill = self.prefill_scheduler
        if self._load_ready:
            return None
        admission = self.load_coordinator.try_load(
            self.seq,
            prealloc_size=self.prealloc_size,
            evictable_seqs=self._evictable_sequences(),
            eviction_helper=prefill.eviction_helper,
        )
        if admission is KVLoadAdmission.NO_LOAD:
            return None
        if admission is KVLoadAdmission.PENDING:
            self._prefix_match.rollback('external lookup pending')
            prefill.last_schedule_had_pending_lookup = True
            return _PrefillAdmissionResult.skip()
        if admission is KVLoadAdmission.STARTED:
            self._prefix_match.commit()
            return _PrefillAdmissionResult.load_started()
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
        yield from reversed(self.stopped)
        yield from reversed(self.evictable_waiting)

    def _try_match_prefix_for_prefill_gate(self) -> bool:
        """Tentatively match once so a request can be rechecked by a gate."""
        prefill = self.prefill_scheduler
        if self._load_ready:
            return False
        if not prefill.block_trie.enabled:
            return False
        if self._has_private_local_tail():
            return False
        self._prefix_match.match()
        return True

    def _accept_gate_enabling_match(self, rejection: _PrefillAdmissionResult) -> None:
        """Preserve a gate's outcome if resource admission loses its match."""
        assert self._prefix_match.is_matched
        self._resource_rollback_rejection = rejection

    def _rollback_match_after_resource_failure(
        self,
        reason: str,
    ) -> _PrefillAdmissionResult | None:
        """Undo the match and recover the rejection it allowed us to defer."""
        self._prefix_match.rollback(reason)
        return self._resource_rollback_rejection

    def _has_private_local_tail(self) -> bool:
        """Whether a private partial block prevents another trie match.

        A block beyond ``history // block_size`` occupies the logical range
        that another shared match would append, misaligning table positions.
        Block-granular external load must instead reuse that private boundary
        block as its first destination.
        """
        if not self.load_coordinator.lookup_enabled:
            return False
        seq = self.seq
        return seq.num_blocks > int(seq.num_history_ids) // seq.block_size

    def _token_budget_rejection(self):
        if self.turn_policy.allows_nonfinal_long_prefill:
            return _PrefillAdmissionResult.stop()
        return _PrefillAdmissionResult.skip()

    def _apply_prefill_admission_gates(self):
        """Apply long-prefill and token-budget admission gates."""
        result = self._apply_nonfinal_long_prefill_gate()
        if result is not None:
            return result
        return self._apply_prefill_token_budget_gate()

    def _apply_nonfinal_long_prefill_gate(self):
        """Reject non-final long prefills when this turn excludes them."""
        prefill = self.prefill_scheduler
        seq = self.seq
        if (self.turn_policy.allows_nonfinal_long_prefill
                or prefill._prefill_kv_token_limit(seq) is None):
            return None

        if not self._try_match_prefix_for_prefill_gate():
            return _PrefillAdmissionResult.skip()
        if prefill._prefill_kv_token_limit(seq) is not None:
            self._prefix_match.rollback(
                'still non-final long prefill on short turn')
            return _PrefillAdmissionResult.skip()
        self._accept_gate_enabling_match(
            _PrefillAdmissionResult.skip())
        return None

    def _apply_prefill_token_budget_gate(self):
        """Reject a second prefill that exceeds this turn's token budget."""
        prefill = self.prefill_scheduler
        seq = self.seq
        prefill_token_count = prefill._prefill_admission_token_count(seq)
        token_budget = prefill.cache_config.max_prefill_token_num
        if (not self.batch_has_prefill
                or self.batch_prefill_tokens + prefill_token_count
                <= token_budget):
            return None

        rejection = self._token_budget_rejection()
        if self._prefix_match.is_matched:
            self._prefix_match.rollback('still exceeds prefill token budget')
            return rejection

        if not self._try_match_prefix_for_prefill_gate():
            return rejection

        prefill_token_count = prefill._prefill_admission_token_count(seq)
        if self.batch_prefill_tokens + prefill_token_count > token_budget:
            self._prefix_match.rollback('still exceeds prefill token budget')
            return rejection

        self._accept_gate_enabling_match(rejection)
        return None

    def _prepare_and_evict(self):
        """Apply chunk allocation limits and evict for this prefill."""
        prefill = self.prefill_scheduler
        seq = self.seq
        alloc_size = prefill._prepare_prefill_allocation(seq, self.prealloc_size)
        self._effective_prealloc_size = alloc_size
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

    def _allocate_and_commit(self):
        prefill = self.prefill_scheduler
        seq = self.seq
        # Prefix-cache matching can advance the sequence step and shrink the
        # remaining prefill tail. Charge the admitted batch with the
        # post-match/post-rollback cost, not the conservative pre-match
        # estimate used to decide whether this sequence is worth trying.
        prefill_token_count = prefill._prefill_admission_token_count(seq)
        prefill.block_manager.allocate(seq, self._effective_prealloc_size)
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
        if self._load_ready:
            # Preserve the load record through the remaining prefill so its
            # reservation can be released only after model output advances the
            # sequence to input_end_pos.
            self.load_coordinator.mark_prefill_scheduled(seq)
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

    def _make_runtime_state_available(self):
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
        stopped: SeqList,
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

        evictable = stopped + waiting
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
        stopped: SeqList,
        num_ready: int,
        num_running: int,
        turn_policy: _PrefillTurnPolicy,
        prealloc_size: int = 0,
    ):
        """Select and activate one prefill batch."""
        self.last_schedule_had_pending_lookup = False
        max_batches = self.scheduler_config.max_batches - num_ready - num_running
        running: SeqList = []
        batch_prefill_tokens = 0

        if max_batches <= 0 or not waiting:
            return running

        waiting = _PrefillReorderer(self).reorder(
            waiting,
            turn_policy=turn_policy,
        )
        skipped_waiting: SeqList = []
        while waiting and len(running) < max_batches:
            seq = waiting.pop(0)
            evictable_waiting = skipped_waiting + waiting
            admission = _PrefillAdmissionAttempt(
                self,
                seq,
                stopped=stopped,
                evictable_waiting=evictable_waiting,
                prealloc_size=prealloc_size,
                batch_prefill_tokens=batch_prefill_tokens,
                batch_has_prefill=bool(running),
                turn_policy=turn_policy,
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
            seq.state.activate()
            running.append(seq)
            batch_prefill_tokens += admission.prefill_token_count
            seq.record_event(EventType.SCHEDULED)

            if seq.kv_token_limit is not None:
                break

        return running
