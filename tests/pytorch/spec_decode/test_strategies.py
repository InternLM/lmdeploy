# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for SchedulerSequenceARSpec state management.

Tests focus on _update_token_ids_decode, especially for the speculative-decoding path where multiple tokens can be
accepted in one step, and the stop_pos parameter which truncates acceptance inline.
"""
from unittest.mock import MagicMock

import numpy as np

from lmdeploy.pytorch.messages import (
    SamplingParam,
    SequenceMeta,
)
from lmdeploy.pytorch.strategies.ar_spec.sequence import (
    ARSpecSequenceStrategy,
    SchedulerSequenceARSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seq(prefill_tokens=None):
    """Create a minimal SchedulerSequenceARSpec with an optional prefill."""
    strategy = ARSpecSequenceStrategy()
    seq_meta = SequenceMeta(block_size=16, strategy=strategy)

    session = MagicMock()
    session.seq_meta = seq_meta

    seq = SchedulerSequenceARSpec(seq_id=0, session=session)

    if prefill_tokens is not None:
        token_ids = np.array(prefill_tokens, dtype=np.int64)
        seq._update_token_ids_inputs(token_ids)

    return seq


def _cache_contents(seq):
    """Return the visible contents of history_cache as a list."""
    return seq.history_cache.get_real().tolist()


def _state(seq):
    """Return a dict snapshot of the key counters."""
    return dict(
        num_valid_ids=seq._num_valid_ids,
        num_history_ids=seq._num_history_ids,
        num_token_ids=seq._num_token_ids,
        cache_len=len(seq.history_cache),
    )


# ---------------------------------------------------------------------------
# Tests for _update_token_ids_prefill
# ---------------------------------------------------------------------------

class TestPrefill:
    """State transitions in _update_token_ids_prefill.

    Prefill always generates exactly 1 token (main-model output for the last prefill position).  stop_pos is -1
    (continue → append drafts) or 0 (stop → suppress drafts).
    """

    def test_single_token_updates_counters(self):
        """Basic prefill: counters advance correctly for 1 generated token."""
        seq = _make_seq([1, 2, 3])
        seq._update_token_ids_prefill(
            np.array([10], dtype=np.int64),
            draft_token_ids=np.array([], dtype=np.int64),
        )
        s = _state(seq)
        assert s['num_valid_ids'] == 4        # 3 input + 1 generated
        assert s['num_history_ids'] == 3      # previous num_token_ids
        assert s['num_token_ids'] == 1        # just the 1 generated token
        assert seq.num_new_tokens == 1

    def test_drafts_appended_when_no_stop(self):
        """stop_pos=-1: draft tokens are appended to cache and counted."""
        seq = _make_seq([1, 2, 3])
        drafts = np.array([100, 101], dtype=np.int64)
        seq._update_token_ids_prefill(
            np.array([10], dtype=np.int64),
            draft_token_ids=drafts,
        )
        s = _state(seq)
        assert s['num_token_ids'] == 3        # 1 generated + 2 draft
        assert s['cache_len'] == 6            # 3 input + 1 generated + 2 draft
        # draft tokens sit at the tail of the cache
        cache = _cache_contents(seq)
        assert cache[-2:] == [100, 101]

    def test_drafts_suppressed_at_stop_pos_zero(self):
        """stop_pos=0: draft tokens are NOT appended."""
        seq = _make_seq([1, 2, 3])
        seq._update_token_ids_prefill(
            np.array([10], dtype=np.int64),
            draft_token_ids=np.array([100, 101], dtype=np.int64),
            stop_pos=0,
        )
        s = _state(seq)
        assert s['num_token_ids'] == 1        # only the 1 generated token
        assert s['cache_len'] == 4            # 3 input + 1 generated, no draft

    def test_invariant_cache_len_equals_history_plus_token(self):
        """cache_len == num_history_ids + num_token_ids after prefill."""
        seq = _make_seq([1, 2, 3])
        seq._update_token_ids_prefill(
            np.array([10], dtype=np.int64),
            draft_token_ids=np.array([100, 101], dtype=np.int64),
        )
        s = _state(seq)
        assert s['cache_len'] == s['num_history_ids'] + s['num_token_ids']

    def test_num_valid_ids_excludes_draft_tokens(self):
        """Draft tokens are speculative — num_valid_ids counts only the 1
        generated token."""
        seq = _make_seq([1, 2, 3])
        seq._update_token_ids_prefill(
            np.array([10], dtype=np.int64),
            draft_token_ids=np.array([100, 101, 102], dtype=np.int64),  # 3 drafts
        )
        # num_valid_ids grows by exactly 1, not 1+3
        assert seq._num_valid_ids == 4

    def test_num_token_ids_includes_draft_tokens(self):
        """num_token_ids = 1 (generated) + len(draft_token_ids)."""
        seq = _make_seq([1, 2, 3])
        seq._update_token_ids_prefill(
            np.array([10], dtype=np.int64),
            draft_token_ids=np.array([100, 101, 102], dtype=np.int64),  # 3 drafts
        )
        assert seq._num_token_ids == 4   # 1 generated + 3 draft


# ---------------------------------------------------------------------------
# Tests for _update_token_ids_decode with spec tokens from prior step
# ---------------------------------------------------------------------------

class TestDecode:
    """Verify decode when the previous step had spec tokens."""

    def _setup_after_prefill_with_drafts(self, prefill, draft_tokens):
        """Return seq after prefill + spec draft appended."""
        seq = _make_seq(prefill[:-1])
        # Simulate the prefill step with draft tokens
        draft = np.array(draft_tokens, dtype=np.int64)
        seq._update_token_ids_prefill(np.array(prefill[-1:], dtype=np.int64), draft)
        return seq

    def test_all_spec_accepted(self):
        """All 2 spec tokens accepted; verify state."""
        # Manually set up state as if a prefill produced 3 tokens + 2 spec
        seq = self._setup_after_prefill_with_drafts([10, 20, 30], [40, 50])
        # Decode: all 2 positions verified with one bonus token; valid = [40, 50, 60]
        seq._update_token_ids_decode(np.array([40, 50, 60]), draft_token_ids=np.array([70, 80], dtype=np.int64))
        s = _state(seq)
        assert s['num_valid_ids'] == 6   # 3 + 3
        assert s['num_history_ids'] == 5  # num_valid - 1
        assert s['num_token_ids'] == 3
        assert len(seq.history_cache) == 8  # with two new draft tokens appended

    def test_partial_spec_accepted(self):
        """Only 1 of 2 spec tokens accepted (second rejected = -1)."""
        # Manually set up state as if a prefill produced 3 tokens + 2 spec
        seq = self._setup_after_prefill_with_drafts([10, 20, 30], [40, 50])
        # Decode with 1 accepted, 1 rejected: token_ids = [40, -1, -1]
        seq._update_token_ids_decode(np.array([40, -1, -1]), draft_token_ids=np.array([70, 80], dtype=np.int64))
        s = _state(seq)
        assert s['num_valid_ids'] == 4   # 3 + 1
        assert s['num_history_ids'] == 3  # 4 - 1
        assert s['num_token_ids'] == 3  # 1 new token + 2 draft
        assert s['cache_len'] == 4 + 2  # resized to valid (4) + 2 new draft tokens

    def test_all_spec_rejected(self):
        """All spec tokens rejected; only 1 bonus token accepted."""
        # Manually set up state as if a prefill produced 3 tokens + 2 spec
        seq = self._setup_after_prefill_with_drafts([10, 20, 30], [40, 50])
        # Decode with 0 accepted, 1 rejected: token_ids = [60, -1, -1]
        seq._update_token_ids_decode(np.array([60, -1, -1]), draft_token_ids=np.array([70, 80], dtype=np.int64))
        s = _state(seq)
        assert s['num_valid_ids'] == 4   # 3 + 1
        assert s['num_history_ids'] == 3
        assert s['num_token_ids'] == 3
        assert s['cache_len'] == 4 + 2  # resized to valid (4) + 2 new draft tokens

    def test_cache_after_partial_acceptance(self):
        """When prior spec ids > 0, only valid_ids[-1] enters token_ids."""
        # Manually set up state as if a prefill produced 3 tokens + 2 spec
        seq = self._setup_after_prefill_with_drafts([10, 20, 30], [40, 50])
        # Decode with 1 accepted, 1 rejected: token_ids = [40, 60, -1]
        seq._update_token_ids_decode(np.array([40, 60, -1]), draft_token_ids=np.array([70, 80], dtype=np.int64))
        cache = _cache_contents(seq)
        assert cache == [10, 20, 30, 40, 60, 70, 80]
        s = _state(seq)
        assert s['num_valid_ids'] == 5   # 3 + 2
        assert s['num_history_ids'] == 4
        assert s['num_token_ids'] == 3
        assert s['cache_len'] == 7

    def test_with_stop_pos(self):
        """When stop_pos truncates acceptance, only tokens up to stop_pos are
        accepted."""
        # Manually set up state as if a prefill produced 3 tokens + 2 spec
        seq = self._setup_after_prefill_with_drafts([10, 20, 30], [40, 50])
        # Decode with 2 accepted but stop_pos=1: token_ids = [40, 50, 60], but only [40, 50] accepted
        seq._update_token_ids_decode(
            np.array([40, 50, 60]),
            draft_token_ids=np.array([70, 80], dtype=np.int64),
            stop_pos=1,
        )
        cache = _cache_contents(seq)
        assert cache == [10, 20, 30, 40, 50]
        s = _state(seq)
        assert s['num_valid_ids'] == 5   # 3 + 2 (stop_pos=1 means only first two accepted)
        assert s['num_history_ids'] == 4
        assert s['num_token_ids'] == 1
        assert s['cache_len'] == 5


# ---------------------------------------------------------------------------
# Tests for stop_pos parameter of _update_token_ids_decode
# ---------------------------------------------------------------------------

class TestDecodeStopPos:
    """stop_pos truncates accepted tokens inline during the decode update."""

    def _seq_ready_for_decode(self, prior_valid=3, num_spec=2):
        """Return a seq with prior_valid tokens and num_spec pending spec
        tokens."""
        seq = _make_seq()
        seq._num_valid_ids = prior_valid
        seq._num_history_ids = prior_valid - 1
        base = list(range(prior_valid - 1))
        old_last = [prior_valid - 1]
        spec_tokens = list(range(100, 100 + num_spec))
        seq.history_cache.append(np.array(base + old_last + spec_tokens, dtype=np.int64))
        seq._num_token_ids = 1 + num_spec
        return seq

    def test_stop_at_last_accepted_token(self):
        """stop_pos = N-1: all tokens accepted; num_token_ids set to 1."""
        seq = self._seq_ready_for_decode(prior_valid=3, num_spec=2)
        seq._update_token_ids_decode(
            np.array([200, 201, 202]),
            draft_token_ids=np.array([], dtype=np.int64),
            stop_pos=2,  # last of 3 accepted
        )
        assert seq._num_valid_ids == 6   # 3 + 3
        assert seq._num_history_ids == 5
        assert seq._num_token_ids == 1
        assert len(seq.history_cache) == 6

    def test_stop_at_first_accepted_token(self):
        """stop_pos = 0: only the first accepted token kept."""
        seq = self._seq_ready_for_decode(prior_valid=3, num_spec=2)
        seq._update_token_ids_decode(
            np.array([200, 201, 202]),
            draft_token_ids=np.array([], dtype=np.int64),
            stop_pos=0,
        )
        assert seq._num_valid_ids == 4   # 3 + 1
        assert seq._num_history_ids == 3
        assert seq._num_token_ids == 1
        assert len(seq.history_cache) == 4

    def test_stop_at_middle_accepted_token(self):
        """stop_pos = 1 with N=3 accepted: 2 tokens kept."""
        seq = self._seq_ready_for_decode(prior_valid=3, num_spec=2)
        seq._update_token_ids_decode(
            np.array([200, 201, 202]),
            draft_token_ids=np.array([], dtype=np.int64),
            stop_pos=1,
        )
        assert seq._num_valid_ids == 5   # 3 + 2
        assert seq._num_history_ids == 4
        assert seq._num_token_ids == 1
        assert len(seq.history_cache) == 5

    def test_stop_suppresses_new_draft_tokens(self):
        """Draft tokens are not appended when stop_pos is set."""
        seq = self._seq_ready_for_decode(prior_valid=3, num_spec=2)
        seq._update_token_ids_decode(
            np.array([200, 201, 202]),
            draft_token_ids=np.array([300, 301], dtype=np.int64),
            stop_pos=1,
        )
        assert seq._num_valid_ids == 5   # 3 + 2
        assert len(seq.history_cache) == 5   # no draft appended
        assert seq._num_token_ids == 1

    def test_invariant_after_stop(self):
        """cache_len == num_history_ids + num_token_ids for all stop_pos
        values."""
        for num_accepted in (1, 2, 4):
            for pos in range(num_accepted):
                for num_new_draft in (0, 2):
                    seq = self._seq_ready_for_decode(prior_valid=5, num_spec=num_accepted - 1)
                    accepted = np.array(list(range(200, 200 + num_accepted)), dtype=np.int64)
                    drafts = np.array(list(range(300, 300 + num_new_draft)), dtype=np.int64)
                    seq._update_token_ids_decode(accepted, draft_token_ids=drafts, stop_pos=pos)
                    s = _state(seq)
                    assert s['cache_len'] == s['num_history_ids'] + s['num_token_ids'], (
                        f'Invariant broken: num_accepted={num_accepted}, pos={pos}, '
                        f'num_new_draft={num_new_draft}, state={s}'
                    )

    def test_no_spec_tokens_stop(self):
        """Regular decode (no prior spec): stop at the single new token."""
        seq = _make_seq([10, 20, 30])
        seq._update_token_ids_decode(
            np.array([99]),
            draft_token_ids=np.array([], dtype=np.int64),
            stop_pos=0,
        )
        assert seq._num_valid_ids == 4
        assert seq._num_token_ids == 1
        assert len(seq.history_cache) == 4


# ---------------------------------------------------------------------------
# Helpers for routed_experts tests
# ---------------------------------------------------------------------------

def _make_seq_with_experts(prefill_tokens=None):
    """Create a SchedulerSequenceARSpec with return_routed_experts=True."""
    strategy = ARSpecSequenceStrategy()
    seq_meta = SequenceMeta(block_size=16, strategy=strategy)
    session = MagicMock()
    session.seq_meta = seq_meta
    sampling_param = SamplingParam(return_routed_experts=True)
    seq = SchedulerSequenceARSpec(seq_id=0, session=session, sampling_param=sampling_param)
    if prefill_tokens is not None:
        seq._update_token_ids_inputs(np.array(prefill_tokens, dtype=np.int64))
    return seq


def _experts(n, k=2):
    """Return a dummy routed_experts array of shape [n, 1, k]."""
    return np.arange(n * k, dtype=np.uint16).reshape(n, 1, k)


# ---------------------------------------------------------------------------
# Tests for routed_experts in _update_token_ids_decode
# ---------------------------------------------------------------------------

class TestRoutedExpertsDecode:
    """routed_experts handling in _update_token_ids_decode."""

    def _seq_with_prior_spec(self, prior_valid=3, num_spec=2):
        """Seq with prior_valid tokens and num_spec pending spec tokens."""
        seq = _make_seq_with_experts()
        seq._num_valid_ids = prior_valid
        seq._num_history_ids = prior_valid - 1
        base = list(range(prior_valid - 1))
        seq.history_cache.append(
            np.array(base + [prior_valid - 1] + list(range(100, 100 + num_spec)), dtype=np.int64))
        seq._num_token_ids = 1 + num_spec
        return seq

    def test_experts_clipped_to_num_valid(self):
        """When 2 of 3 tokens are accepted, experts are clipped to 2."""
        seq = self._seq_with_prior_spec(prior_valid=3, num_spec=2)
        seq._update_token_ids_decode(
            np.array([30, 40, -1]),       # 2 valid
            draft_token_ids=np.array([], dtype=np.int64),
            routed_experts=_experts(3),   # 3 expert rows provided
        )
        assert len(seq.all_routed_experts) == 2

    def test_experts_all_accepted(self):
        """When all tokens are accepted, all expert rows are kept."""
        seq = self._seq_with_prior_spec(prior_valid=3, num_spec=2)
        seq._update_token_ids_decode(
            np.array([30, 40, 50]),
            draft_token_ids=np.array([], dtype=np.int64),
            routed_experts=_experts(3),
        )
        assert len(seq.all_routed_experts) == 3

    def test_experts_clipped_by_stop_pos(self):
        """stop_pos limits num_valid, which limits expert rows kept."""
        seq = self._seq_with_prior_spec(prior_valid=3, num_spec=2)
        # 3 valid tokens, but stop at pos=1 → only 2 accepted
        seq._update_token_ids_decode(
            np.array([30, 40, 50]),
            draft_token_ids=np.array([], dtype=np.int64),
            routed_experts=_experts(3),
            stop_pos=1,
        )
        assert len(seq.all_routed_experts) == 2  # stop_pos+1

    def test_experts_none_is_noop(self):
        """Passing routed_experts=None leaves all_routed_experts unchanged."""
        seq = self._seq_with_prior_spec(prior_valid=3, num_spec=2)
        before = len(seq.all_routed_experts)
        seq._update_token_ids_decode(
            np.array([30, 40, -1]),
            draft_token_ids=np.array([], dtype=np.int64),
            routed_experts=None,
        )
        assert len(seq.all_routed_experts) == before


# ---------------------------------------------------------------------------
# Tests for routed_experts in _update_token_ids_prefill
# ---------------------------------------------------------------------------

class TestRoutedExpertsPrefill:
    """routed_experts handling in _update_token_ids_prefill.

    Prefill always generates exactly 1 token (the main-model output for the last position). stop_pos is either -1
    (continue, append drafts) or 0 (stop, suppress drafts). There is no rejection sampler in prefill, so routed_experts
    are never clipped regardless of stop_pos.
    """

    def test_prefill_appends_expert(self):
        """Normal prefill (stop_pos=-1) appends the single expert row."""
        seq = _make_seq_with_experts()
        seq._update_token_ids_prefill(
            np.array([10], dtype=np.int64),          # 1 generated token
            draft_token_ids=np.array([], dtype=np.int64),
            routed_experts=_experts(1),               # 1 expert row
        )
        assert len(seq.all_routed_experts) == 1

    def test_prefill_stop_pos_zero_appends_expert(self):
        """stop_pos=0 still appends the expert row — no clipping in prefill."""
        seq = _make_seq_with_experts()
        seq._update_token_ids_prefill(
            np.array([10], dtype=np.int64),
            draft_token_ids=np.array([], dtype=np.int64),
            routed_experts=_experts(1),
            stop_pos=0,
        )
        assert len(seq.all_routed_experts) == 1  # not clipped

    def test_prefill_experts_none_is_noop(self):
        """Passing routed_experts=None leaves all_routed_experts unchanged."""
        seq = _make_seq_with_experts()
        before = len(seq.all_routed_experts)
        seq._update_token_ids_prefill(
            np.array([10], dtype=np.int64),
            draft_token_ids=np.array([], dtype=np.int64),
            routed_experts=None,
        )
        assert len(seq.all_routed_experts) == before

    def test_expert_after_evict(self):
        """set_step(0) evicts all cached experts; reprefill over all valid ids
        re-accumulates them.

        Evict only happens when the sequence is still running (not stopped), so prefill always has draft tokens and
        decode always attaches new drafts.
        """
        seq = _make_seq_with_experts([1, 2, 3])   # 3 input tokens

        # prefill: main model generates token 10, draft model produces 2 draft tokens
        # → num_valid_ids = 4, num_token_ids = 3 (1 + 2 drafts)
        seq._update_token_ids_prefill(
            np.array([10], dtype=np.int64),
            draft_token_ids=np.array([100, 101], dtype=np.int64),
            routed_experts=_experts(3),
        )
        assert len(seq.all_routed_experts) == 3

        # decode: both draft tokens accepted + 1 bonus, 2 new drafts attached
        # → num_valid_ids = 7, len(experts) = 1 + 3 = 4
        seq._update_token_ids_decode(
            np.array([100, 101, 50]),
            draft_token_ids=np.array([200, 201], dtype=np.int64),
            routed_experts=_experts(3),
        )
        assert len(seq.all_routed_experts) == 6

        num_valid = seq.num_valid_ids   # == 7

        # evict: set_step(0) clears all cached experts
        seq.set_step(0)
        assert len(seq.all_routed_experts) == 0
        assert seq.routed_experts is None

        new_routed_experts = _experts(num_valid)
        # reprefill: all num_valid tokens reprocessed, draft tokens re-attached
        seq._update_token_ids_prefill(
            np.array([60]),
            draft_token_ids=np.array([300, 301], dtype=np.int64),
            routed_experts=new_routed_experts,   # one row per valid token
        )
        assert seq.routed_experts is not None
        assert len(seq.routed_experts) == num_valid
        assert np.array_equal(seq.all_routed_experts, new_routed_experts)

    def test_set_step_keeps_transition_aligned_experts(self):
        """set_step(step) keeps routed experts aligned to step transitions."""
        seq = _make_seq_with_experts([1, 2, 3, 4, 5, 6])
        seq.append_routed_experts(_experts(6))

        seq.set_step(5)

        assert len(seq.all_routed_experts) == 5

# ---------------------------------------------------------------------------
# Tests for _update_token_ids_inputs across multiple turns
# ---------------------------------------------------------------------------


class TestMultiTurnUpdateInputs:
    """Test _update_token_ids_inputs — especially how it resets state for a new
    turn."""

    def _do_turn(self, seq, prompt, draft_tokens=None):
        """Run one full turn: prefill + decode, return seq."""
        draft = np.array(draft_tokens or [], dtype=np.int64)
        seq._update_token_ids_prefill(np.array(prompt, dtype=np.int64), draft)
        # Decode: treat all prefill outputs as valid, no spec rejection
        seq._update_token_ids_decode(np.array(prompt, dtype=np.int64),
                                     draft_token_ids=np.array([], dtype=np.int64))
        return seq

    def test_initial_inputs_sets_state(self):
        """First inputs call initialises counters correctly."""
        seq = _make_seq()
        seq._update_token_ids_inputs(np.array([1, 2, 3], dtype=np.int64))
        assert seq._num_valid_ids == 3
        assert seq._num_token_ids == 3
        assert seq.num_new_tokens == 0
        assert seq.output_start_pos == 3

    def test_consecutive_inputs_accumulate_valid_ids(self):
        """Two inputs calls grow num_valid_ids and update output_start_pos."""
        seq = _make_seq()
        seq._update_token_ids_inputs(np.array([1, 2, 3], dtype=np.int64))
        seq._update_token_ids_inputs(np.array([4, 5], dtype=np.int64))
        assert seq._num_valid_ids == 5
        assert seq._num_token_ids == 2
        assert seq.output_start_pos == 5
        assert seq.num_new_tokens == 0

    def test_inputs_after_decode_resets_new_tokens(self):
        """After a decode that generated tokens, inputs resets num_new_tokens
        to 0."""
        seq = _make_seq([1, 2, 3])
        self._do_turn(seq, prompt=[10])       # generates 2 tokens: 1 from prefill + 1 from decode
        assert seq.num_new_tokens == 2        # prefill contributes 1, decode contributes 1

        seq._update_token_ids_inputs(np.array([100, 101], dtype=np.int64))
        assert seq.num_new_tokens == 0
        assert seq._num_token_ids == 2

    def test_inputs_after_decode_updates_valid_ids_and_start_pos(self):
        """Inputs on turn 2 grows num_valid_ids by exactly len(new_tokens)."""
        seq = _make_seq([1, 2, 3])
        self._do_turn(seq, prompt=[10])
        valid_before = seq._num_valid_ids

        seq._update_token_ids_inputs(np.array([100, 101], dtype=np.int64))
        assert seq._num_valid_ids == valid_before + 2
        assert seq.output_start_pos == valid_before + 2

    def test_inputs_appends_tokens_to_cache(self):
        """New input tokens appear at the tail of history_cache."""
        seq = _make_seq([1, 2, 3])
        self._do_turn(seq, prompt=[10])
        seq._update_token_ids_inputs(np.array([100, 101], dtype=np.int64))
        cache = _cache_contents(seq)
        assert cache[-2] == 100
        assert cache[-1] == 101

    def test_generated_ids_empty_after_inputs(self):
        """generated_ids is empty immediately after inputs (no new outputs
        yet)."""
        seq = _make_seq([1, 2, 3])
        self._do_turn(seq, prompt=[10])
        seq._update_token_ids_inputs(np.array([100, 101], dtype=np.int64))
        assert len(seq.generated_ids) == 0
