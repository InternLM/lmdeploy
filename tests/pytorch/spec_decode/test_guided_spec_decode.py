# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for MTP (speculative decoding) + Guided Decoding integration.

1. _expand/_slice_sampling_inputs non-tensor field handling
   - response_formats, session_ctx must be repeated/sliced alongside tensor
     fields when spec decode expands or slices the batch dimension.
   - Boundary cases: num_tokens=1, empty response_formats, None session_ctx.

2. Grammar state management (fork / rollback / accept_string)
   - fork() produces an independent GrammarMatcher snapshot.
   - accept_string() advances state; rollback(n) reverts it.
   - Fork-based strategy: fork before draft generation, accept final tokens
     on original matcher only.

3. Positional-serial grammar mask
   - Prove that different spec positions need different grammar masks.

4. Draft model grammar masking
   - Masked argmax picks valid tokens; unmasked may pick invalid.

5. Grammar state after rejection sampling
   - After rejection, grammar state must reflect exactly the accepted tokens.

NOTE: With BPE tokenizers (like Qwen), accept_token() may not visibly
advance grammar state because individual BPE tokens can be multi-character
partial bytes. Tests that need to observe state transitions use
accept_string() which reliably advances the grammar character-by-character.
"""
import pytest
import torch
import xgrammar as xgr

from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.spec_decode.spec_agent import (
    _expand_sampling_inputs,
    _slice_sampling_inputs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QWEN_MODEL = 'Qwen/Qwen3.5-0.8B'


@pytest.fixture(scope='module')
def tokenizer_info():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(_QWEN_MODEL, trust_remote_code=True)
    return xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size)


@pytest.fixture(scope='module')
def compiler(tokenizer_info):
    return xgr.GrammarCompiler(tokenizer_info)


def _json_matcher(compiler, schema):
    compiled = compiler.compile_json_schema(schema)
    return xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)


def _allowed_ids(bitmask, row=0):
    """Extract allowed token IDs from an xgrammar bitmask.

    Use ``xgr.allocate_token_bitmask(batch_size, vocab_size)`` or
    ``xgr.get_bitmask_shape(batch_size, vocab_size)`` to obtain / query the
    bitmask shape — do NOT hard-code ``ceil(vocab_size / 32)`` yourself.

    The internal bit-packing format may change across xgrammar versions.
    Current format: int32 words, each bit maps to one token.
    We decode bit-by-bit so the helper stays format-agnostic.
    """
    bm_np = bitmask.numpy()
    ids = set()
    for word_idx in range(bm_np.shape[1]):
        word = int(bm_np[row, word_idx]) & 0xFFFFFFFF  # treat as unsigned
        if word != 0:
            for bit in range(32):
                if word & (1 << bit):
                    ids.add(word_idx * 32 + bit)
    return ids


# ===========================================================================
# 1. _expand_sampling_inputs — non-tensor field expansion
# ===========================================================================


class TestExpandSamplingInputsNonTensor:
    """_expand_sampling_inputs must repeat non-tensor fields (response_formats,
    session_ctx, logits_processors, session_to_cleanup) so that every expanded
    position carries the same guided-decoding context as its source batch
    element."""

    # ---- response_formats (tuple) ----

    def test_response_formats_repeated(self):
        fmt = {'type': 'json_schema', 'json_schema': {'name': 't', 'schema': {'type': 'object'}}}
        si = SamplingInputs(
            max_top_k=1,
            batch_size=2,
            response_formats=(fmt, None),
        )
        expanded = _expand_sampling_inputs(si, num_tokens=3)
        # batch_size = 2 × 3 = 6
        assert expanded.batch_size == 6
        assert len(expanded.response_formats) == 6
        # [fmt, fmt, fmt, None, None, None]
        assert expanded.response_formats[:3] == (fmt, fmt, fmt)
        assert expanded.response_formats[3:] == (None, None, None)

    def test_response_formats_mixed_batch(self):
        guided = {'type': 'json_schema', 'json_schema': {'name': 't', 'schema': {'type': 'object'}}}
        si = SamplingInputs(
            max_top_k=1,
            batch_size=3,
            response_formats=(guided, None, guided),
        )
        expanded = _expand_sampling_inputs(si, num_tokens=2)
        assert len(expanded.response_formats) == 6
        # [guided, guided, None, None, guided, guided]
        assert expanded.response_formats[0] == guided
        assert expanded.response_formats[1] == guided
        assert expanded.response_formats[2] is None
        assert expanded.response_formats[3] is None
        assert expanded.response_formats[4] == guided
        assert expanded.response_formats[5] == guided

    def test_response_formats_empty(self):
        si = SamplingInputs(max_top_k=1, batch_size=2, response_formats=())
        expanded = _expand_sampling_inputs(si, num_tokens=3)
        assert expanded.response_formats == ()

    def test_num_tokens_1_identity(self):
        fmt = {'type': 'json_schema', 'json_schema': {'name': 't', 'schema': {'type': 'object'}}}
        si = SamplingInputs(max_top_k=1, batch_size=2, response_formats=(fmt, None))
        result = _expand_sampling_inputs(si, num_tokens=1)
        assert result is si

    def test_session_ctx_none(self):
        si = SamplingInputs(max_top_k=1, batch_size=2, session_ctx=None)
        expanded = _expand_sampling_inputs(si, num_tokens=3)
        assert expanded.session_ctx is None

    # ---- tensor fields still correct ----

    def test_tensor_fields_still_expanded(self):
        si = SamplingInputs(
            max_top_k=1,
            batch_size=2,
            temperature=torch.tensor([0.5, 1.0]),
            top_k=torch.tensor([1, 10]),
            random_offsets=torch.tensor([100, 200]),
            response_formats=({'type': 'json_schema'}, None),
            session_ctx=[{'session_id': 1, 'seq_id': 10}, {'session_id': 2, 'seq_id': 20}],
        )
        expanded = _expand_sampling_inputs(si, num_tokens=3)
        # Tensor fields
        assert expanded.temperature.shape[0] == 6
        torch.testing.assert_close(expanded.temperature, torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0, 1.0]))
        torch.testing.assert_close(expanded.random_offsets, torch.tensor([100, 101, 102, 200, 201, 202]))
        # Non-tensor fields
        assert len(expanded.response_formats) == 6
        assert len(expanded.session_ctx) == 6


# ===========================================================================
# 2. _slice_sampling_inputs — non-tensor field slicing
# ===========================================================================


class TestSliceSamplingInputsNonTensor:
    """After expansion, _slice_sampling_inputs must also slice non-tensor
    fields back to the expected size."""

    def _make_expanded(self, num_tokens=3, batch_size=2):
        """Create an already-expanded SamplingInputs (as if expansion handled
        non-tensor fields correctly)."""
        total = batch_size * num_tokens
        fmt = {'type': 'json_schema', 'json_schema': {'name': 't', 'schema': {'type': 'object'}}}
        return SamplingInputs(
            max_top_k=1,
            batch_size=total,
            temperature=torch.ones(total),
            response_formats=tuple([fmt] * num_tokens + [None] * num_tokens),
            session_ctx=[{'session_id': 1, 'seq_id': 10}] * num_tokens
                        + [{'session_id': 2, 'seq_id': 20}] * num_tokens,
        )

    def test_slice_is_last_true(self):
        """is_last=True → one element per original batch (the last token)."""
        si = self._make_expanded(num_tokens=3, batch_size=2)
        sliced = _slice_sampling_inputs(si, num_tokens=3, is_last=True)
        assert sliced.batch_size == 2
        assert len(sliced.response_formats) == 2
        assert len(sliced.session_ctx) == 2
        # Last token per batch: indices 2 and 5
        assert sliced.response_formats[0] == si.response_formats[2]
        assert sliced.response_formats[1] is None
        assert sliced.session_ctx[0] == {'session_id': 1, 'seq_id': 10}
        assert sliced.session_ctx[1] == {'session_id': 2, 'seq_id': 20}

    def test_slice_is_last_false(self):
        """is_last=False → num_tokens-1 elements per original batch."""
        si = self._make_expanded(num_tokens=3, batch_size=2)
        sliced = _slice_sampling_inputs(si, num_tokens=3, is_last=False)
        assert sliced.batch_size == 4  # 2 * (3-1)
        assert len(sliced.response_formats) == 4
        assert len(sliced.session_ctx) == 4
        # First 2 tokens per batch: [fmt, fmt, None, None]
        assert sliced.response_formats[0] == si.response_formats[0]
        assert sliced.response_formats[1] == si.response_formats[1]
        assert sliced.response_formats[2] is None
        assert sliced.response_formats[3] is None

    def test_slice_num_tokens_1_identity(self):
        si = SamplingInputs(
            max_top_k=1,
            batch_size=2,
            temperature=torch.ones(2),
            response_formats=({'type': 'json_schema'}, None),
        )
        result = _slice_sampling_inputs(si, num_tokens=1)
        assert result is si


# ===========================================================================
# 3. Grammar state management — fork
# ===========================================================================


class TestGrammarFork:
    """Fork() creates an independent GrammarMatcher snapshot."""

    def test_fork_is_independent_object(self, compiler):
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        matcher = _json_matcher(compiler, schema)
        forked = matcher.fork()
        assert forked is not matcher

    def test_accept_string_on_fork_does_not_affect_original(self, compiler, tokenizer_info):
        """accept_string on a forked matcher does not change the original."""
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        original = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        # Record original's allowed tokens
        bm_orig = xgr.allocate_token_bitmask(1, vocab_size)
        original.fill_next_token_bitmask(bm_orig, 0)
        orig_allowed = _allowed_ids(bm_orig)

        # Fork and advance fork via accept_string (reliably changes state)
        forked = original.fork()
        forked.accept_string('{"')

        # Original should still be at initial state
        bm_check = xgr.allocate_token_bitmask(1, vocab_size)
        original.fill_next_token_bitmask(bm_check, 0)
        check_allowed = _allowed_ids(bm_check)
        assert check_allowed == orig_allowed

    def test_fork_chain_for_spec_positions(self, compiler, tokenizer_info):
        """Simulate the spec-decode fork pattern: for each speculative
        position, fork from current state, then advance current state
        via accept_string.

        Each fork captures the grammar state at position i.
        The original matcher stays at the initial state.
        """
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        original = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        num_spec = 4
        forks = []
        current = original.fork()

        # Advance through JSON construction using accept_string
        advance_strings = ['{"', 'name', '"', ':', '"']
        for i in range(min(num_spec, len(advance_strings))):
            pos_fork = current.fork()
            forks.append(pos_fork)
            current.accept_string(advance_strings[i])

        # Original unchanged
        bm_orig = xgr.allocate_token_bitmask(1, vocab_size)
        original.fill_next_token_bitmask(bm_orig, 0)
        orig_allowed = _allowed_ids(bm_orig)

        bm_fresh = xgr.allocate_token_bitmask(1, vocab_size)
        _json_matcher(compiler, schema).fill_next_token_bitmask(bm_fresh, 0)
        fresh_allowed = _allowed_ids(bm_fresh)

        assert orig_allowed == fresh_allowed

        # Forks should have progressively different states
        fork_allowed_sets = []
        for fk in forks:
            bm = xgr.allocate_token_bitmask(1, vocab_size)
            fk.fill_next_token_bitmask(bm, 0)
            fork_allowed_sets.append(_allowed_ids(bm))

        # First and last must differ (grammar state advanced via accept_string)
        assert fork_allowed_sets[0] != fork_allowed_sets[-1], (
            'Fork chain must capture progressively different grammar states'
        )


# ===========================================================================
# 4. Grammar state management — rollback
# ===========================================================================


class TestGrammarRollback:
    """Rollback(n) reverts the last n accept_string / accept_token calls.

    Note: With BPE tokenizers, rollback counts the number of *accept* calls
    (not characters). accept_string('abc') counts as 1 accept step.
    """

    def test_rollback_one_accept_string_step(self, compiler, tokenizer_info):
        """Rollback 1 accept_string step reverts to previous state."""
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        # Record initial state
        bm0 = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(bm0, 0)
        initial = _allowed_ids(bm0)

        # Advance with accept_string
        matcher.accept_string('{"')

        # Rollback 1 step
        matcher.rollback(1)

        # Should be back to initial
        bm1 = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(bm1, 0)
        assert _allowed_ids(bm1) == initial

    def test_rollback_partial(self, compiler, tokenizer_info):
        """Accept 3 steps, rollback 1 → state equals state after 2 steps."""
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        # Accept 2 steps, record state
        matcher.accept_string('{"')
        matcher.accept_string('name')

        bm_after_2 = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(bm_after_2, 0)
        expected = _allowed_ids(bm_after_2)

        # Accept 1 more
        matcher.accept_string('"')

        # Rollback 1
        matcher.rollback(1)

        bm_check = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(bm_check, 0)
        assert _allowed_ids(bm_check) == expected

    def test_rollback_after_partial_rejection(self, compiler, tokenizer_info):
        """Simulate rejection sampling: advance N steps, rollback to K < N."""
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        # Advance 4 steps (simulating draft generation)
        advance_steps = ['{"', 'name', '"', ':']
        for s in advance_steps:
            matcher.accept_string(s)

        # Rejection: only 1 step accepted → rollback 3
        num_accepted = 1
        rollback_count = len(advance_steps) - num_accepted
        matcher.rollback(rollback_count)

        # State should match a matcher that accepted only 1 step
        reference = _json_matcher(compiler, schema)
        reference.accept_string(advance_steps[0])

        bm_ref = xgr.allocate_token_bitmask(1, vocab_size)
        reference.fill_next_token_bitmask(bm_ref, 0)
        bm_actual = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(bm_actual, 0)

        assert _allowed_ids(bm_actual) == _allowed_ids(bm_ref)


# ===========================================================================
# 5. Grammar state management — fork-based strategy for spec decode
# ===========================================================================


class TestGrammarForkStrategy:
    """The recommended approach: use fork() during draft generation and
    target verification, then accept only the final output tokens on the
    original matcher."""

    def test_fork_strategy_target_verification(self, compiler, tokenizer_info):
        """Target model verification uses forked matchers to apply position-
        dependent grammar masks without mutating the original.

        After rejection sampling, accept the final output on the original.
        """
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        original = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size
        num_spec = 3

        # Record original's initial state
        bm_orig = xgr.allocate_token_bitmask(1, vocab_size)
        original.fill_next_token_bitmask(bm_orig, 0)
        orig_allowed = _allowed_ids(bm_orig)

        # Phase 1: Fork-based verification masks (don't mutate original)
        advance_strings = ['{"', 'name', '"']
        forks = []
        current = original.fork()
        for i in range(num_spec):
            pos_fork = current.fork()
            forks.append(pos_fork)
            if i < len(advance_strings):
                current.accept_string(advance_strings[i])

        # Original must be unchanged
        bm_check = xgr.allocate_token_bitmask(1, vocab_size)
        original.fill_next_token_bitmask(bm_check, 0)
        assert _allowed_ids(bm_check) == orig_allowed

        # Phase 2: After rejection sampling, accept output on original
        # Simulate: all spec tokens accepted + bonus
        output_strings = ['{"', 'name', '"', ':']
        for s in output_strings:
            original.accept_string(s)

        # Original should have advanced
        bm_final = xgr.allocate_token_bitmask(1, vocab_size)
        original.fill_next_token_bitmask(bm_final, 0)
        final_allowed = _allowed_ids(bm_final)
        assert final_allowed != orig_allowed


# ===========================================================================
# 6. Positional-serial grammar mask
# ===========================================================================


class TestPositionalSerialGrammarMask:
    """In spec decode, the grammar mask for position i depends on what tokens
    were accepted at positions 0..i-1.

    Applying the same mask to all positions (parallel mask) is INCORRECT for position 1+.
    """

    def test_mask_changes_after_accept_string(self, compiler, tokenizer_info):
        """After accept_string, the allowed token set changes."""
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        bm0 = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(bm0, 0)
        before = _allowed_ids(bm0)

        matcher.accept_string('{"')

        bm1 = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(bm1, 0)
        after = _allowed_ids(bm1)

        assert before != after, 'Grammar mask must change after accept_string'

    def test_parallel_mask_incorrect_for_later_positions(self, compiler, tokenizer_info):
        """Parallel (same mask for all positions) differs from serial
        (position-dependent mask).

        This proves spec decode MUST use serial mask application.
        """
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size
        num_spec = 3

        # Parallel: same initial mask for all positions
        parallel_bm = xgr.allocate_token_bitmask(num_spec, vocab_size)
        matcher.fill_next_token_bitmask(parallel_bm, 0)
        for pos in range(1, num_spec):
            parallel_bm.numpy()[pos] = parallel_bm.numpy()[0]

        # Serial: advance grammar state per position using accept_string
        serial_bm = xgr.allocate_token_bitmask(num_spec, vocab_size)
        temp = _json_matcher(compiler, schema)
        advance_strings = ['{"', 'name', '"']
        for pos in range(num_spec):
            temp.fill_next_token_bitmask(serial_bm, pos)
            if pos < len(advance_strings):
                # Pick an allowed token from the mask, then accept_string
                # to advance grammar state for next position
                temp.accept_string(advance_strings[pos])

        # At position 1+, the masks must differ from the parallel (initial) mask
        pos0_parallel = _allowed_ids(parallel_bm, row=0)
        pos0_serial = _allowed_ids(serial_bm, row=0)
        assert pos0_parallel == pos0_serial, 'Position 0 masks should match'

        pos1_parallel = _allowed_ids(parallel_bm, row=1)
        pos1_serial = _allowed_ids(serial_bm, row=1)
        assert pos1_parallel != pos1_serial, (
            'Parallel mask (same for all positions) is incorrect for pos 1+'
        )

    def test_serial_mask_with_fork_per_position(self, compiler, tokenizer_info):
        """Each speculative position uses a fork to get the correct mask
        without mutating the original matcher."""
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        original = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size
        num_spec = 4

        advance_strings = ['{"', 'name', '"', ':']
        position_forks = []
        current = original.fork()  # work on a fork, not original directly
        for pos in range(num_spec):
            fork_at_pos = current.fork()
            position_forks.append(fork_at_pos)
            if pos < len(advance_strings):
                current.accept_string(advance_strings[pos])

        # Verify: each fork captures grammar state at position i
        allowed_per_pos = []
        for fk in position_forks:
            bm = xgr.allocate_token_bitmask(1, vocab_size)
            fk.fill_next_token_bitmask(bm, 0)
            allowed_per_pos.append(_allowed_ids(bm))

        # First and last must differ (grammar state advanced via accept_string)
        assert allowed_per_pos[0] != allowed_per_pos[-1], (
            'Position forks must capture different grammar states'
        )

        # Original matcher should be unchanged
        bm_orig = xgr.allocate_token_bitmask(1, vocab_size)
        original.fill_next_token_bitmask(bm_orig, 0)
        bm_fresh = xgr.allocate_token_bitmask(1, vocab_size)
        _json_matcher(compiler, schema).fill_next_token_bitmask(bm_fresh, 0)
        assert _allowed_ids(bm_orig) == _allowed_ids(bm_fresh)


# ===========================================================================
# 7. Draft model grammar masking (logic validation)
# ===========================================================================


class TestDraftModelGrammarMasking:
    """Draft model must apply grammar mask before argmax to produce
    grammatically valid draft tokens."""

    def test_unmasked_argmax_may_pick_invalid_token(self, compiler, tokenizer_info):
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        bm = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(bm, 0)
        allowed = _allowed_ids(bm)
        disallowed = set(range(vocab_size)) - allowed

        if not disallowed:
            pytest.skip('all tokens allowed (unlikely with real tokenizer)')

        # Create logits that strongly prefer a disallowed token
        logits = torch.full((1, vocab_size), -100.0)
        bad_token = int(list(disallowed)[0])
        logits[0, bad_token] = 100.0

        unmasked_choice = logits.argmax(dim=-1).item()
        assert unmasked_choice not in allowed

    def test_masked_argmax_picks_valid_token(self, compiler, tokenizer_info):
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        bm = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(bm, 0)
        allowed = _allowed_ids(bm)
        disallowed = set(range(vocab_size)) - allowed

        logits = torch.full((1, vocab_size), -100.0)
        if disallowed:
            logits[0, int(list(disallowed)[0])] = 100.0

        xgr.apply_token_bitmask_inplace(logits, bm)
        masked_choice = logits.argmax(dim=-1).item()
        assert masked_choice in allowed

    def test_draft_chain_all_valid(self, compiler, tokenizer_info):
        """Generate a chain of draft tokens with grammar mask at each step.

        Every token should be valid at its position.
        """
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size
        num_spec = 4

        for step in range(num_spec):
            bm = xgr.allocate_token_bitmask(1, vocab_size)
            matcher.fill_next_token_bitmask(bm, 0)
            allowed = _allowed_ids(bm)
            if len(allowed) == 0:
                break  # grammar terminated

            logits = torch.randn(1, vocab_size)
            xgr.apply_token_bitmask_inplace(logits, bm)
            token = logits.argmax(dim=-1).item()
            assert token in allowed, f'Step {step}: token {token} not in allowed set'
            matcher.accept_token(token)


# ===========================================================================
# 8. Grammar state after rejection sampling
# ===========================================================================


class TestGrammarStateAfterRejection:
    """After rejection sampling, the grammar matcher's state must reflect
    exactly the accepted tokens."""

    def test_rollback_then_accept_rejection_output(self, compiler, tokenizer_info):
        """Draft model advanced grammar state N steps via accept_string.

        Rejection says only K < N were accepted. Rollback N-K steps, then accept the target model's output tokens.
        """
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        # Accept 4 draft steps
        draft_steps = ['{"', 'name', '"', ':']
        for s in draft_steps:
            matcher.accept_string(s)

        # Rejection: only 1 step accepted → rollback 3
        num_accepted = 1
        rollback_count = len(draft_steps) - num_accepted
        matcher.rollback(rollback_count)

        # Now accept target's output (the 1 accepted step + 1 bonus)
        target_output_steps = [draft_steps[0], '"']  # 1 accepted + bonus
        for s in target_output_steps:
            matcher.accept_string(s)

        # Build reference from scratch
        reference = _json_matcher(compiler, schema)
        for s in target_output_steps:
            reference.accept_string(s)

        # Compare states
        bm_actual = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(bm_actual, 0)
        bm_ref = xgr.allocate_token_bitmask(1, vocab_size)
        reference.fill_next_token_bitmask(bm_ref, 0)

        assert _allowed_ids(bm_actual) == _allowed_ids(bm_ref)

    def test_fork_strategy_all_accepted(self, compiler, tokenizer_info):
        """All draft tokens accepted → accept all + bonus on original."""
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        original = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        # All spec tokens accepted + bonus
        accepted_strings = ['{"', 'name', '"', ':']  # 3 spec + 1 bonus
        for s in accepted_strings:
            original.accept_string(s)

        # Verify state matches reference
        reference = _json_matcher(compiler, schema)
        for s in accepted_strings:
            reference.accept_string(s)

        bm_orig = xgr.allocate_token_bitmask(1, vocab_size)
        original.fill_next_token_bitmask(bm_orig, 0)
        bm_ref = xgr.allocate_token_bitmask(1, vocab_size)
        reference.fill_next_token_bitmask(bm_ref, 0)

        assert _allowed_ids(bm_orig) == _allowed_ids(bm_ref)

    def test_fork_strategy_partial_rejection(self, compiler, tokenizer_info):
        """Partial rejection: only some draft tokens accepted.
        Original matcher accepts exactly the final output tokens."""
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        original = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        # Simulate: 3 spec steps, only 1 accepted + 1 bonus
        final_strings = ['{"', ':']  # 1 accepted draft + 1 bonus
        for s in final_strings:
            original.accept_string(s)

        # Verify
        reference = _json_matcher(compiler, schema)
        for s in final_strings:
            reference.accept_string(s)

        bm_orig = xgr.allocate_token_bitmask(1, vocab_size)
        original.fill_next_token_bitmask(bm_orig, 0)
        bm_ref = xgr.allocate_token_bitmask(1, vocab_size)
        reference.fill_next_token_bitmask(bm_ref, 0)

        assert _allowed_ids(bm_orig) == _allowed_ids(bm_ref)
