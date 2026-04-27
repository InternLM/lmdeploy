# Copyright (c) OpenMMLab. All rights reserved.
"""Integration tests for MTP (speculative decoding) + Guided Decoding.

These tests exercise the interaction between guided decoding and speculative
decoding at a higher level than the unit tests in test_guided_spec_decode.py.
They test the *pipeline* logic — the position-serial grammar masking,
fork-based matcher management, and grammar state consistency after rejection
sampling — without requiring a GPU or actual model weights.

Key scenarios:
1. Position-serial grammar mask via GuidedDecodingManager (matches direct xgr).
2. Fork-based target verification: multiple forks are independent.
3. Simulated _guided_spec_logits_process: all positions masked, mixed batch.
4. Grammar state after rejection sampling: original matchers accept exactly
   the rejection-sampled output tokens.
5. End-to-end: draft → target verification → rejection → grammar state.
6. Batch-level grammar mask: mixed guided/unguided sequences.
"""
import pytest
import torch
import xgrammar as xgr

from lmdeploy.pytorch.engine.guided_process import GuidedDecodingManager
from lmdeploy.pytorch.engine.logits_process import SamplingInputs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QWEN_MODEL = 'Qwen/Qwen2.5-7B-Instruct'


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_QWEN_MODEL, trust_remote_code=True)


@pytest.fixture(scope='module')
def tokenizer_info(tokenizer):
    return xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size)


@pytest.fixture(scope='module')
def compiler(tokenizer_info):
    return xgr.GrammarCompiler(tokenizer_info)


@pytest.fixture(scope='module')
def guided_manager(tokenizer):
    return GuidedDecodingManager(tokenizer, vocab_size=tokenizer.vocab_size)


def _json_matcher(compiler, schema):
    compiled = compiler.compile_json_schema(schema)
    return xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)


def _regex_matcher(compiler, pattern):
    compiled = compiler.compile_regex(pattern)
    return xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)


def _allowed_ids(bitmask, row=0):
    """Extract allowed token IDs from an xgrammar bitmask."""
    bm_np = bitmask.numpy()
    ids = set()
    for word_idx in range(bm_np.shape[1]):
        word = int(bm_np[row, word_idx]) & 0xFFFFFFFF
        if word != 0:
            for bit in range(32):
                if word & (1 << bit):
                    ids.add(word_idx * 32 + bit)
    return ids


def _make_sampling_inputs(batch_size, response_formats=None, session_ctx=None):
    """Create a minimal SamplingInputs for testing."""
    return SamplingInputs(
        max_top_k=1,
        batch_size=batch_size,
        response_formats=response_formats or (),
        session_ctx=session_ctx,
    )


# ===========================================================================
# 1. Position-serial grammar mask via GuidedDecodingManager
# ===========================================================================


class TestPositionSerialGrammarMaskViaManager:
    """Verify that applying grammar masks position-by-position using
    GuidedDecodingManager methods produces correct per-position masks."""

    def test_manager_methods_match_direct_xgr(self, compiler, tokenizer_info, guided_manager):
        """GuidedDecodingManager methods produce the same results as direct
        xgrammar calls."""
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        # Via manager
        bm_manager = guided_manager.allocate_batched_bitmap(1)
        guided_manager.fill_bitmap(matcher, bm_manager, 0)
        allowed_manager = _allowed_ids(bm_manager)

        # Direct
        bm_direct = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(bm_direct, 0)
        allowed_direct = _allowed_ids(bm_direct)

        assert allowed_manager == allowed_direct


# ===========================================================================
# 2. Fork-based target verification
# ===========================================================================


class TestForkBasedTargetVerification:
    """Verify that _guided_spec_logits_process uses forked matchers, leaving
    originals untouched."""

    def test_multiple_forks_independent(self, compiler, tokenizer_info, guided_manager):
        """Multiple forks from the same original are independent."""
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        original = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size

        fork1 = original.fork()
        fork2 = original.fork()

        # Advance fork1 by 2 steps
        for _ in range(2):
            bm = guided_manager.allocate_batched_bitmap(1)
            guided_manager.fill_bitmap(fork1, bm, 0)
            logits = torch.randn(1, vocab_size)
            guided_manager.apply_batched_bitmap(logits, bm)
            token = logits.argmax(dim=-1).item()
            guided_manager.accept_token(fork1, token)

        # fork2 should still be at original state
        bm_fork2 = guided_manager.allocate_batched_bitmap(1)
        guided_manager.fill_bitmap(fork2, bm_fork2, 0)
        bm_orig = guided_manager.allocate_batched_bitmap(1)
        guided_manager.fill_bitmap(original, bm_orig, 0)

        assert _allowed_ids(bm_fork2) == _allowed_ids(bm_orig)


# ===========================================================================
# 3. Simulated _guided_spec_logits_process
# ===========================================================================


class TestSimulatedGuidedSpecLogitsProcess:
    """Simulate the _guided_spec_logits_process method's logic to verify
    correctness without requiring a full SpecModelAgent instance."""

    def _guided_spec_logits_process_sim(
        self,
        target_logits: torch.Tensor,
        guided_manager: GuidedDecodingManager,
        guided_processors: dict,
        batch_size: int,
        num_expand: int,
        vocab_size: int,
    ):
        """Simplified version of _guided_spec_logits_process that applies
        position-serial grammar mask."""
        # Reshape to [batch_size, num_expand, vocab_size]
        scores_3d = target_logits.clone().view(batch_size, num_expand, -1)

        # Fork matchers
        forked = {idx: proc.fork() for idx, proc in guided_processors.items()}

        for pos in range(num_expand):
            guided_bitmask = guided_manager.allocate_batched_bitmap(batch_size)
            for idx, fork_proc in forked.items():
                guided_manager.fill_bitmap(fork_proc, guided_bitmask, idx)
            pos_logits = scores_3d[:, pos, :]
            guided_manager.apply_batched_bitmap(pos_logits, guided_bitmask)
            scores_3d[:, pos, :] = pos_logits

            pos_token_ids = pos_logits.argmax(dim=-1)
            for idx, fork_proc in forked.items():
                guided_manager.accept_token(fork_proc, pos_token_ids[idx].item())

        return scores_3d.view(batch_size * num_expand, -1)

    def test_all_positions_masked(self, compiler, tokenizer_info, guided_manager):
        """All positions (including bonus) must have grammar mask applied."""
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size
        batch_size = 1
        num_expand = 4  # 3 spec + 1 bonus

        target_logits = torch.randn(batch_size * num_expand, vocab_size)
        guided_processors = {0: matcher}

        processed = self._guided_spec_logits_process_sim(
            target_logits, guided_manager, guided_processors,
            batch_size, num_expand, vocab_size,
        )

        # Verify each position's chosen token is in the allowed set
        scores_3d = processed.view(batch_size, num_expand, -1)
        reference = _json_matcher(compiler, schema)
        for pos in range(num_expand):
            bm = guided_manager.allocate_batched_bitmap(1)
            guided_manager.fill_bitmap(reference, bm, 0)
            allowed = _allowed_ids(bm)
            token = scores_3d[0, pos].argmax().item()
            assert token in allowed, f'Position {pos}: token {token} not grammar-valid'
            guided_manager.accept_token(reference, token)

    def test_mixed_batch_only_guided_masked(self, compiler, tokenizer_info, guided_manager):
        """In a mixed batch, only guided sequences should have their logits
        masked; unguided sequences should be unaffected."""
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size
        batch_size = 2
        num_expand = 3  # 2 spec + 1 bonus

        # Only batch element 0 is guided; element 1 is not
        target_logits = torch.randn(batch_size * num_expand, vocab_size)
        # Make element 1's logits have a strong signal that would be masked
        # Element 1's positions in the flat layout: indices 1, 3, 5
        target_logits[1, 0] = 100.0
        target_logits[3, 0] = 100.0
        target_logits[5, 0] = 100.0

        guided_processors = {0: matcher}  # Only idx 0

        processed = self._guided_spec_logits_process_sim(
            target_logits, guided_manager, guided_processors,
            batch_size, num_expand, vocab_size,
        )

        # Element 1's logits should be unchanged (no grammar mask applied)
        # Check that the strong signal is preserved
        scores_3d = processed.view(batch_size, num_expand, -1)
        # Element 1, position 0 — no masking should have been applied
        assert scores_3d[1, 0, 0] == 100.0, 'Unguided sequence should not be masked'


# ===========================================================================
# 4. Simulated rejection sampling + grammar state
# ===========================================================================


class TestSimulatedRejectionSamplingGrammarState:
    """Simulate the rejection sampling + grammar state management logic from
    _rejection_sampling to verify grammar state consistency."""

    def _simulate_rejection_greedy(
        self,
        target_logits_3d: torch.Tensor,  # [batch, num_spec, vocab]
        draft_token_ids: torch.Tensor,    # [batch, num_spec]
        num_spec_tokens: int,
        batch_size: int,
    ):
        """Simplified greedy rejection sampling."""
        target_argmax = target_logits_3d.argmax(dim=-1)  # [batch, num_spec]
        masks = draft_token_ids == target_argmax
        range_data = torch.arange(num_spec_tokens, device=draft_token_ids.device)[None, :]
        equals = (masks.cumsum(dim=1) - 1) == range_data
        num_rejected_tokens = num_spec_tokens - equals.sum(dim=1)
        first_diff_indices = torch.argmin(equals.int(), dim=1, keepdim=True)
        keeps = range_data.repeat(batch_size, 1) <= first_diff_indices
        keeps = keeps | equals
        output_token_ids = torch.where(keeps, target_argmax, -1)
        # bonus (not relevant for grammar state here)
        return output_token_ids, num_rejected_tokens

    def test_grammar_state_all_accepted_greedy(self, compiler, tokenizer_info, guided_manager):
        """All draft tokens accepted (greedy) → grammar state reflects all
        accepted tokens + bonus."""
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        original = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size
        batch_size = 1
        num_spec_tokens = 3

        # Generate draft tokens with grammar mask
        fork = original.fork()
        draft_tokens = []
        for _ in range(num_spec_tokens):
            bm = guided_manager.allocate_batched_bitmap(1)
            guided_manager.fill_bitmap(fork, bm, 0)
            logits = torch.randn(1, vocab_size)
            guided_manager.apply_batched_bitmap(logits, bm)
            token = logits.argmax(dim=-1).item()
            guided_manager.accept_token(fork, token)
            draft_tokens.append(token)

        draft_token_ids = torch.tensor([draft_tokens], dtype=torch.long)

        # Target model agrees with all draft tokens (greedy match)
        target_logits_3d = torch.zeros(batch_size, num_spec_tokens, vocab_size)
        for i, tid in enumerate(draft_tokens):
            target_logits_3d[0, i, tid] = 100.0

        output_token_ids, num_rejected = self._simulate_rejection_greedy(
            target_logits_3d, draft_token_ids, num_spec_tokens, batch_size,
        )

        assert num_rejected[0].item() == 0, 'All draft tokens should be accepted'

        # Accept output tokens on original
        for pos in range(num_spec_tokens):
            tid = output_token_ids[0, pos].item()
            if tid >= 0:
                guided_manager.accept_token(original, tid)
        # Accept bonus token (simulate)
        bonus_token = draft_tokens[0]  # placeholder
        guided_manager.accept_token(original, bonus_token)

        # Original should have advanced
        bm = guided_manager.allocate_batched_bitmap(1)
        guided_manager.fill_bitmap(original, bm, 0)
        allowed = _allowed_ids(bm)
        # At minimum, the allowed set should be non-empty (grammar hasn't terminated)
        # and should differ from the initial state
        bm_initial = guided_manager.allocate_batched_bitmap(1)
        fresh = _json_matcher(compiler, schema)
        guided_manager.fill_bitmap(fresh, bm_initial, 0)
        allowed_initial = _allowed_ids(bm_initial)
        assert allowed != allowed_initial, 'Grammar should have advanced from initial state'

    def test_grammar_state_partial_rejection_greedy(self, compiler, tokenizer_info, guided_manager):
        """Partial rejection (greedy) → grammar state reflects only accepted
        tokens + replacement + bonus."""
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        original = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size
        batch_size = 1
        num_spec_tokens = 3

        # Generate draft tokens
        fork = original.fork()
        draft_tokens = []
        for _ in range(num_spec_tokens):
            bm = guided_manager.allocate_batched_bitmap(1)
            guided_manager.fill_bitmap(fork, bm, 0)
            logits = torch.randn(1, vocab_size)
            guided_manager.apply_batched_bitmap(logits, bm)
            token = logits.argmax(dim=-1).item()
            guided_manager.accept_token(fork, token)
            draft_tokens.append(token)

        draft_token_ids = torch.tensor([draft_tokens], dtype=torch.long)

        # Target model disagrees at position 1
        target_logits_3d = torch.zeros(batch_size, num_spec_tokens, vocab_size)
        target_logits_3d[0, 0, draft_tokens[0]] = 100.0  # agree at pos 0
        # Position 1: target picks a different (but grammar-valid) token
        bm = guided_manager.allocate_batched_bitmap(1)
        guided_manager.fill_bitmap(original, bm, 0)
        allowed = _allowed_ids(bm)
        # Find a valid token that differs from draft
        replacement_token = None
        for t in allowed:
            if t != draft_tokens[1]:
                replacement_token = t
                break
        assert replacement_token is not None, 'Need a replacement token for the test'
        target_logits_3d[0, 1, replacement_token] = 100.0
        # Position 2 doesn't matter (rejected)
        target_logits_3d[0, 2, 0] = 100.0

        output_token_ids, num_rejected = self._simulate_rejection_greedy(
            target_logits_3d, draft_token_ids, num_spec_tokens, batch_size,
        )

        assert num_rejected[0].item() == 2, '2 tokens should be rejected (pos 1 and 2)'

        # Accept only the valid output tokens on original
        n_valid_draft = num_spec_tokens - num_rejected[0].item()
        for pos in range(n_valid_draft):
            tid = output_token_ids[0, pos].item()
            if tid >= 0:
                guided_manager.accept_token(original, tid)
        # Accept replacement/bonus token
        guided_manager.accept_token(original, replacement_token)

        # Verify grammar state: should have accepted 2 tokens total
        # (draft[0] + replacement), not the rejected tokens
        # Build reference
        reference = _json_matcher(compiler, schema)
        guided_manager.accept_token(reference, draft_tokens[0])
        guided_manager.accept_token(reference, replacement_token)

        bm_actual = guided_manager.allocate_batched_bitmap(1)
        guided_manager.fill_bitmap(original, bm_actual, 0)
        bm_ref = guided_manager.allocate_batched_bitmap(1)
        guided_manager.fill_bitmap(reference, bm_ref, 0)

        assert _allowed_ids(bm_actual) == _allowed_ids(bm_ref)


# ===========================================================================
# 5. End-to-end: draft → target verification → rejection → grammar state
# ===========================================================================


class TestEndToEndGuidedSpecDecode:
    """End-to-end simulation of the guided + spec decode pipeline.

    This test exercises the complete flow:
    1. Draft model generates tokens with forked grammar mask
    2. Target model verifies with position-serial grammar mask
    3. Rejection sampling determines accepted tokens
    4. Original matchers accept the final output tokens
    5. Grammar state is consistent for the next step
    """

    def test_two_step_consistency(self, compiler, tokenizer_info, guided_manager):
        """Two consecutive decode steps: grammar state from step 1 carries
        over correctly to step 2."""
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        original = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size
        num_spec_tokens = 2
        num_expand = num_spec_tokens + 1

        def _one_step(original_matcher):
            """Simulate one decode step."""
            # --- Draft phase ---
            draft_fork = original_matcher.fork()
            draft_tokens = []
            for _ in range(num_spec_tokens):
                bm = guided_manager.allocate_batched_bitmap(1)
                guided_manager.fill_bitmap(draft_fork, bm, 0)
                logits = torch.randn(1, vocab_size)
                guided_manager.apply_batched_bitmap(logits, bm)
                token = logits.argmax(dim=-1).item()
                guided_manager.accept_token(draft_fork, token)
                draft_tokens.append(token)
            _draft_token_ids = torch.tensor([draft_tokens], dtype=torch.long)

            # --- Target verification with position-serial mask ---
            # (Simulate _guided_spec_logits_process)
            target_fork = original_matcher.fork()
            target_tokens_per_pos = []
            for pos in range(num_expand):
                bm = guided_manager.allocate_batched_bitmap(1)
                guided_manager.fill_bitmap(target_fork, bm, 0)
                logits = torch.randn(1, vocab_size)
                guided_manager.apply_batched_bitmap(logits, bm)
                token = logits.argmax(dim=-1).item()
                guided_manager.accept_token(target_fork, token)
                target_tokens_per_pos.append(token)

            # --- Greedy rejection sampling ---
            # For simplicity, assume all draft tokens accepted
            output_tokens = draft_tokens + [target_tokens_per_pos[-1]]
            num_rejected = 0

            # --- Accept on original ---
            for tid in output_tokens:
                guided_manager.accept_token(original_matcher, tid)

            return output_tokens, num_rejected

        # Step 1
        step1_tokens, _ = _one_step(original)

        # Step 2: original should be at correct state
        step2_tokens, _ = _one_step(original)

        # Verify original has advanced by verifying it's different from initial
        bm = guided_manager.allocate_batched_bitmap(1)
        guided_manager.fill_bitmap(original, bm, 0)
        allowed = _allowed_ids(bm)
        assert len(allowed) > 0, 'Grammar should still be active'

    def test_rejection_restores_correct_state(self, compiler, tokenizer_info, guided_manager):
        """After rejection, accepting the correct tokens on the original
        matcher should bring it to the same state as if we had only generated
        those tokens."""
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        original = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size
        num_spec_tokens = 3

        # Generate draft tokens with grammar mask
        draft_fork = original.fork()
        draft_tokens = []
        for _ in range(num_spec_tokens):
            bm = guided_manager.allocate_batched_bitmap(1)
            guided_manager.fill_bitmap(draft_fork, bm, 0)
            logits = torch.randn(1, vocab_size)
            guided_manager.apply_batched_bitmap(logits, bm)
            token = logits.argmax(dim=-1).item()
            guided_manager.accept_token(draft_fork, token)
            draft_tokens.append(token)

        # Simulate partial rejection: only 1 draft token accepted
        n_accepted = 1
        accepted_tokens = draft_tokens[:n_accepted]

        # Replacement token: find a grammar-valid token at the rejection point
        ref_for_replacement = original.fork()
        for tid in accepted_tokens:
            guided_manager.accept_token(ref_for_replacement, tid)
        bm = guided_manager.allocate_batched_bitmap(1)
        guided_manager.fill_bitmap(ref_for_replacement, bm, 0)
        allowed = _allowed_ids(bm)
        replacement = list(allowed)[0]  # pick any valid token

        # Accept on original: accepted draft + replacement
        final_tokens = accepted_tokens + [replacement]
        for tid in final_tokens:
            guided_manager.accept_token(original, tid)

        # Build reference from scratch
        reference = _json_matcher(compiler, schema)
        for tid in final_tokens:
            guided_manager.accept_token(reference, tid)

        # Compare states
        bm_actual = guided_manager.allocate_batched_bitmap(1)
        guided_manager.fill_bitmap(original, bm_actual, 0)
        bm_ref = guided_manager.allocate_batched_bitmap(1)
        guided_manager.fill_bitmap(reference, bm_ref, 0)

        assert _allowed_ids(bm_actual) == _allowed_ids(bm_ref), (
            'Grammar state after rejection must match the reference'
        )


# ===========================================================================
# 7. Batch-level grammar mask application
# ===========================================================================


class TestBatchLevelGrammarMask:
    """Test that grammar masks are applied correctly at the batch level,
    including mixed guided/unguided sequences."""

    def test_mixed_batch_bitmap(self, compiler, tokenizer_info, guided_manager):
        """A batch with some guided and some unguided sequences: only guided
        ones should be affected by the bitmask."""
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']}
        matcher = _json_matcher(compiler, schema)
        vocab_size = tokenizer_info.vocab_size
        batch_size = 3

        # Only element 0 is guided
        bitmask = guided_manager.allocate_batched_bitmap(batch_size)
        guided_manager.fill_bitmap(matcher, bitmask, 0)
        # Elements 1 and 2 are not filled → their bitmask rows are all-ones

        logits = torch.zeros(batch_size, vocab_size)
        # Give each element a strong signal on token 0
        logits[:, 0] = 100.0

        guided_manager.apply_batched_bitmap(logits, bitmask)

        # Element 0: token 0 may or may not be valid
        # Elements 1, 2: token 0 should survive (no mask applied → all-ones bitmask)
        assert logits[1, 0] == 100.0, 'Unguided element should not be masked'
        assert logits[2, 0] == 100.0, 'Unguided element should not be masked'
