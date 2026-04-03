import torch

from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.spec_decode.reject_sampler import (
    PLACEHOLDER_TOKEN_ID,
    _extract_outputs,
    rejection_greedy_sample_kernel,
    rejection_sample,
    sample_recovered_tokens_kernel,
    torch_greedy_rejection_sample,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _make_peaked_logits(token_ids_2d, vocab_size):
    """Build logits where argmax(dim=-1) == token_ids_2d.

    token_ids_2d: list[list[int]] or Tensor [batch, num_spec]
    """
    if isinstance(token_ids_2d, torch.Tensor):
        token_ids_2d = token_ids_2d.tolist()
    batch_size = len(token_ids_2d)
    num_spec = len(token_ids_2d[0])
    logits = torch.full((batch_size, num_spec, vocab_size), -100.0, device=device)
    for b in range(batch_size):
        for s in range(num_spec):
            logits[b, s, token_ids_2d[b][s]] = 100.0
    return logits


def _extract_valid(output_token_ids):
    """Return list-of-lists of non-placeholder token ids."""
    out = []
    for row in output_token_ids:
        out.append(row[row != PLACEHOLDER_TOKEN_ID].tolist())
    return out


class TestRejectSample:
    """Tests for rejection_sample and torch_greedy_rejection_sample.

    Uses hand-crafted token ids so the expected output is obvious.
    """

    # ----- greedy: rejection_sample -----

    def test_greedy_all_match(self):
        """All draft == target -> accept all + bonus."""
        target_logits = _make_peaked_logits([[10, 20, 30], [10, 20, 30]], 64)
        draft = torch.tensor([[10, 20, 30], [10, 20, 30]], dtype=torch.long, device=device)
        bonus = torch.tensor([99, 88], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert out.shape == (2, 4)
        assert (rej == 0).all()
        assert _extract_valid(out) == [[10, 20, 30, 99], [10, 20, 30, 88]]
        assert last.tolist() == [99, 88]

    def test_greedy_first_mismatch(self):
        """Mismatch at position 0 -> keep only target[0]."""
        target_logits = _make_peaked_logits([[10, 20, 30], [10, 20, 30]], 64)
        draft = torch.tensor([[11, 20, 30], [11, 20, 30]], dtype=torch.long, device=device)
        bonus = torch.tensor([99, 88], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert (rej == 3).all()
        assert _extract_valid(out) == [[10], [10]]
        assert last.tolist() == [10, 10]

    def test_greedy_middle_mismatch(self):
        """Mismatch at position 2 -> keep target[0:3], no bonus."""
        target_logits = _make_peaked_logits([[10, 20, 30], [10, 20, 30]], 64)
        draft = torch.tensor([[10, 20, 31], [10, 20, 31]], dtype=torch.long, device=device)
        bonus = torch.tensor([99, 88], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert (rej == 1).all()
        assert _extract_valid(out) == [[10, 20, 30], [10, 20, 30]]
        assert last.tolist() == [30, 30]

    def test_greedy_mixed_mismatch_positions(self):
        """Each row has a different first-mismatch position.

        Row 0: all match       -> [10, 20, 30, 99]
        Row 1: mismatch at 0   -> [10]
        Row 2: mismatch at 1   -> [10, 20]
        Row 3: mismatch at 2   -> [10, 20, 30]
        """
        target_logits = _make_peaked_logits([[10, 20, 30]] * 4, 64)
        draft = torch.tensor([
            [10, 20, 30],
            [11, 20, 30],
            [10, 21, 30],
            [10, 20, 31],
        ],
                             dtype=torch.long,
                             device=device)
        bonus = torch.tensor([99, 88, 77, 66], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert rej.tolist() == [0, 3, 2, 1]
        assert _extract_valid(out) == [
            [10, 20, 30, 99],
            [10],
            [10, 20],
            [10, 20, 30],
        ]
        assert last.tolist() == [99, 10, 20, 30]

    def test_greedy_all_rejected(self):
        """Every draft wrong -> only first target token kept."""
        target_logits = _make_peaked_logits([[10, 20, 30], [10, 20, 30]], 64)
        draft = torch.tensor([[11, 21, 31], [11, 21, 31]], dtype=torch.long, device=device)
        bonus = torch.tensor([99, 88], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert (rej == 3).all()
        assert _extract_valid(out) == [[10], [10]]
        assert last.tolist() == [10, 10]

    # ----- greedy: compare rejection_sample vs torch_greedy_rejection_sample -----

    def test_greedy_triton_vs_torch_all_match(self):
        """Both implementations agree when all drafts match."""
        target_logits = _make_peaked_logits([[10, 20, 30]] * 4, 64)
        draft = torch.tensor([[10, 20, 30]] * 4, dtype=torch.long, device=device)
        bonus = torch.tensor([99, 88, 77, 66], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        t_out, t_rej, t_last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)
        p_out, p_rej, p_last = torch_greedy_rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert torch.equal(t_out, p_out)
        assert torch.equal(t_rej, p_rej)
        assert torch.equal(t_last, p_last)

    def test_greedy_triton_vs_torch_mixed(self):
        """Both implementations agree with mixed mismatch positions."""
        target_logits = _make_peaked_logits([[10, 20, 30]] * 4, 64)
        draft = torch.tensor(
            [
                [10, 20, 30],  # all match
                [11, 20, 30],  # pos 0
                [10, 21, 30],  # pos 1
                [10, 20, 31],  # pos 2
            ],
            dtype=torch.long,
            device=device)
        bonus = torch.tensor([99, 88, 77, 66], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        t_out, t_rej, t_last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)
        p_out, p_rej, p_last = torch_greedy_rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert torch.equal(t_out, p_out), f'output mismatch: triton={t_out}, torch={p_out}'
        assert torch.equal(t_rej, p_rej), f'num_rejected: triton={t_rej}, torch={p_rej}'
        assert torch.equal(t_last, p_last), f'last_token: triton={t_last}, torch={p_last}'

    def test_greedy_triton_vs_torch_large_batch(self):
        """Both implementations agree with large batch + random mismatches."""
        batch_size, num_spec, vocab = 32, 6, 128
        target_logits = torch.randn(batch_size, num_spec, vocab, device=device)
        target_argmax = target_logits.argmax(dim=-1)
        draft = target_argmax.clone()

        torch.manual_seed(42)
        flip = torch.rand(batch_size, num_spec) < 0.3
        for i in range(batch_size):
            for j in range(num_spec):
                if flip[i, j]:
                    draft[i, j] = (target_argmax[i, j] + 1) % vocab

        bonus = torch.randint(0, vocab, (batch_size, ), device=device)
        si = SamplingInputs(max_top_k=1)

        t_out, t_rej, t_last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)
        p_out, p_rej, p_last = torch_greedy_rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert torch.equal(t_out, p_out)
        assert torch.equal(t_rej, p_rej)
        assert torch.equal(t_last, p_last)

    # ----- mixed greedy/random -----

    def test_mixed_batch(self):
        """Mixed batch: greedy (top_k=1) and random sequences."""
        target_logits = _make_peaked_logits([[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]], 32)
        draft = target_logits.argmax(dim=-1)
        bonus = torch.tensor([99, 88, 77, 66], dtype=torch.long, device=device)

        top_k = torch.tensor([1, 10, 1, 10], device=device)
        si = SamplingInputs(max_top_k=10, top_k=top_k)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        # Greedy rows (0, 2) must accept all since draft == target
        assert rej[0] == 0
        assert rej[2] == 0
        assert last[0] == 99
        assert last[2] == 77

    # ----- random: no draft_probs (MTP/n-gram) -----

    def test_random_no_draft_probs(self):
        """Random sampling with draft_probs=None (MTP/n-gram mode)."""
        target_logits = _make_peaked_logits([[5, 6, 7], [5, 6, 7]], 32)
        draft = torch.tensor([[5, 6, 7], [5, 6, 7]], dtype=torch.long, device=device)
        bonus = torch.tensor([99, 88], dtype=torch.long, device=device)

        top_k = torch.tensor([10, 10], device=device)
        si = SamplingInputs(max_top_k=10, top_k=top_k)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si, draft_probs=None)

        assert out.shape == (2, 4)
        assert (last >= 0).all()

    # ----- random: all reject (peaked target, wrong draft) -----

    def test_random_all_reject(self):
        """Random: draft picks wrong token, peaked target -> reject at pos 0."""
        vocab = 32
        target_logits = _make_peaked_logits([[5, 6, 7], [5, 6, 7]], vocab)
        draft = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long, device=device)
        draft_probs = torch.zeros(2, 3, vocab, device=device)
        draft_probs[:, :, 1] = 1.0
        bonus = torch.tensor([99, 88], dtype=torch.long, device=device)

        top_k = torch.tensor([10, 10], device=device)
        si = SamplingInputs(max_top_k=10, top_k=top_k)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si, draft_probs=draft_probs)

        # Reject at pos 0 -> 1 recovered token per row
        valid = _extract_valid(out)
        for i in range(2):
            assert len(valid[i]) == 1, f'row {i}: expected 1 token, got {valid[i]}'

    # ----- _extract_outputs -----

    def test_extract_outputs(self):
        """Test _extract_outputs helper function."""
        output_token_ids = torch.tensor([
            [5, 6, 7, 8],  # all valid
            [10, 11, -1, -1],  # 2 valid
        ])
        out, rej, last = _extract_outputs(output_token_ids, num_spec_tokens=3)
        assert rej.tolist() == [0, 2]
        assert last.tolist() == [8, 11]


# ---------------------------------------------------------------------------
# Triton kernel tests (direct kernel invocations)
# ---------------------------------------------------------------------------


class TestTritonKernels:

    def test_greedy_kernel_mixed(self):
        """rejection_greedy_sample_kernel with mixed match/mismatch rows.

        Row 0: all match       -> [10, 20, 30, 40, 50]
        Row 1: mismatch at 2   -> [10, 20, 30, -1, -1]
        Row 2: mismatch at 0   -> [10, -1, -1, -1, -1]
        """
        batch_size, num_spec = 3, 4
        target_argmax = torch.tensor([[10, 20, 30, 40]] * 3, dtype=torch.long, device=device)
        draft = torch.tensor([
            [10, 20, 30, 40],
            [10, 20, 99, 40],
            [99, 20, 30, 40],
        ], dtype=torch.long, device=device)
        bonus = torch.tensor([50, 51, 52], dtype=torch.long, device=device)
        out = torch.full((batch_size, num_spec + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.long, device=device)

        rejection_greedy_sample_kernel[(batch_size, )](out, draft, target_argmax, bonus, None, num_spec)

        assert out[0].tolist() == [10, 20, 30, 40, 50]
        assert out[1].tolist() == [10, 20, 30, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]
        assert out[2].tolist() == [
            10, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID
        ]

    def test_greedy_kernel_with_is_greedy_mask(self):
        """rejection_greedy_sample_kernel skips non-greedy rows.

        Row 0: greedy, all match -> [10, 20, 30, 50]
        Row 1: not greedy        -> all placeholder (untouched)
        """
        batch_size, num_spec = 2, 3
        target_argmax = torch.tensor([[10, 20, 30]] * 2, dtype=torch.long, device=device)
        draft = target_argmax.clone()
        bonus = torch.tensor([50, 51], dtype=torch.long, device=device)
        out = torch.full((batch_size, num_spec + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.long, device=device)
        is_greedy = torch.tensor([True, False], dtype=torch.bool, device=device)

        rejection_greedy_sample_kernel[(batch_size, )](out, draft, target_argmax, bonus, is_greedy, num_spec)

        assert out[0].tolist() == [10, 20, 30, 50]
        assert (out[1] == PLACEHOLDER_TOKEN_ID).all()

    def test_recovered_tokens_with_draft_probs(self):
        """sample_recovered_tokens_kernel: recovered token = argmax of
        max(0, target - draft) * inv_q.

        target = [0, 0, 0, 0.9, 0, 0.1, 0, 0]
        draft  = [0, 0, 0, 0.9, 0, 0,   0, 0]
        diff   = [0, 0, 0, 0,   0, 0.1, 0, 0]  -> token 5
        """
        batch_size, num_spec, vocab = 2, 2, 8
        target_probs = torch.zeros(batch_size, num_spec, vocab, device=device, dtype=torch.float32)
        target_probs[:, :, 3] = 0.9
        target_probs[:, :, 5] = 0.1

        draft_probs = torch.zeros_like(target_probs)
        draft_probs[:, :, 3] = 0.9

        draft_ids = torch.full((batch_size, num_spec), 3, dtype=torch.long, device=device)
        inv_q = torch.ones(batch_size, vocab, device=device, dtype=torch.float32)
        recovered = torch.empty(batch_size, num_spec, dtype=torch.long, device=device)

        sample_recovered_tokens_kernel[(batch_size, num_spec)](recovered,
                                                               draft_ids,
                                                               draft_probs,
                                                               target_probs,
                                                               inv_q,
                                                               num_spec,
                                                               vocab,
                                                               8,
                                                               NO_DRAFT_PROBS=False)

        assert (recovered == 5).all()

    def test_recovered_tokens_no_draft_probs(self):
        """sample_recovered_tokens_kernel with NO_DRAFT_PROBS: draft token
        is masked out, recovered = argmax of remaining target probs.

        target   = [0, 0, 0.6, 0, 0, 0.3, 0, 0.1]
        draft_id = 2 (masked out)
        remaining= [0, 0, 0,   0, 0, 0.3, 0, 0.1]  -> token 5
        """
        batch_size, num_spec, vocab = 2, 2, 8
        target_probs = torch.zeros(batch_size, num_spec, vocab, device=device, dtype=torch.float32)
        target_probs[:, :, 2] = 0.6
        target_probs[:, :, 5] = 0.3
        target_probs[:, :, 7] = 0.1

        draft_ids = torch.full((batch_size, num_spec), 2, dtype=torch.long, device=device)
        inv_q = torch.ones(batch_size, vocab, device=device, dtype=torch.float32)
        recovered = torch.empty(batch_size, num_spec, dtype=torch.long, device=device)

        sample_recovered_tokens_kernel[(batch_size, num_spec)](recovered,
                                                               draft_ids,
                                                               None,
                                                               target_probs,
                                                               inv_q,
                                                               num_spec,
                                                               vocab,
                                                               8,
                                                               NO_DRAFT_PROBS=True)

        assert (recovered == 5).all()
