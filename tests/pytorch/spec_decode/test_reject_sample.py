from dataclasses import replace

import pytest
import torch

from lmdeploy.pytorch.backends.default.rejection_sampling import (
    DefaultRejectionSamplingImpl,
    torch_greedy_rejection_sample,
)
from lmdeploy.pytorch.backends.rejection_sampling import PLACEHOLDER_TOKEN_ID
from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.spec_decode.reject_sampler import RejectionSampler
from lmdeploy.pytorch.spec_decode.spec_agent import _expand_sampling_inputs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_rejection_sampler = (RejectionSampler('cuda')
                      if torch.cuda.is_available() else None)


def rejection_sample(target_logits,
                     draft_token_ids,
                     bonus_token_ids,
                     sampling_inputs,
                     draft_probs=None):
    batch_size, num_spec_tokens, vocab_size = target_logits.shape
    bonus_logits = target_logits.new_full((batch_size, 1, vocab_size), -100.0)
    bonus_logits.scatter_(2, bonus_token_ids[:, None, None], 100.0)
    target_logits = torch.cat([target_logits, bonus_logits], dim=1)

    updates = {'batch_size': batch_size}
    if sampling_inputs.max_top_k != 1:
        if sampling_inputs.random_seeds is None:
            updates['random_seeds'] = torch.arange(batch_size,
                                                   dtype=torch.long,
                                                   device=target_logits.device)
        if sampling_inputs.random_offsets is None:
            updates['random_offsets'] = torch.zeros(batch_size,
                                                    dtype=torch.long,
                                                    device=target_logits.device)
    sampling_inputs = replace(sampling_inputs, **updates)
    expanded_sampling_inputs = _expand_sampling_inputs(
        sampling_inputs,
        num_spec_tokens + 1,
    )
    return _rejection_sampler(target_logits,
                              draft_token_ids,
                              expanded_sampling_inputs,
                              draft_probs=draft_probs)


def torch_greedy_reference(target_logits, draft_token_ids, bonus_token_ids):
    target_token_ids = torch.cat(
        [target_logits.argmax(dim=-1), bonus_token_ids[:, None]],
        dim=1,
    )
    return torch_greedy_rejection_sample(target_token_ids, draft_token_ids)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
class TestRejectSample:
    """Tests for rejection_sample and torch_greedy_rejection_sample.

    Uses hand-crafted token ids so the expected output is obvious.
    """

    # ----- greedy: rejection_sample -----

    def test_greedy_all_match(self):
        """All draft == target -> accept all + bonus."""
        target_logits = _make_peaked_logits([[10, 20, 30], [10, 20, 30]], 64)
        draft = torch.tensor([[10, 20, 30], [10, 20, 30]], dtype=torch.long, device=device)
        bonus = torch.tensor([50, 51], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert out.shape == (2, 4)
        assert (rej == 0).all()
        assert _extract_valid(out) == [[10, 20, 30, 50], [10, 20, 30, 51]]
        assert last.tolist() == [50, 51]

    def test_greedy_first_mismatch(self):
        """Mismatch at position 0 -> keep only target[0]."""
        target_logits = _make_peaked_logits([[10, 20, 30], [10, 20, 30]], 64)
        draft = torch.tensor([[11, 20, 30], [11, 20, 30]], dtype=torch.long, device=device)
        bonus = torch.tensor([50, 51], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert (rej == 3).all()
        assert _extract_valid(out) == [[10], [10]]
        assert last.tolist() == [10, 10]

    def test_greedy_middle_mismatch(self):
        """Mismatch at position 2 -> keep target[0:3], no bonus."""
        target_logits = _make_peaked_logits([[10, 20, 30], [10, 20, 30]], 64)
        draft = torch.tensor([[10, 20, 31], [10, 20, 31]], dtype=torch.long, device=device)
        bonus = torch.tensor([50, 51], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert (rej == 1).all()
        assert _extract_valid(out) == [[10, 20, 30], [10, 20, 30]]
        assert last.tolist() == [30, 30]

    def test_greedy_mixed_mismatch_positions(self):
        """Each row has a different first-mismatch position.

        Row 0: all match       -> [10, 20, 30, 50]
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
        bonus = torch.tensor([50, 51, 52, 53], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        assert rej.tolist() == [0, 3, 2, 1]
        assert _extract_valid(out) == [
            [10, 20, 30, 50],
            [10],
            [10, 20],
            [10, 20, 30],
        ]
        assert last.tolist() == [50, 10, 20, 30]

    def test_greedy_all_rejected(self):
        """Every draft wrong -> only first target token kept."""
        target_logits = _make_peaked_logits([[10, 20, 30], [10, 20, 30]], 64)
        draft = torch.tensor([[11, 21, 31], [11, 21, 31]], dtype=torch.long, device=device)
        bonus = torch.tensor([50, 51], dtype=torch.long, device=device)
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
        bonus = torch.tensor([50, 51, 52, 53], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        t_out, t_rej, t_last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)
        p_out, p_rej, p_last = torch_greedy_reference(target_logits, draft, bonus)

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
        bonus = torch.tensor([50, 51, 52, 53], dtype=torch.long, device=device)
        si = SamplingInputs(max_top_k=1)

        t_out, t_rej, t_last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)
        p_out, p_rej, p_last = torch_greedy_reference(target_logits, draft, bonus)

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
        p_out, p_rej, p_last = torch_greedy_reference(target_logits, draft, bonus)

        assert torch.equal(t_out, p_out)
        assert torch.equal(t_rej, p_rej)
        assert torch.equal(t_last, p_last)

    # ----- mixed greedy/random -----

    def test_mixed_batch(self):
        """Mixed batch: greedy (top_k=1) and random sequences."""
        target_logits = _make_peaked_logits([[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]], 32)
        draft = target_logits.argmax(dim=-1)
        bonus = torch.tensor([20, 21, 22, 23], dtype=torch.long, device=device)

        top_k = torch.tensor([1, 10, 1, 10], device=device)
        si = SamplingInputs(max_top_k=10, has_greedy=True, top_k=top_k)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si)

        # Greedy rows (0, 2) must accept all since draft == target
        assert rej[0] == 0
        assert rej[2] == 0
        assert last[0] == 20
        assert last[2] == 22

    # ----- random: no draft_probs (MTP/n-gram) -----

    def test_random_no_draft_probs(self):
        """Random sampling with draft_probs=None (MTP/n-gram mode)."""
        target_logits = _make_peaked_logits([[5, 6, 7], [5, 6, 7]], 32)
        draft = torch.tensor([[5, 6, 7], [5, 6, 7]], dtype=torch.long, device=device)
        bonus = torch.tensor([20, 21], dtype=torch.long, device=device)

        top_k = torch.tensor([10, 10], device=device)
        si = SamplingInputs(max_top_k=10, has_greedy=False, top_k=top_k)

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
        bonus = torch.tensor([20, 21], dtype=torch.long, device=device)

        top_k = torch.tensor([10, 10], device=device)
        si = SamplingInputs(max_top_k=10, has_greedy=False, top_k=top_k)

        out, rej, last = rejection_sample(target_logits, draft, bonus, sampling_inputs=si, draft_probs=draft_probs)

        # Reject at pos 0 -> 1 recovered token per row
        valid = _extract_valid(out)
        for i in range(2):
            assert len(valid[i]) == 1, f'row {i}: expected 1 token, got {valid[i]}'

    @pytest.mark.parametrize('all_accepted', [False, True])
    def test_request_seed_survives_rng_interference_and_batch_changes(self, all_accepted):
        batch, width, vocab = 16, 4, 32
        logits = torch.linspace(-2, 2, vocab, device=device).expand(batch, width, -1).contiguous()
        drafts = torch.full((batch, width - 1), vocab - 1, device=device, dtype=torch.long)
        if all_accepted:
            logits[:, :-1] = -float('inf')
            logits[:, :-1, vocab - 1] = 0
        rows = torch.arange(batch, device=device)

        def sample(indices, seed_delta=0, offset_delta=0):
            inputs = SamplingInputs(
                batch_size=len(indices), max_top_k=0, has_greedy=True,
                top_k=torch.where(indices % 4 == 0, 1, 0),
                random_seeds=indices + 1234 + seed_delta,
                random_offsets=indices + 100 + offset_delta)
            return _rejection_sampler(logits[indices], drafts[indices], _expand_sampling_inputs(inputs, width))

        # Includes the bonus sampler and the strided per-request RNG inputs.
        rng_state = torch.cuda.get_rng_state()
        expected = sample(rows)
        if all_accepted:
            assert (expected[1] == 0).all()
            torch.testing.assert_close(expected[0][:, :-1], drafts)
        assert torch.equal(torch.cuda.get_rng_state(), rng_state)
        torch.rand(4096, device=device)
        for indices in (rows, rows.flip(0), rows[::2], rows[1:2]):
            for actual, reference in zip(sample(indices), expected):
                torch.testing.assert_close(actual, reference[indices], rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = sample(rows)
        graph.replay()
        for actual, reference in zip(captured, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert not torch.equal(sample(rows, seed_delta=1)[0], expected[0])
        assert not torch.equal(sample(rows, offset_delta=1)[0], expected[0])

    @pytest.mark.parametrize('has_draft_probs', [False, True])
    def test_seeded_cuda_matches_portable_rejection(self, has_draft_probs):
        from lmdeploy.pytorch.backends.cuda.rejection_sampling import CudaRejectionSamplingImpl

        torch.manual_seed(42)
        logits = torch.randn(32, 3, 4099, device=device)
        drafts = torch.randint(4099, (32, 3), device=device)
        bonus = torch.randint(4099, (32,), device=device)
        # Force different accepted-prefix lengths; recovery spans multiple vocabulary blocks.
        for pos in range(3):
            rows = torch.arange(32, device=device)[torch.arange(32, device=device) % 4 > pos]
            logits[rows, pos] = -1000
            logits[rows, pos, drafts[rows, pos]] = 1000
        kwargs = dict(
            is_greedy=torch.arange(32, device=device) % 5 == 0,
            random_seeds=torch.arange(32, device=device) + 1234,
            random_offsets=torch.arange(32, device=device) + 2**32,
            draft_probs=torch.randn_like(logits).softmax(-1) if has_draft_probs else None)
        expected = DefaultRejectionSamplingImpl().forward(logits, drafts, bonus, **kwargs)
        actual = CudaRejectionSamplingImpl().forward(logits, drafts, bonus, **kwargs)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize('backend, arch, warp_size', [('cuda', 90, 32), ('hip', 'gfx942', 64)])
def test_recovery_kernel_compiles(backend, arch, warp_size):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from lmdeploy.pytorch.kernels.cuda.rejection_sampling import _sample_recovered_tokens_kernel

    signature = dict(
        partial_scores_ptr='*fp32', partial_tokens_ptr='*i64', num_rejected_tokens_ptr='*i64',
        draft_token_ids_ptr='*i64', draft_probs_ptr='*fp32', target_probs_ptr='*fp32',
        random_seeds_ptr='*i64', random_offsets_ptr='*i64', is_greedy_ptr='*i1',
        seed_stride='i32', offset_stride='i32', num_spec_tokens='i32', vocab_size='i32', num_vocab_blocks='i32')
    source = ASTSource(_sample_recovered_tokens_kernel, signature,
                       constexprs=dict(BLOCK_SIZE=1024, NO_DRAFT_PROBS=True))
    triton.compile(source, target=GPUTarget(backend, arch, warp_size))


@pytest.mark.parametrize('backend', ['default', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA'))])
def test_seeded_rejection_preserves_target_distribution(backend):
    from lmdeploy.pytorch.backends.cuda.rejection_sampling import CudaRejectionSamplingImpl

    sample_device = 'cpu' if backend == 'default' else 'cuda'
    impl = DefaultRejectionSamplingImpl() if backend == 'default' else CudaRejectionSamplingImpl()
    num_trials = 10000
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4], device=sample_device)
    logits = probabilities.log().repeat(num_trials, 1, 1)
    drafts = torch.full((num_trials, 1), 2, dtype=torch.long, device=sample_device)
    output, rejected, _ = impl.forward(
        logits, drafts, drafts[:, 0], is_greedy=None,
        random_seeds=torch.arange(num_trials, device=sample_device),
        random_offsets=torch.full((num_trials,), 100, device=sample_device))
    frequencies = torch.bincount(output[:, 0], minlength=4).float() / num_trials
    torch.testing.assert_close(frequencies, probabilities, rtol=0, atol=0.02)
    assert abs((rejected == 0).float().mean().item() - 0.3) < 0.02

    # This seed rounds the recovery uniform for token 1 to 1 in fp32 without clamping.
    logits = torch.tensor([[[-float('inf'), 0.0, -float('inf')]]], device=sample_device)
    drafts = torch.zeros((1, 1), dtype=torch.long, device=sample_device)
    output, _, last = impl.forward(
        logits, drafts, drafts[:, 0], is_greedy=None,
        random_seeds=torch.tensor([5249637], device=sample_device),
        random_offsets=torch.zeros(1, dtype=torch.long, device=sample_device))
    assert output.tolist() == [[1, PLACEHOLDER_TOKEN_ID]]
    assert last.item() == 1
