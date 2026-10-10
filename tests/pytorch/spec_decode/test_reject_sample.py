from dataclasses import replace

import pytest
import torch

from lmdeploy.pytorch.backends.default.rejection_sampling import (
    DefaultRejectionSamplingImpl,
    torch_greedy_rejection_sample,
)
from lmdeploy.pytorch.backends.rejection_sampling import PLACEHOLDER_TOKEN_ID
from lmdeploy.pytorch.engine.logits_process import FusedLogitsProcessor, SamplingInputs
from lmdeploy.pytorch.spec_decode.reject_sampler import RejectionSampler
from lmdeploy.pytorch.spec_decode.spec_agent import _expand_sampling_inputs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_rejection_sampler = (RejectionSampler('cuda')
                      if torch.cuda.is_available() else None)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


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


@pytest.mark.parametrize('backend', ['cuda', 'default'])
@pytest.mark.parametrize('top_k,top_p,min_p,use_draft_probs,kept',
                         [(2, 1., 0., False, 2), (0, .65, 0., False, 2), (0, 1., .6, True, 2),
                          (0, 1., 0., False, 4)])
def test_filtered_target_distribution(backend, top_k, top_p, min_p, use_draft_probs, kept):
    """Both accepted drafts and recovered tokens must follow the filtered target."""
    batch_size = 8192
    probs = torch.tensor([.4, .3, .2, .1], device=device)
    logits = probs.log().expand(batch_size, 3, 4).contiguous()
    draft = torch.arange(batch_size, device=device).remainder(4)[:, None].expand(-1, 2).contiguous()
    policy = SamplingInputs(batch_size=batch_size, max_top_k=top_k,
                            top_p=torch.full((batch_size,), top_p, device=device) if top_p < 1 else None,
                            min_p=torch.full((batch_size,), min_p, device=device) if min_p > 0 else None,
                            random_seeds=torch.arange(batch_size, device=device),
                            random_offsets=torch.full((batch_size,), 100, device=device))
    sampler = RejectionSampler('cuda')
    if backend == 'default':
        sampler._impl = DefaultRejectionSamplingImpl()
    draft_probs = torch.full_like(logits[:, :-1], .25) if use_draft_probs else None
    output, _, _ = sampler(logits, draft, _expand_sampling_inputs(policy, 3), draft_probs=draft_probs)
    valid = output[output >= 0]
    assert (valid < kept).all()
    expected = probs.clone()
    expected[kept:] = 0
    expected /= expected.sum()
    actual = torch.bincount(output[:, 0], minlength=4) / batch_size
    torch.testing.assert_close(actual, expected, atol=.025, rtol=0)


@pytest.mark.parametrize('backend', ['cuda', 'default'])
def test_request_randomness_isolated(backend):
    """Request RNG is independent of global draws, row order, and batch size."""
    batch_size = 128
    logits = torch.tensor([.55, .25, .15, .05], device=device).log().expand(batch_size, 5, 4).contiguous()
    draft = torch.zeros(batch_size, 4, dtype=torch.long, device=device)
    top_k = torch.tensor([1, 4], device=device).repeat(batch_size // 2)
    policy = SamplingInputs(batch_size=batch_size, max_top_k=4, has_greedy=True, top_k=top_k,
                            random_seeds=torch.arange(batch_size, device=device) + 42,
                            random_offsets=torch.full((batch_size,), 100, device=device))
    sampler = RejectionSampler('cuda')
    if backend == 'default':
        sampler._impl = DefaultRejectionSamplingImpl()

    def sample(inputs, rows=slice(None)):
        return sampler(logits[rows], draft[rows], _expand_sampling_inputs(inputs, 5))

    expected = sample(policy)
    assert (expected[0][::2] == 0).all()
    torch.rand(4096, device=device)
    for actual, reference in zip(sample(policy), expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    rows = torch.arange(batch_size - 1, 0, -2, device=device)
    subset = replace(policy, batch_size=len(rows), top_k=top_k[rows], random_seeds=policy.random_seeds[rows],
                     random_offsets=policy.random_offsets[rows])
    for actual, reference in zip(sample(subset, rows), expected):
        torch.testing.assert_close(actual, reference[rows], atol=0, rtol=0)
    for changed in (replace(policy, random_seeds=policy.random_seeds + 1000),
                    replace(policy, random_offsets=policy.random_offsets + 1)):
        assert (sample(changed)[0][:, 0] != expected[0][:, 0]).any()


def test_bonus_filter_applied_once():
    """Reapplying top-p to an already filtered bonus would remove token 1."""
    batch_size = 1024
    policy = SamplingInputs(batch_size=batch_size, max_top_k=3,
                            top_p=torch.full((batch_size,), .7, device=device),
                            random_seeds=torch.arange(batch_size, device=device),
                            random_offsets=torch.zeros(batch_size, dtype=torch.long, device=device))
    logits = torch.tensor([[0., -torch.inf, -torch.inf], [.6, .25, .15]], device=device)
    logits[1] = logits[1].log()
    logits = logits.expand(batch_size, -1, -1).contiguous()
    expanded = _expand_sampling_inputs(policy, 2)
    output, rejected, _ = _rejection_sampler(logits, torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                                           expanded)
    expected = FusedLogitsProcessor(replace(policy, random_offsets=policy.random_offsets + 1)).sampling(logits[:, -1])
    assert (rejected == 0).all()
    assert (expected == 1).any()
    torch.testing.assert_close(output[:, -1], expected, atol=0, rtol=0)


def test_request_random_streams_match_cpu_reference():
    from lmdeploy.pytorch.backends.default.rejection_sampling import rejection_sampling_random as reference
    from lmdeploy.pytorch.kernels.cuda.rejection_sampling import rejection_sampling_random

    seeds = torch.arange(16, device=device)[::2]
    offsets = torch.arange(16, device=device)[::2] + (1 << 28)
    actual = rejection_sampling_random(seeds, offsets, 4, 257)
    expected = reference(seeds.cpu(), offsets.cpu(), 4, 257)
    for cuda_value, cpu_value in zip(actual, expected):
        torch.testing.assert_close(cuda_value.cpu(), cpu_value)
