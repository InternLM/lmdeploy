# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.backends.default.rejection_sampling import DefaultRejectionSamplingImpl
from lmdeploy.pytorch.engine.logits_process import SamplingInputsDelta
from lmdeploy.pytorch.strategies.ar.step_inputs import step_sampling_delta


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_rejection_rng_survives_rebatching_and_unrelated_sampling(device):
    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('requires CUDA')
        from lmdeploy.pytorch.backends.cuda.rejection_sampling import CudaRejectionSamplingImpl
        sampler = CudaRejectionSamplingImpl()
    else:
        sampler = DefaultRejectionSamplingImpl()
    torch.manual_seed(7)
    logits = torch.randn(16, 5, 127, device=device)
    drafts = torch.randint(127, (16, 5), device=device)
    bonus = torch.arange(16, device=device)
    seeds = torch.arange(16, device=device) * 100003 + 42
    offsets = torch.arange(16, device=device) + 32769

    def sample(rows):
        return sampler.forward(logits[rows], drafts[rows], bonus[rows], is_greedy=None,
                               random_seeds=seeds[rows], random_offsets=offsets[rows])

    expected = sample(slice(None))
    torch.rand(4096, device=device)
    order = torch.arange(15, -1, -1, device=device)
    for actual, wanted in zip(sample(order), expected):
        torch.testing.assert_close(actual, wanted[order], rtol=0, atol=0)
    for actual, wanted in zip(sample(slice(4, 5)), expected):
        torch.testing.assert_close(actual, wanted[4:5], rtol=0, atol=0)


def test_request_rng_preserves_target_distribution():
    sampler = DefaultRejectionSamplingImpl()
    n = 12000
    probabilities = torch.tensor([0.2, 0.3, 0.5])
    output, _, _ = sampler.forward(probabilities.log().expand(n, 1, 3).contiguous(),
                                   torch.zeros(n, 1, dtype=torch.long), torch.zeros(n, dtype=torch.long),
                                   is_greedy=None, random_seeds=torch.arange(n),
                                   random_offsets=torch.full((n,), 65))
    observed = output[:, 0].bincount(minlength=3).float() / n
    torch.testing.assert_close(observed, probabilities, atol=0.015, rtol=0)


def test_sampling_progress_counts_accepted_outputs_not_verification_steps():
    delta = SamplingInputsDelta(num_ignore_eos=torch.tensor([5, 5]),
                                random_offsets=torch.tensor([65, 65]))
    step_sampling_delta(delta, torch.tensor([10, 11]), random_offset_step=torch.tensor([6, 2]))
    assert delta.random_offsets.tolist() == [71, 67]
    assert delta.num_ignore_eos.tolist() == [-1, 3]
