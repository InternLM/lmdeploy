import torch

from lmdeploy.pytorch.spec_decode.spec_agent import _expand_sampling_inputs

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_slice_sampling_inputs_decode():
    """Test _slice_sampling_inputs with decoding (num_tokens_per_batch > 1)."""
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs
    from lmdeploy.pytorch.spec_decode.spec_agent import _slice_sampling_inputs

    batch_size = 2
    num_tokens_per_batch = 3

    temperature = torch.tensor([0.5, 1.0], device=device)
    top_k = torch.tensor([1, 10], device=device)
    random_offsets = torch.tensor([100, 200], device=device)

    sampling_inputs = SamplingInputs(
        max_top_k=10,
        top_k=top_k,
        temperature=temperature,
        random_offsets=random_offsets,
        max_num_logprobs=-1,
        batch_size=batch_size,
    )

    # First expand
    expanded = _expand_sampling_inputs(sampling_inputs, num_tokens_per_batch)
    assert expanded.batch_size == batch_size * num_tokens_per_batch
    # random_offsets should be offset by arange per batch element
    # batch 0: [100, 101, 102], batch 1: [200, 201, 202]
    expected_offsets = torch.tensor([100, 101, 102, 200, 201, 202], device=device)
    torch.testing.assert_close(expanded.random_offsets, expected_offsets)

    # Then slice back (is_last=True, takes last token per batch)
    sliced = _slice_sampling_inputs(expanded, num_tokens_per_batch)
    assert sliced.batch_size == batch_size
    torch.testing.assert_close(sliced.temperature, temperature)
    torch.testing.assert_close(sliced.top_k, top_k)
    assert sliced.max_top_k == 10
    # last token per batch: offsets [102, 202]
    torch.testing.assert_close(sliced.random_offsets, torch.tensor([102, 202], device=device))

    # Slice with is_last=False (takes tokens except the last one per batch)
    sliced_draft = _slice_sampling_inputs(expanded, num_tokens_per_batch, is_last=False)
    assert sliced_draft.batch_size == batch_size * (num_tokens_per_batch - 1)
    # drops last per batch: [100, 101, 200, 201]
    torch.testing.assert_close(sliced_draft.random_offsets, torch.tensor([100, 101, 200, 201], device=device))


def test_slice_sampling_inputs_prefill():
    """Test _slice_sampling_inputs with prefill (num_tokens_per_batch=1 returns
    same object)."""
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs
    from lmdeploy.pytorch.spec_decode.spec_agent import _slice_sampling_inputs

    sampling_inputs = SamplingInputs(max_top_k=1, batch_size=2)
    result = _slice_sampling_inputs(sampling_inputs, 1)
    assert result is sampling_inputs
