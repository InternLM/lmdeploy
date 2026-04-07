import asyncio

import pytest
import torch

from lmdeploy.pytorch.spec_decode.spec_agent import _expand_sampling_inputs

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _run_async(coro):
    """Helper to run async function in sync test."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_spec_agent_for_sampling(misc_config=None):
    """Create a minimal SpecModelAgent with only the fields needed for
    async_sampling_logits."""
    from lmdeploy.pytorch.config import MiscConfig
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

    agent = object.__new__(SpecModelAgent)
    agent.misc_config = misc_config or MiscConfig()
    return agent


def test_async_sampling_logits_greedy_prefill():
    """Test async_sampling_logits with greedy sampling, prefill (1
    position)."""
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs

    agent = _make_spec_agent_for_sampling()

    batch_size = 2
    num_tokens_per_batch = 1  # prefill
    vocab_size = 32

    target_logits = torch.randn(batch_size, num_tokens_per_batch, vocab_size, device=device)
    sampling_inputs = SamplingInputs(
        max_top_k=1,
        max_num_logprobs=-1,
        logits_processors=[],
        batch_size=batch_size,
    )
    # num_tokens_per_batch=1, no expansion needed

    processed, next_ids, logprobs = _run_async(agent.async_sampling_logits(target_logits, sampling_inputs))

    # Output shapes
    assert processed.shape == (batch_size, num_tokens_per_batch, vocab_size)
    assert next_ids.shape == (batch_size, )
    # Greedy: next_ids should be argmax of bonus (only) position
    expected = target_logits[:, -1, :].argmax(dim=-1)
    torch.testing.assert_close(next_ids, expected)
    # No logprobs requested
    assert logprobs is None


def test_async_sampling_logits_greedy_decoding():
    """Test async_sampling_logits with greedy sampling, decoding (multiple
    positions)."""
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs

    agent = _make_spec_agent_for_sampling()

    batch_size = 3
    num_spec_tokens = 4
    num_tokens_per_batch = 1 + num_spec_tokens  # decoding
    vocab_size = 64

    target_logits = torch.randn(batch_size, num_tokens_per_batch, vocab_size, device=device)
    sampling_inputs = SamplingInputs(
        max_top_k=1,
        max_num_logprobs=-1,
        logits_processors=[],
        batch_size=batch_size,
    )
    # Expand for decoding
    expanded = _expand_sampling_inputs(sampling_inputs, num_tokens_per_batch)

    processed, next_ids, logprobs = _run_async(agent.async_sampling_logits(target_logits, expanded))

    assert processed.shape == (batch_size, num_tokens_per_batch, vocab_size)
    assert next_ids.shape == (batch_size, )
    # Greedy: bonus token is argmax of last position
    expected = target_logits[:, -1, :].argmax(dim=-1)
    torch.testing.assert_close(next_ids, expected)
    assert logprobs is None


def test_async_sampling_logits_random():
    """Test async_sampling_logits with random sampling."""
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs

    agent = _make_spec_agent_for_sampling()

    batch_size = 2
    num_tokens_per_batch = 4
    vocab_size = 32

    target_logits = torch.randn(batch_size, num_tokens_per_batch, vocab_size, device=device)
    temperature = torch.ones(batch_size, device=device)
    top_k = torch.full((batch_size, ), 10, device=device)
    random_seeds = torch.randint(0, 2**31, (batch_size, ), dtype=torch.long, device=device)
    random_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)

    sampling_inputs = SamplingInputs(
        max_top_k=10,
        top_k=top_k,
        temperature=temperature,
        random_seeds=random_seeds,
        random_offsets=random_offsets,
        max_num_logprobs=-1,
        logits_processors=[],
        batch_size=batch_size,
    )
    expanded = _expand_sampling_inputs(sampling_inputs, num_tokens_per_batch)

    processed, next_ids, logprobs = _run_async(agent.async_sampling_logits(target_logits, expanded))

    assert processed.shape == (batch_size, num_tokens_per_batch, vocab_size)
    assert next_ids.shape == (batch_size, )
    # Token ids should be valid
    assert (next_ids >= 0).all()
    assert (next_ids < vocab_size).all()
    assert logprobs is None


def test_async_sampling_logits_with_logprobs():
    """Test async_sampling_logits returns logprobs when requested."""
    from lmdeploy.pytorch.config import MiscConfig
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs

    misc_config = MiscConfig(logprobs_mode='raw_logprobs')
    agent = _make_spec_agent_for_sampling(misc_config=misc_config)

    batch_size = 2
    num_tokens_per_batch = 3
    vocab_size = 32

    target_logits = torch.randn(batch_size, num_tokens_per_batch, vocab_size, device=device)
    sampling_inputs = SamplingInputs(
        max_top_k=1,
        max_num_logprobs=5,
        logits_processors=[],
        batch_size=batch_size,
    )
    expanded = _expand_sampling_inputs(sampling_inputs, num_tokens_per_batch)

    processed, next_ids, logprobs = _run_async(agent.async_sampling_logits(target_logits, expanded))

    assert processed.shape == (batch_size, num_tokens_per_batch, vocab_size)
    assert next_ids.shape == (batch_size, )
    assert logprobs is not None
    # raw_logprobs shape: [batch_size * num_tokens_per_batch, vocab_size]
    assert logprobs.shape == (batch_size * num_tokens_per_batch, vocab_size)


def test_async_sampling_logits_temperature():
    """Test that temperature scaling is applied correctly."""
    from lmdeploy.pytorch.engine.logits_process import SamplingInputs

    agent = _make_spec_agent_for_sampling()

    batch_size = 1
    num_tokens_per_batch = 1
    vocab_size = 16

    # Strong logits: token 0 = 100, rest = 0
    target_logits = torch.zeros(batch_size, num_tokens_per_batch, vocab_size, device=device)
    target_logits[0, 0, 0] = 100.0
    original_val = target_logits[0, 0, 0].item()

    temperature = torch.tensor([0.5], device=device)
    sampling_inputs = SamplingInputs(
        max_top_k=1,
        temperature=temperature,
        max_num_logprobs=-1,
        logits_processors=[],
        batch_size=batch_size,
    )
    # num_tokens_per_batch=1, no expansion needed

    processed, next_ids, _ = _run_async(agent.async_sampling_logits(target_logits, sampling_inputs))

    # Temperature divides logits: 100/0.5 = 200 at position 0
    # Greedy should still pick token 0
    assert next_ids[0] == 0
    # Temperature-scaled logits should be 200
    assert processed[0, 0, 0].item() == pytest.approx(original_val / 0.5, rel=1e-3)


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
