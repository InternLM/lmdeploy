import asyncio

import torch

from lmdeploy.pytorch.spec_decode.spec_agent import _expand_sampling_inputs

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _make_non_last_chunk_inputs(dp_meta=None):
    from lmdeploy.pytorch.model_inputs import ModelInputs
    from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs

    batch_size = 2
    inputs = ModelInputs(
        input_ids=torch.zeros((1, batch_size), dtype=torch.long),
        seq_length=torch.ones(batch_size, dtype=torch.long),
        history_lengths=torch.zeros(batch_size, dtype=torch.long),
        block_offsets=torch.zeros((batch_size, 1), dtype=torch.long),
        is_decoding=True,
        num_ignored_history=torch.zeros(batch_size, dtype=torch.long),
        max_q_seqlen=1,
        max_kv_seqlen=1,
        sum_kv_seqlen=batch_size,
        dp_meta=dp_meta,
        is_chunk=True,
        is_first_chunk=False,
        is_last_chunk=False,
    )
    extra_inputs = ARSpecExtraInputs(
        next_token_ids=torch.zeros(batch_size, dtype=torch.long),
        last_token_indices=torch.zeros(batch_size, dtype=torch.long),
        num_rejected_tokens=torch.zeros(batch_size, dtype=torch.long),
        output_token_ids=torch.zeros((batch_size, 1), dtype=torch.long),
    )
    return inputs, extra_inputs


class _DummyDraftModel:

    class Meta:
        padding_batch_size = None

    def __init__(self):
        self.meta = self.Meta()
        self.update_inputs_calls = 0

    def get_meta(self):
        return self.meta

    def update_inputs(self, inputs):
        self.update_inputs_calls += 1
        return inputs


class _DummyProposer:

    def __init__(self):
        self.get_outputs_calls = 0
        self.update_inputs_decoding_calls = 0
        self.model = _DummyDraftModel()

    def get_outputs(self, outputs, inputs, extra_inputs=None):
        batch_size = inputs.seq_length.size(0)
        draft_token_ids = inputs.input_ids.new_full((batch_size, 1), self.get_outputs_calls)
        self.get_outputs_calls += 1
        return draft_token_ids, [{'call': self.get_outputs_calls}], inputs.target_hidden_states

    def update_inputs_decoding(self, inputs, extra_inputs, draft_token_ids, target_hidden_states, model_metas):
        self.update_inputs_decoding_calls += 1
        return inputs


def test_prepare_inputs_from_main_dp_non_last_first_chunk_shifts_last_token_indices():
    """DP non-last first chunks run draft forwards, so indices must match
    shifted draft inputs."""
    from lmdeploy.pytorch.model_inputs import DPMeta, ModelInputs
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent
    from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs

    agent = object.__new__(SpecModelAgent)
    agent._prev_chunk_last = {}
    agent.proposer = _DummyProposer()

    model_inputs = ModelInputs(
        input_ids=torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
        seq_length=torch.tensor([4], dtype=torch.long),
        history_lengths=torch.tensor([0], dtype=torch.long),
        block_offsets=torch.zeros((1, 1), dtype=torch.long),
        is_decoding=False,
        num_ignored_history=torch.zeros(1, dtype=torch.long),
        max_q_seqlen=4,
        max_kv_seqlen=4,
        sum_kv_seqlen=4,
        dp_meta=DPMeta(dp_batches=[1, 32], is_decoding=False, dp_is_decoding=False),
        is_chunk=True,
        is_first_chunk=True,
        is_last_chunk=False,
    )
    target_hidden_states = torch.arange(4 * 2, dtype=torch.float32).view(1, 4, 2)
    extra_inputs = ARSpecExtraInputs(
        next_token_ids=torch.tensor([0], dtype=torch.long),
        last_token_indices=torch.tensor([3], dtype=torch.long),
        target_hidden_states=target_hidden_states,
    )

    draft_inputs, draft_extra_inputs = agent._prepare_inputs_from_main(model_inputs, extra_inputs)

    torch.testing.assert_close(draft_inputs.input_ids, torch.tensor([[11, 12, 13]], dtype=torch.long))
    torch.testing.assert_close(draft_inputs.seq_length, torch.tensor([3], dtype=torch.long))
    assert draft_inputs.max_q_seqlen == 3
    assert draft_inputs.max_kv_seqlen == 3
    assert draft_inputs.sum_kv_seqlen == 3
    torch.testing.assert_close(draft_extra_inputs.last_token_indices, torch.tensor([2], dtype=torch.long))
    assert draft_extra_inputs.last_token_indices.max().item() < draft_inputs.input_ids.size(1)
    torch.testing.assert_close(draft_inputs.target_hidden_states, target_hidden_states[:, :-1])
    torch.testing.assert_close(agent._prev_chunk_last['hidden_states'], target_hidden_states[:, -1:])
    assert draft_inputs.dp_meta is model_inputs.dp_meta
    assert agent.proposer.model.update_inputs_calls == 1


def test_prepare_inputs_from_main_last_chunk_keeps_long_context_kv_metadata():
    """Last chunks keep aggregate KV metadata aligned after input rewriting."""
    from lmdeploy.pytorch.model_inputs import DPMeta, ModelInputs
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent
    from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs

    agent = object.__new__(SpecModelAgent)
    saved_hidden_states = torch.tensor([[[100.0, 101.0]]])
    agent._prev_chunk_last = {'hidden_states': saved_hidden_states}
    agent.proposer = _DummyProposer()

    long_kv_seqlen = 94218
    model_inputs = ModelInputs(
        input_ids=torch.tensor([[20, 21, 22]], dtype=torch.long),
        seq_length=torch.tensor([3], dtype=torch.long),
        history_lengths=torch.tensor([long_kv_seqlen - 3], dtype=torch.long),
        block_offsets=torch.zeros((1, 1), dtype=torch.long),
        is_decoding=False,
        num_ignored_history=torch.zeros(1, dtype=torch.long),
        max_q_seqlen=3,
        max_kv_seqlen=long_kv_seqlen,
        sum_kv_seqlen=long_kv_seqlen,
        dp_meta=DPMeta(dp_batches=[1, 32], is_decoding=False, dp_is_decoding=False),
        is_chunk=True,
        is_first_chunk=False,
        is_last_chunk=True,
    )
    target_hidden_states = torch.arange(3 * 2, dtype=torch.float32).view(1, 3, 2)
    extra_inputs = ARSpecExtraInputs(
        next_token_ids=torch.tensor([23], dtype=torch.long),
        last_token_indices=torch.tensor([2], dtype=torch.long),
        target_hidden_states=target_hidden_states,
    )

    draft_inputs, draft_extra_inputs = agent._prepare_inputs_from_main(model_inputs, extra_inputs)

    torch.testing.assert_close(draft_inputs.input_ids, torch.tensor([[20, 21, 22, 23]], dtype=torch.long))
    torch.testing.assert_close(draft_inputs.seq_length, torch.tensor([4], dtype=torch.long))
    assert draft_inputs.max_q_seqlen == 4
    assert draft_inputs.max_kv_seqlen == long_kv_seqlen
    assert draft_inputs.sum_kv_seqlen == long_kv_seqlen
    torch.testing.assert_close(draft_inputs.history_lengths, torch.tensor([long_kv_seqlen - 4], dtype=torch.long))
    torch.testing.assert_close(draft_inputs.seq_length + draft_inputs.history_lengths,
                               torch.tensor([long_kv_seqlen], dtype=torch.long))
    assert draft_inputs.sum_kv_seqlen == int((draft_inputs.seq_length + draft_inputs.history_lengths).sum())
    assert draft_inputs.max_kv_seqlen == int((draft_inputs.seq_length + draft_inputs.history_lengths).max())
    torch.testing.assert_close(draft_extra_inputs.last_token_indices, torch.tensor([3], dtype=torch.long))
    assert draft_extra_inputs.last_token_indices.max().item() < draft_inputs.input_ids.size(1)
    torch.testing.assert_close(draft_inputs.target_hidden_states,
                               torch.cat([saved_hidden_states, target_hidden_states], dim=1))
    assert 'hidden_states' not in agent._prev_chunk_last
    assert draft_inputs.dp_meta is model_inputs.dp_meta
    assert agent.proposer.model.update_inputs_calls == 1


def test_spec_model_agent_method_when_enabled():
    """Enabled SpecModelAgent should expose the configured spec method."""
    from lmdeploy.pytorch.config import DistConfig, SpecDecodeConfig
    from lmdeploy.pytorch.distributed import DistContext
    from lmdeploy.pytorch.spec_decode.base import BaseSpecModelAgent
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

    specdecode_config = SpecDecodeConfig(model='draft-model', method='mtp', num_speculative_tokens=3)
    agent = object.__new__(SpecModelAgent)
    BaseSpecModelAgent.__init__(
        agent,
        specdecode_config=specdecode_config,
        backend_config=None,
        inputs_strategy=None,
        agent_strategy=None,
        misc_config=None,
        dist_ctx=DistContext.build(dist_config=DistConfig()),
        device='cpu',
    )

    assert agent.is_enabled()
    assert agent.method == specdecode_config.method


def test_async_model_forward_dp1_non_last_chunk_skips_remaining_spec_forwards():
    """DP=1 non-last chunks should keep the local shortcut."""
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

    inputs, extra_inputs = _make_non_last_chunk_inputs()

    agent = object.__new__(SpecModelAgent)
    agent.num_spec_tokens = 3
    agent.rank = 0
    agent.proposer = _DummyProposer()
    forward_calls = 0

    def _forward_impl(_inputs):
        nonlocal forward_calls
        forward_calls += 1
        return {'call': forward_calls}

    agent._forward_impl = _forward_impl

    output = asyncio.run(agent._async_model_forward(inputs, extra_inputs, sampling_inputs=None))

    expected = torch.zeros((2, 3), dtype=torch.long)
    torch.testing.assert_close(output.output_draft_token_ids, expected)
    assert forward_calls == 1
    assert agent.proposer.get_outputs_calls == 0
    assert agent.proposer.update_inputs_decoding_calls == 0


def test_async_model_forward_dp_non_last_chunk_runs_all_spec_forwards(monkeypatch):
    """DP non-last chunks should still execute the full draft-forward loop."""
    import lmdeploy.pytorch.spec_decode.spec_agent as spec_agent_mod
    from lmdeploy.pytorch.model_inputs import DPMeta
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

    monkeypatch.setattr(spec_agent_mod.DPMeta, 'build', staticmethod(lambda seqlen, num_tokens: DPMeta()))
    inputs, extra_inputs = _make_non_last_chunk_inputs(dp_meta=DPMeta(dp_batches=[2, 2]))

    agent = object.__new__(SpecModelAgent)
    agent.num_spec_tokens = 3
    agent.rank = 0
    agent.proposer = _DummyProposer()
    forward_calls = 0

    def _forward_impl(_inputs):
        nonlocal forward_calls
        forward_calls += 1
        return {'call': forward_calls}

    agent._forward_impl = _forward_impl

    output = asyncio.run(agent._async_model_forward(inputs, extra_inputs, sampling_inputs=None))

    expected = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    torch.testing.assert_close(output.output_draft_token_ids, expected)
    assert forward_calls == agent.num_spec_tokens
    assert agent.proposer.get_outputs_calls == agent.num_spec_tokens
    assert agent.proposer.update_inputs_decoding_calls == 1
    assert agent.proposer.model.update_inputs_calls == agent.num_spec_tokens - 1


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
