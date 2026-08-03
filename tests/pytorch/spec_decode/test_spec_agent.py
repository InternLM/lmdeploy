import asyncio
import inspect
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.model_inputs import DPMeta, ModelInputs
from lmdeploy.pytorch.spec_decode.guided_spec_helper import GuidedSpecHelper
from lmdeploy.pytorch.spec_decode.proposers.base import (
    ProposalContext,
    ProposalMethod,
    ProposalWarmupCase,
    ProposalWarmupPlan,
)
from lmdeploy.pytorch.spec_decode.proposers.dflash import DFlash
from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent, _expand_sampling_inputs
from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs

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
        self.update_inputs_dp_is_decoding = []

    def get_meta(self):
        return self.meta

    def update_inputs(self, inputs):
        self.update_inputs_calls += 1
        if inputs.dp_meta is not None:
            self.update_inputs_dp_is_decoding.append(inputs.dp_meta.dp_is_decoding)
        return inputs


class _DummyProposer:

    proposal_method = ProposalMethod.AUTOREGRESSIVE

    def __init__(self):
        self.get_outputs_calls = 0
        self.update_inputs_decoding_calls = 0
        self.model = _DummyDraftModel()

    async def get_outputs(self, outputs, inputs, extra_inputs=None, guided_processors=None):
        batch_size = inputs.seq_length.size(0)
        draft_token_ids = inputs.input_ids.new_full((batch_size, 1), self.get_outputs_calls)
        self.get_outputs_calls += 1
        return draft_token_ids, [{'call': self.get_outputs_calls}], inputs.target_hidden_states

    def update_inputs_decoding(self, inputs, extra_inputs, draft_token_ids, target_hidden_states, model_metas):
        self.update_inputs_decoding_calls += 1
        batch_size = inputs.seq_length.size(0)
        return inputs.clone(
            input_ids=draft_token_ids,
            seq_length=inputs.seq_length.new_ones(batch_size),
            history_lengths=inputs.history_lengths + inputs.seq_length,
            is_decoding=True,
            max_q_seqlen=1,
            max_kv_seqlen=inputs.max_kv_seqlen + 1,
            sum_kv_seqlen=inputs.sum_kv_seqlen + batch_size,
            target_hidden_states=target_hidden_states,
            model_metas=model_metas,
        )


def test_guided_serial_bitmask_updates_inference_tensor():
    """Serial guided masking can update logits produced under inference
    mode."""

    class _Processor:

        def fork(self):
            return _Processor()

    class _GuidedManager:

        def __init__(self):
            self.accepted = []
            self.seen_is_contiguous = []
            self.seen_is_inference = []
            self.seen_inference_mode = []

        def allocate_batched_bitmap(self, batch_size):
            return torch.zeros(batch_size, 1, dtype=torch.int32)

        def fill_bitmap(self, processor, guided_bitmask, index):
            return None

        def apply_batched_bitmap(self, logits, guided_bitmask):
            self.seen_is_contiguous.append(logits.is_contiguous())
            self.seen_is_inference.append(logits.is_inference())
            self.seen_inference_mode.append(torch.is_inference_mode_enabled())
            logits[:, 0] = -123.0

        def is_terminated(self, processor):
            return False

        def accept_token(self, processor, token):
            self.accepted.append(token)

    manager = _GuidedManager()
    helper = GuidedSpecHelper(manager)
    batch_size = 2
    num_spec_tokens = 2
    num_expand = num_spec_tokens + 1
    vocab_size = 4
    with torch.inference_mode():
        scores_3d = torch.zeros(batch_size, num_expand, vocab_size)

    asyncio.run(
        helper.apply_serial_bitmask(
            scores_3d,
            processors={0: _Processor(), 1: _Processor()},
            draft_token_ids=torch.tensor([[10, 11], [20, 21]], dtype=torch.long),
            num_spec_tokens=num_spec_tokens,
        ))

    torch.testing.assert_close(scores_3d[:, :, 0], torch.full((batch_size, num_expand), -123.0))
    assert manager.seen_is_contiguous == [False, False, False]
    assert manager.seen_is_inference == [True, True, True]
    assert manager.seen_inference_mode == [True, True, True]
    assert manager.accepted == [10, 20, 11, 21]

class _NoForkGuidedHelper:

    def get_processors(self, session_ctx, response_formats):
        return {0: object()}


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
        dp_meta=DPMeta(dp_batches=[1, 32], dp_is_decoding=False),
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
        dp_meta=DPMeta(dp_batches=[1, 32], dp_is_decoding=False),
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


def test_qwen35_mtp_reuses_main_dist_context(monkeypatch):
    """Qwen3.5 MTP mirrors the target topology, so it should share groups."""
    from lmdeploy.pytorch.config import DistConfig, SpecDecodeConfig
    from lmdeploy.pytorch.distributed import DistContext
    from lmdeploy.pytorch.spec_decode import base as base_mod

    dist_config = DistConfig(dp=2, ep=2)
    dist_ctx = DistContext(rank=1, dp_rank=1, dist_config=dist_config, ep_gpu_group=object())
    specdecode_config = SpecDecodeConfig(model='draft-model',
                                         method='qwen3_5_mtp',
                                         dist_config=DistConfig(dp=2, ep=2),
                                         num_speculative_tokens=3)

    def fail_build(*args, **kwargs):
        raise AssertionError('qwen3_5_mtp should not build a separate draft DistContext')

    monkeypatch.setattr(base_mod.DistContext, 'build', staticmethod(fail_build))

    assert base_mod._build_draft_dist_ctx(dist_ctx, specdecode_config) is dist_ctx


def test_non_qwen35_mtp_builds_draft_dist_context(monkeypatch):
    """Other speculative methods keep their separate draft distribution
    path."""
    from lmdeploy.pytorch.config import DistConfig, SpecDecodeConfig
    from lmdeploy.pytorch.distributed import DistContext
    from lmdeploy.pytorch.spec_decode import base as base_mod

    main_dist_config = DistConfig(dp=2, ep=2)
    draft_dist_config = DistConfig()
    dist_ctx = DistContext(rank=1, dp_rank=1, dist_config=main_dist_config)
    specdecode_config = SpecDecodeConfig(model='draft-model',
                                         method='mtp',
                                         dist_config=draft_dist_config,
                                         num_speculative_tokens=3)
    draft_dist_ctx = DistContext(rank=1, dist_config=draft_dist_config)
    build_calls = []

    def fake_build(*, rank, dist_config):
        build_calls.append((rank, dist_config))
        return draft_dist_ctx

    monkeypatch.setattr(base_mod.DistContext, 'build', staticmethod(fake_build))

    assert base_mod._build_draft_dist_ctx(dist_ctx, specdecode_config) is draft_dist_ctx
    assert build_calls == [(dist_ctx.rank, draft_dist_config)]


def test_async_model_forward_dp1_non_last_chunk_skips_remaining_spec_forwards():
    """DP=1 non-last chunks should keep the local shortcut."""
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

    inputs, extra_inputs = _make_non_last_chunk_inputs()

    agent = object.__new__(SpecModelAgent)
    agent.num_spec_tokens = 3
    agent.rank = 0
    agent.proposer = _DummyProposer()
    agent.guided_helper = GuidedSpecHelper()
    forward_calls = 0

    def _forward_impl(_inputs):
        nonlocal forward_calls
        forward_calls += 1
        return {'call': forward_calls}

    agent._forward_impl = _forward_impl

    output = asyncio.run(agent._async_autoregressive_model_forward(inputs, extra_inputs, sampling_inputs=None))

    expected = torch.zeros((2, 3), dtype=torch.long)
    torch.testing.assert_close(output.output_draft_token_ids, expected)
    assert forward_calls == 1
    assert agent.proposer.get_outputs_calls == 0
    assert agent.proposer.update_inputs_decoding_calls == 0


def test_async_model_forward_dp_non_last_chunk_pads_block_offsets(monkeypatch):
    """DP non-last chunks should pad block offsets for draft decodes."""
    import lmdeploy.pytorch.spec_decode.spec_agent as spec_agent_mod
    from lmdeploy.pytorch.model_inputs import DPMeta
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

    monkeypatch.setattr(spec_agent_mod.DPMeta, 'build', staticmethod(lambda seqlen, num_tokens: DPMeta()))
    inputs, extra_inputs = _make_non_last_chunk_inputs(dp_meta=DPMeta(dp_batches=[2, 2]))

    agent = object.__new__(SpecModelAgent)
    agent.num_spec_tokens = 3
    agent.rank = 0
    agent.proposer = _DummyProposer()
    agent.guided_helper = GuidedSpecHelper()
    agent.cache_config = SimpleNamespace(kernel_block_size=1, num_reserved_gpu_blocks=1)
    forward_calls = 0
    forwarded_inputs = []

    def _forward_impl(_inputs):
        nonlocal forward_calls
        forward_calls += 1
        forwarded_inputs.append(_inputs)
        return {'call': forward_calls}

    agent._forward_impl = _forward_impl

    output = asyncio.run(agent._async_autoregressive_model_forward(inputs, extra_inputs, sampling_inputs=None))

    expected = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    torch.testing.assert_close(output.output_draft_token_ids, expected)
    assert forward_calls == agent.num_spec_tokens
    assert agent.proposer.get_outputs_calls == agent.num_spec_tokens
    assert agent.proposer.update_inputs_decoding_calls == 1
    assert agent.proposer.model.update_inputs_calls == agent.num_spec_tokens - 1
    assert forwarded_inputs[0] is inputs
    assert [inp.block_offsets.size(1) for inp in forwarded_inputs] == [1, 2, 2]
    torch.testing.assert_close(forwarded_inputs[1].block_offsets[:, 1], torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(forwarded_inputs[2].block_offsets[:, 1], torch.zeros(2, dtype=torch.long))
    assert all(not inp.is_dummy for inp in forwarded_inputs)
    assert all(inp.is_decoding for inp in forwarded_inputs[1:])


def test_async_model_forward_preserves_dp_global_decoding_in_draft_loop(monkeypatch):
    """Rebuilt draft-loop DPMeta must keep DP-global decode state."""
    import lmdeploy.pytorch.spec_decode.spec_agent as spec_agent_mod
    from lmdeploy.pytorch.model_inputs import DPMeta
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

    monkeypatch.setattr(spec_agent_mod.DPMeta, 'build', staticmethod(lambda seqlen, num_tokens: DPMeta()))
    inputs, extra_inputs = _make_non_last_chunk_inputs(dp_meta=DPMeta(dp_batches=[2, 2], dp_is_decoding=True))
    inputs.is_chunk = False

    agent = object.__new__(SpecModelAgent)
    agent.num_spec_tokens = 3
    agent.rank = 0
    agent.proposer = _DummyProposer()
    agent.guided_helper = GuidedSpecHelper()
    forward_calls = 0

    def _forward_impl(_inputs):
        nonlocal forward_calls
        forward_calls += 1
        return {'call': forward_calls}

    agent._forward_impl = _forward_impl

    asyncio.run(agent._async_autoregressive_model_forward(inputs, extra_inputs, sampling_inputs=None))

    assert agent.proposer.model.update_inputs_dp_is_decoding == [True, True]


def test_spec_model_agent_warmup_adds_dp_meta_for_draft_capture(monkeypatch):
    """Draft warmup must mark decode graph captures as DP-global decode."""
    import lmdeploy.pytorch.spec_decode.spec_agent as spec_agent_mod
    from lmdeploy.pytorch.config import DistConfig
    from lmdeploy.pytorch.distributed import DistContext, DistGroup
    from lmdeploy.pytorch.model_inputs import DPMeta, ModelInputs
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

    class DummyInputsStrategy:

        def make_dummy(self,
                       batch_size: int,
                       is_decoding: bool,
                       device: str = 'cpu',
                       vocab_size: int = 1,
                       max_q_seqlen: int = 1,
                       target_hidden_size: int = None,
                       target_dtype: torch.dtype = torch.float32,
                       meta=None):
            input_ids = torch.zeros((1, batch_size * max_q_seqlen), dtype=torch.long)
            seq_length = torch.full((batch_size, ), max_q_seqlen, dtype=torch.long)
            inputs = ModelInputs(input_ids=input_ids,
                                 seq_length=seq_length,
                                 history_lengths=torch.zeros(batch_size, dtype=torch.long),
                                 block_offsets=torch.zeros((batch_size, 1), dtype=torch.long),
                                 is_decoding=is_decoding,
                                 num_ignored_history=torch.zeros(batch_size, dtype=torch.long),
                                 max_q_seqlen=max_q_seqlen,
                                 max_kv_seqlen=max_q_seqlen,
                                 sum_kv_seqlen=batch_size * max_q_seqlen)
            if target_hidden_size is not None:
                inputs.target_hidden_states = torch.zeros((1, batch_size * max_q_seqlen, target_hidden_size),
                                                          dtype=target_dtype)
            return inputs

    class DummyDraftModel:

        def get_capture_batch_sizes(self):
            return [2]

    class DummyProposer:

        def __init__(self):
            self.model = DummyDraftModel()

        def get_target_hidden_size(self, target_model_config):
            return 4

    build_calls = []

    def fake_dp_meta_build(seqlen, num_tokens):
        build_calls.append((seqlen, list(num_tokens)))
        return DPMeta(tp_sizes=[seqlen], moe_tp_sizes=[seqlen])

    monkeypatch.setattr(spec_agent_mod.DPMeta, 'build', staticmethod(fake_dp_meta_build))

    dist_config = DistConfig(dp=2, ep=2)
    cpu_group = object()
    draft_dist_ctx = DistContext(rank=0,
                                 dp_rank=0,
                                 dist_config=dist_config,
                                 cpu_group=cpu_group,
                                 attn_tp_group=DistGroup(rank=0),
                                 mlp_tp_group=DistGroup(rank=0),
                                 moe_tp_group=DistGroup(rank=0),
                                 tp_group=DistGroup(rank=0))
    barrier_calls = []
    sync_calls = []
    monkeypatch.setattr(spec_agent_mod.dist, 'barrier', lambda group=None: barrier_calls.append(group))
    monkeypatch.setattr(spec_agent_mod.torch.cuda, 'synchronize', lambda: sync_calls.append(True))

    agent = object.__new__(SpecModelAgent)
    agent.draft_dist_ctx = draft_dist_ctx
    agent.inputs_strategy = DummyInputsStrategy()
    agent.proposer = DummyProposer()
    agent.model_config = SimpleNamespace(vocab_size=11, dtype=torch.float32, hidden_size=8)
    agent.num_spec_tokens = 3
    agent.make_dummy_meta = None

    forwarded = []

    def forward_impl(inputs):
        forwarded.append({
            'num_tokens': inputs.input_ids.numel(),
            'batch_size': inputs.seq_length.numel(),
            'is_decoding': inputs.is_decoding,
            'dp_batches': inputs.dp_meta.dp_batches,
            'dp_is_decoding': inputs.dp_meta.dp_is_decoding,
            'global_is_decoding': inputs.global_is_decoding(),
        })

    agent._forward_impl = forward_impl

    agent.warmup(max_batches=4, target_model_config=SimpleNamespace())

    assert barrier_calls == [cpu_group]
    assert len(sync_calls) == 3
    assert build_calls == [(4, [4, 4]), (8, [8, 8]), (2, [2, 2])]
    assert forwarded == [
        {
            'num_tokens': 4,
            'batch_size': 4,
            'is_decoding': False,
            'dp_batches': [4, 4],
            'dp_is_decoding': False,
            'global_is_decoding': False,
        },
        {
            'num_tokens': 8,
            'batch_size': 2,
            'is_decoding': True,
            'dp_batches': [2, 2],
            'dp_is_decoding': True,
            'global_is_decoding': True,
        },
        {
            'num_tokens': 2,
            'batch_size': 2,
            'is_decoding': True,
            'dp_batches': [2, 2],
            'dp_is_decoding': True,
            'global_is_decoding': True,
        },
    ]


def test_dflash_diffusion_warmup_materializes_context_and_captures_only_block_queries(monkeypatch):
    """DFlash warmup must not capture its dead prefill or q_len=1 graphs."""

    class DummyInputsStrategy:

        def __init__(self):
            self.calls = []

        def make_dummy(self,
                       batch_size,
                       is_decoding,
                       device='cpu',
                       vocab_size=1,
                       max_q_seqlen=1,
                       target_hidden_size=None,
                       target_dtype=torch.float32,
                       meta=None):
            self.calls.append((batch_size, is_decoding, max_q_seqlen, target_hidden_size))
            num_tokens = batch_size * max_q_seqlen
            return ModelInputs(
                input_ids=torch.zeros((1, num_tokens), dtype=torch.long),
                seq_length=torch.full((batch_size, ), max_q_seqlen, dtype=torch.long),
                history_lengths=torch.zeros(batch_size, dtype=torch.long),
                block_offsets=torch.zeros((batch_size, 1), dtype=torch.long),
                is_decoding=is_decoding,
                num_ignored_history=torch.zeros(batch_size, dtype=torch.long),
                max_q_seqlen=max_q_seqlen,
                max_kv_seqlen=max_q_seqlen,
                sum_kv_seqlen=num_tokens,
                target_hidden_states=torch.zeros((1, num_tokens, target_hidden_size), dtype=target_dtype),
                target_position_ids=torch.zeros((1, num_tokens), dtype=torch.long),
            )

    class DummyGraphModel:

        def get_capture_batch_sizes(self):
            return [2, 4]

    proposer = _make_dflash_proposer()
    proposer.model = DummyGraphModel()
    materialized = []
    monkeypatch.setattr(proposer, '_materialize_context',
                        lambda inputs, hidden, cache: materialized.append(
                            (inputs.is_decoding, inputs.seq_length.numel(), inputs.max_q_seqlen,
                             tuple(hidden.shape), cache)))

    inputs_strategy = DummyInputsStrategy()
    forward_graph_keys = []
    sync_calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: sync_calls.append(True))

    agent = object.__new__(SpecModelAgent)
    agent.draft_dist_ctx = SimpleNamespace(dist_config=SimpleNamespace(dp=1))
    agent.draft_context = lambda: nullcontext()
    agent.inputs_strategy = inputs_strategy
    agent.proposer = proposer
    agent.cache_engine = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=8, max_prefill_token_num=10))
    agent.model_config = SimpleNamespace(vocab_size=11, dtype=torch.float32, hidden_size=4)
    agent.num_spec_tokens = 3
    agent.make_dummy_meta = None
    dp_markers = []

    def build_dp_meta(inputs):
        marker = object()
        inputs.dp_meta = marker
        dp_markers.append(marker)

    agent._build_warmup_dp_meta = build_dp_meta
    agent._forward_impl = lambda inputs: forward_graph_keys.append(
        (inputs.seq_length.numel(), inputs.max_q_seqlen, inputs.dp_meta)) or {}

    agent.warmup(max_batches=4, target_model_config=SimpleNamespace(hidden_size=4))

    assert inputs_strategy.calls == [(1, False, 8, 8), (4, True, 4, 8), (2, True, 4, 8)]
    assert [(is_decoding, batches, q_len, shape) for is_decoding, batches, q_len, shape, _ in materialized] == [
        (False, 1, 8, (8, 8)),
        (False, 4, 4, (16, 8)),
        (False, 2, 4, (8, 8)),
    ]
    assert all(cache is agent.cache_engine for *_, cache in materialized)
    assert forward_graph_keys == [(4, 4, dp_markers[1]), (2, 4, dp_markers[2])]
    assert len(sync_calls) == 3
    assert 'synchronize' not in inspect.getsource(DFlash.prepare_warmup_forward)


def test_proposal_warmup_plan_is_immutable_and_shape_only():
    case = ProposalWarmupCase(batch_size=4, is_decoding=True, max_q_seqlen=16, target_hidden_size=1024)
    plan = ProposalWarmupPlan(cases=(case, ))

    assert tuple(ProposalWarmupCase.__dataclass_fields__) == (
        'batch_size', 'is_decoding', 'max_q_seqlen', 'target_hidden_size')
    assert tuple(ProposalWarmupPlan.__dataclass_fields__) == ('cases', )
    with pytest.raises(FrozenInstanceError):
        case.batch_size = 2
    with pytest.raises(FrozenInstanceError):
        plan.cases = ()


def _make_dflash_proposer():
    return DFlash(
        SimpleNamespace(
            mask_token_id=99,
            target_layer_ids=(1, 5),
            num_speculative_tokens=3,
            model_config=None,
        ),
        device='cpu',
    )


def test_dflash_proposer_hook_rejects_guided_before_fork():
    agent = object.__new__(SpecModelAgent)
    agent.guided_helper = _NoForkGuidedHelper()
    agent.proposer = _make_dflash_proposer()
    agent.proposer.guided_helper = agent.guided_helper
    agent.cache_engine = None
    agent.rank = 0
    agent._dflash_debug_step = 0
    agent.draft_context = lambda: nullcontext()

    model_inputs = ModelInputs(
        input_ids=torch.tensor([[1]]),
        seq_length=torch.ones(1, dtype=torch.long),
        history_lengths=torch.zeros(1, dtype=torch.long),
        block_offsets=torch.ones((1, 1), dtype=torch.int32),
        is_decoding=False,
        num_ignored_history=torch.zeros(1, dtype=torch.long),
        max_q_seqlen=1,
        max_kv_seqlen=1,
        sum_kv_seqlen=1,
    )
    extra_inputs = ARSpecExtraInputs(next_token_ids=torch.tensor([2]))
    sampling_inputs = SimpleNamespace(session_ctx=object(), response_formats=object())

    with pytest.raises(NotImplementedError, match='DFlash guided decoding'):
        asyncio.run(agent.async_model_forward(model_inputs, extra_inputs, sampling_inputs))


def test_dflash_proposer_hook_materializes_non_last_chunk_without_drafting(monkeypatch):
    import lmdeploy.pytorch.spec_decode.proposers.dflash as dflash_mod

    inputs, extra_inputs = _make_non_last_chunk_inputs()

    agent = object.__new__(SpecModelAgent)
    agent.num_spec_tokens = 3
    agent.rank = 0
    agent._dflash_debug_step = 0
    agent.guided_helper = GuidedSpecHelper()
    agent.proposer = _make_dflash_proposer()
    agent.proposer.guided_helper = agent.guided_helper
    agent.cache_engine = object()
    agent.draft_context = lambda: nullcontext()
    captured = {
        'materialize_context_calls': 0,
        'propose_block_calls': 0,
    }

    def materialize_context(model_inputs, extra_inputs, cache_engine):
        captured['materialize_context_calls'] += 1
        assert cache_engine is agent.cache_engine

    async def propose_block(model_inputs, extra_inputs, cache_engine, guided_processors=None):
        captured['propose_block_calls'] += 1
        return model_inputs.input_ids.new_full((model_inputs.seq_length.size(0), 3), 9)

    monkeypatch.setattr(agent.proposer, 'materialize_context', materialize_context)
    monkeypatch.setattr(agent.proposer, 'propose_block', propose_block)
    monkeypatch.delenv('LMDEPLOY_DFLASH_DEBUG_DIR', raising=False)

    def fail_debug_tensor(*args, **kwargs):
        raise AssertionError('Disabled DFlash tracing must not serialize tensors.')

    monkeypatch.setattr(dflash_mod, 'debug_tensor', fail_debug_tensor)

    output = asyncio.run(agent.async_model_forward(inputs, extra_inputs, sampling_inputs=None))

    torch.testing.assert_close(output.output_draft_token_ids, torch.zeros((2, 3), dtype=torch.long))
    assert captured['materialize_context_calls'] == 1
    assert captured['propose_block_calls'] == 0
    assert output.next_token_ids is extra_inputs.next_token_ids


def test_dflash_proposer_api_uses_explicit_context_not_spec_agent():
    proposer = _make_dflash_proposer()

    assert proposer.proposal_method == ProposalMethod.DIFFUSION
    assert proposer.requires_target_inputs_embeds is False
    assert not hasattr(proposer, 'input_mode')
    assert not hasattr(proposer, 'proposal_mode')
    assert tuple(ProposalContext.__dataclass_fields__) == ('cache_engine', 'rank', 'debug_step')


def test_dflash_proposer_requires_explicit_proposal_context():
    proposer = _make_dflash_proposer()

    with pytest.raises(RuntimeError, match='requires ProposalContext'):
        asyncio.run(proposer.propose(None, None, None))


def test_dflash_spec_agent_reset_runtime_state_discards_chunk_carry_and_debug_step():
    agent = SpecModelAgent.__new__(SpecModelAgent)
    agent._prev_chunk_last = {'hidden_states': object()}
    agent._dflash_debug_step = 7

    agent.reset_runtime_state()

    assert agent._prev_chunk_last == {}
    assert agent._dflash_debug_step == 0


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


def _model_inputs(input_ids,
                  *,
                  is_decoding=False,
                  is_chunk=False,
                  is_first_chunk=False,
                  is_last_chunk=False,
                  dp_meta=None):
    input_ids = torch.tensor([input_ids])
    seq_length = torch.tensor([input_ids.size(1)])
    history_lengths = torch.tensor([0])
    max_q_seqlen = input_ids.size(1)
    return ModelInputs(
        input_ids=input_ids,
        seq_length=seq_length,
        history_lengths=history_lengths,
        block_offsets=torch.zeros(1, 1, dtype=torch.int32),
        is_decoding=is_decoding,
        num_ignored_history=torch.zeros(1, dtype=torch.long),
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=max_q_seqlen,
        sum_kv_seqlen=max_q_seqlen,
        is_chunk=is_chunk,
        is_first_chunk=is_first_chunk,
        is_last_chunk=is_last_chunk,
        dp_meta=dp_meta,
    )


def _extra(hidden_values):
    hidden_states = torch.tensor([hidden_values], dtype=torch.float32)
    return ARSpecExtraInputs(
        target_hidden_states=hidden_states,
        next_token_ids=torch.tensor([99]),
        last_token_indices=torch.tensor([hidden_states.size(1) - 1]),
    )


def test_prepare_inputs_from_main_keeps_chunk_carry_across_decode():
    agent = SpecModelAgent.__new__(SpecModelAgent)
    agent._prev_chunk_last = {}

    first_chunk = _model_inputs([10, 11, 12], is_chunk=True, is_first_chunk=True)
    agent._prepare_inputs_from_main(first_chunk, _extra([[1, 10], [2, 20], [3, 30]]))
    saved_first_chunk_last = agent._prev_chunk_last['hidden_states'].clone()

    decode = _model_inputs([90, 91, 92], is_decoding=True)
    agent._prepare_inputs_from_main(decode, _extra([[9, 90], [8, 80], [7, 70]]))

    assert torch.equal(agent._prev_chunk_last['hidden_states'], saved_first_chunk_last)

    middle_chunk = _model_inputs([20, 21, 22], is_chunk=True)
    draft_inputs, _ = agent._prepare_inputs_from_main(middle_chunk, _extra([[4, 40], [5, 50], [6, 60]]))

    assert torch.equal(draft_inputs.target_hidden_states[:, :1], saved_first_chunk_last)
    assert torch.equal(agent._prev_chunk_last['hidden_states'], torch.tensor([[[6., 60.]]]))


def test_prepare_inputs_from_main_keeps_chunk_carry_across_interleaved_prefill():
    agent = SpecModelAgent.__new__(SpecModelAgent)
    saved = torch.ones(1, 1, 2)
    agent._prev_chunk_last = {'hidden_states': saved.clone()}

    prefill = _model_inputs([10, 11, 12])
    agent._prepare_inputs_from_main(prefill, _extra([[1, 10], [2, 20], [3, 30]]))

    torch.testing.assert_close(agent._prev_chunk_last['hidden_states'], saved)


def test_prepare_inputs_from_main_first_chunk_clears_stale_chunk_carry():
    agent = SpecModelAgent.__new__(SpecModelAgent)
    agent._prev_chunk_last = {'hidden_states': torch.ones(1, 1, 2)}

    first_chunk = _model_inputs([10, 11, 12], is_chunk=True, is_first_chunk=True)
    agent._prepare_inputs_from_main(first_chunk, _extra([[1, 10], [2, 20], [3, 30]]))

    torch.testing.assert_close(agent._prev_chunk_last['hidden_states'], torch.tensor([[[3., 30.]]]))


def test_prepare_inputs_from_main_keeps_chunk_carry_for_dp_local_decode_global_prefill():
    agent = SpecModelAgent.__new__(SpecModelAgent)
    saved = torch.ones(1, 1, 2)
    agent._prev_chunk_last = {'hidden_states': saved.clone()}
    agent.proposer = _DummyProposer()

    dp_meta = DPMeta(dp_batches=[1, 1], dp_is_decoding=False)
    inputs = _model_inputs([90, 91, 92], is_decoding=True, dp_meta=dp_meta)
    agent._prepare_inputs_from_main(inputs, _extra([[9, 90], [8, 80], [7, 70]]))

    assert torch.equal(agent._prev_chunk_last['hidden_states'], saved)
