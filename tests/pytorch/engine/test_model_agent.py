# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from collections import deque
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.pytorch.model_inputs import ModelInputs


def _input_logprob_model_inputs(input_ids, indices):
    size = len(input_ids)
    return ModelInputs(input_ids=torch.tensor([input_ids]),
                       seq_length=torch.tensor([size]),
                       history_lengths=torch.tensor([0]),
                       block_offsets=torch.zeros((1, 1), dtype=torch.long),
                       is_decoding=False,
                       num_ignored_history=torch.tensor([0]),
                       max_q_seqlen=size,
                       max_kv_seqlen=size,
                       sum_kv_seqlen=size,
                       logits_indices=None if indices is None else torch.tensor(indices, dtype=torch.long),
                       seq_logit_length=None if indices is None else torch.tensor([len(indices)]))


def test_get_input_logits_projects_only_selected_rows():
    vocab_size = 8
    hidden = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
    inputs = _input_logprob_model_inputs([0, 1, 2, 3, 4, 5], [0, 1, 3, 4])
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.model_config = SimpleNamespace(vocab_size=vocab_size)
    calls = []

    def get_logits(values):
        calls.append(values.shape[1])
        torch.testing.assert_close(values[0], hidden[0, [0, 1, 3, 4]])
        return torch.arange(values.shape[1] * vocab_size, dtype=torch.float32).reshape(
            1, values.shape[1], vocab_size)

    agent.get_logits = get_logits
    output = agent._get_input_logits(hidden, inputs)

    assert calls == [4]
    assert output.shape == (4, vocab_size)


def test_prefill_input_logprobs_helper_is_semantic():
    agent = BaseModelAgent.__new__(BaseModelAgent)
    inputs = _input_logprob_model_inputs([0, 1, 2], [0, 1])
    assert agent._is_prefill_input_logprobs(inputs)

    missing_lengths = _input_logprob_model_inputs([0, 1, 2], [0, 1])
    missing_lengths.seq_logit_length = None
    assert not agent._is_prefill_input_logprobs(missing_lengths)

    decode = _input_logprob_model_inputs([0, 1, 2], [0, 1])
    decode.is_decoding = True
    assert not agent._is_prefill_input_logprobs(decode)

    dummy = _input_logprob_model_inputs([0, 1, 2], [0, 1])
    dummy.is_dummy = True
    assert not agent._is_prefill_input_logprobs(dummy)

    ordinary = _input_logprob_model_inputs([0, 1, 2], None)
    assert not agent._is_prefill_input_logprobs(ordinary)


@pytest.mark.parametrize('return_logits', [False, True])
def test_async_forward_returns_only_input_logits_without_sampling_policy(
        return_logits):
    vocab_size = 8
    hidden = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    inputs = _input_logprob_model_inputs([0, 1, 2], [1, 2])
    calls = []
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.model_config = SimpleNamespace(vocab_size=vocab_size)

    async def async_forward(_inputs, **kwargs):
        return {'hidden_states': [hidden]}

    def get_logits(values):
        calls.append(values.shape[1])
        return torch.zeros((1, values.shape[1], vocab_size))

    agent.async_forward = async_forward
    agent.get_logits = get_logits
    agent._postprocess_forward_output = lambda output, _inputs: output
    agent.spec_agent = SimpleNamespace(
        update_main_model_outputs=lambda *_: pytest.fail('scoring path must not enter speculative processing'))

    output = asyncio.run(agent._async_model_forward(inputs, return_logits=return_logits))

    assert calls == [2]
    assert 'input_logits' not in output
    assert output['logits'].shape == (2, vocab_size)


def test_async_forward_requires_complete_prefill_input_logprob_metadata():
    vocab_size = 8
    hidden = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
    inputs = _input_logprob_model_inputs([0, 1, 2, 3, 4], [0, 1])
    inputs.seq_logit_length = None
    calls = []
    agent = BaseModelAgent.__new__(BaseModelAgent)

    async def async_forward(_inputs, **kwargs):
        return {'hidden_states': [hidden]}

    def get_logits(values):
        calls.append(values.shape[1])
        return torch.zeros((1, values.shape[1], vocab_size))

    agent.async_forward = async_forward
    agent.get_logits = get_logits
    agent._postprocess_forward_output = lambda output, _inputs: output
    agent.spec_agent = SimpleNamespace(
        is_enabled=lambda: False,
        update_main_model_outputs=lambda output, _inputs:
        (output.pop('hidden_states')[0], output))

    output = asyncio.run(agent._async_model_forward(inputs, return_logits=False))

    assert calls == [5]
    assert 'input_logits' not in output
    assert output['logits'].shape == (1, 5, vocab_size)


def test_async_forward_disabled_path_keeps_one_sampling_projection():
    vocab_size = 8
    hidden = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
    inputs = _input_logprob_model_inputs([0, 1, 2, 3, 4], None)
    calls = []
    agent = BaseModelAgent.__new__(BaseModelAgent)

    async def async_forward(_inputs, **kwargs):
        return {'hidden_states': [hidden]}

    def get_logits(values):
        calls.append(values.shape[1])
        return torch.zeros((1, values.shape[1], vocab_size))

    agent.async_forward = async_forward
    agent.get_logits = get_logits
    agent._postprocess_forward_output = lambda output, _inputs: output
    agent.spec_agent = SimpleNamespace(
        is_enabled=lambda: False,
        update_main_model_outputs=lambda output, _inputs: (output.pop('hidden_states')[0], output))

    output = asyncio.run(agent._async_model_forward(inputs, return_logits=False))

    assert calls == [5]
    assert 'input_logits' not in output


@pytest.mark.parametrize('mode', ['raw_logits', 'raw_logprobs'])
@pytest.mark.parametrize('num_logprobs', [0, 1, 2])
def test_input_compaction_matches_shared_reference(mode, num_logprobs):
    inputs = _input_logprob_model_inputs([0, 1, 2], [0, 1])
    input_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [1.0, 0.5, -0.5, -1.0]])
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.misc_config = SimpleNamespace(logprobs_mode=mode)

    outputs = agent._get_outputs_with_logprobs(input_logits, inputs, num_logprobs, model_metas=[{'meta': 1}])
    compact = outputs.logprobs

    reference = input_logits.log_softmax(-1) if mode == 'raw_logprobs' else input_logits
    targets = torch.tensor([1, 2])
    torch.testing.assert_close(compact.vals[:, 0], reference.gather(-1, targets[:, None])[:, 0])
    assert compact.indices[:, 0].tolist() == targets.tolist()
    assert compact.vals.shape == (2, num_logprobs + 1)
    assert outputs.next_token_ids.tolist() == [0]
    assert outputs.stopped.tolist() == [True]
    assert outputs.stop_pos.tolist() == [-1]
    assert outputs.model_metas == [{'meta': 1}]


def test_input_compaction_reuses_previous_chunk_last_logit_for_cross_chunk_target():
    first_inputs = _input_logprob_model_inputs([0, 1, 2], [0, 1, 2])
    first_inputs.is_chunk = True
    first_inputs.is_first_chunk = True
    first_inputs.is_last_chunk = False
    first_logits = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.misc_config = SimpleNamespace(logprobs_mode='raw_logits')
    agent._prev_chunk_last_logit = None

    first_outputs = agent._get_outputs_with_logprobs(first_logits, first_inputs, 0, model_metas=None)
    assert first_outputs.logprobs.indices[:, 0].tolist() == [1, 2]
    torch.testing.assert_close(agent._prev_chunk_last_logit, first_logits[-1:])

    final_inputs = _input_logprob_model_inputs([3, 4, 5], [0, 1])
    final_inputs.is_chunk = True
    final_inputs.is_first_chunk = False
    final_inputs.is_last_chunk = True
    final_logits = torch.arange(100, 116, dtype=torch.float32).reshape(2, 8)

    outputs = agent._get_outputs_with_logprobs(final_logits, final_inputs, 0, model_metas=None)
    compact = outputs.logprobs

    expected_logits = torch.cat([first_logits[-1:], final_logits], dim=0)
    assert compact.indices[:, 0].tolist() == [3, 4, 5]
    torch.testing.assert_close(compact.vals[:, 0], expected_logits.gather(-1, compact.indices.long())[:, 0])
    assert agent._prev_chunk_last_logit is None


def test_input_compaction_accepts_empty_pre_boundary_chunk():
    inputs = _input_logprob_model_inputs([0, 1, 2], [])
    inputs.is_chunk = True
    inputs.is_first_chunk = True
    inputs.is_last_chunk = False
    input_logits = torch.empty((0, 8))
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.misc_config = SimpleNamespace(logprobs_mode='raw_logits')
    agent._prev_chunk_last_logit = None

    outputs = agent._get_outputs_with_logprobs(input_logits, inputs, 2, model_metas=None)

    assert outputs.logprobs.vals.shape == (0, 3)
    assert outputs.logprobs.indices.shape == (0, 3)
    assert agent._prev_chunk_last_logit is None


def test_dp_dummy_skips_input_compaction_without_sampling_inputs():
    inputs = _input_logprob_model_inputs([0], None)
    inputs.is_dummy = True
    generated = SimpleNamespace(next_token_ids=torch.tensor([0]),
                                output_token_ids=torch.tensor([[0]]),
                                logprobs=None)

    async def spec_sampling(_inputs, _extra, _sampling):
        return generated

    async def spec_forward(_inputs, extra, _sampling):
        return extra

    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.rank = 0
    agent.spec_agent = SimpleNamespace(is_enabled=lambda: True,
                                       async_sampling_logits=spec_sampling,
                                       async_model_forward=spec_forward)

    result = asyncio.run(
        agent._step_postprocess_with_output(last_logits=torch.zeros((1, 8)),
                                            logits=torch.zeros((1, 8)),
                                            inputs=inputs,
                                            sampling_inputs=None,
                                            stopping_criteria=None,
                                            model_metas=None,
                                            need_broadcast_next=False))

    assert result[-2] is generated.next_token_ids


def test_mtp_sampling_still_delegates_generated_rows():
    generated = SimpleNamespace(next_token_ids=torch.tensor([5]),
                                output_token_ids=torch.tensor([[5, 6]]),
                                logprobs='spec-generated')

    async def spec_sampling(_inputs, _extra, _sampling):
        return generated

    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.spec_agent = SimpleNamespace(is_enabled=lambda: True,
                                       async_sampling_logits=spec_sampling)
    result = asyncio.run(
        agent.async_sampling_logits(torch.zeros((1, 8)),
                                    _input_logprob_model_inputs([0, 1, 2], [0, 1]),
                                    SimpleNamespace(), SamplingInputs(max_num_logprobs=0)))

    assert result[0] is generated.next_token_ids
    assert result[1] == 'spec-generated'
    assert result[2] is generated.output_token_ids


def test_ordinary_non_final_chunk_keeps_upstream_generated_logprob_row(monkeypatch):
    import lmdeploy.pytorch.engine.model_agent.agent as agent_module

    class _Processor:

        def __init__(self, *args, **kwargs):
            pass

        async def __call__(self, logits):
            return logits, logits

        def sampling(self, logits):
            return torch.tensor([3])

        async def accept_guided_tokens(self, token_ids):
            pass

        def compute_logprobs(self, raw_logprobs, token_ids):
            indices = token_ids[:, None]
            return raw_logprobs.gather(-1, indices), indices.to(torch.int32)

    monkeypatch.setattr(agent_module, 'FusedLogitsProcessor', _Processor)
    inputs = _input_logprob_model_inputs([0, 1, 2], [0, 1])
    inputs.is_chunk = True
    inputs.is_last_chunk = False
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.misc_config = SimpleNamespace(logprobs_mode='raw_logits')
    agent.guided_decoding_manager = None
    agent.spec_agent = SimpleNamespace(is_enabled=lambda: False)
    agent.agent_strategy = SimpleNamespace(post_sampling=lambda _i, _l, ids, extra: (ids, extra))

    result = asyncio.run(
        agent.async_sampling_logits(torch.zeros((1, 8)), inputs, SimpleNamespace(),
                                    SamplingInputs(max_num_logprobs=0)))

    assert result[1] is not None


@pytest.fixture
def event_loop():
    try:
        old_loop = asyncio.get_event_loop()
    except RuntimeError:
        old_loop = None
    new_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(new_loop)
        yield new_loop
    finally:
        pending = asyncio.all_tasks(new_loop)
        for task in pending:
            task.cancel()
        if pending:
            new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        new_loop.run_until_complete(new_loop.shutdown_asyncgens())
        new_loop.stop()
        new_loop.close()
        asyncio.set_event_loop(old_loop)


def _make_agent_with_queues():
    """Create a minimal BaseModelAgent-like object with internal queues."""
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

    # Bypass __init__ — we only need the queues.
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent._pre_in_que = asyncio.Queue()
    agent._in_que = asyncio.Queue()
    agent._out_que = asyncio.Queue()
    return agent


def test_prepare_inputs_prefill_keeps_chunk_model_metas_across_interleaved_prefill():
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

    agent = BaseModelAgent.__new__(BaseModelAgent)
    prev_output = {'model_metas': [{'chunk': 1}]}
    agent._prev_chunk_output = prev_output

    normal_prefill = SimpleNamespace(is_chunk=False,
                                     is_first_chunk=False,
                                     is_last_chunk=False,
                                     model_metas=[{
                                         'normal': 1
                                     }])

    agent._prepare_inputs_prefill(normal_prefill, delta=None)

    assert agent._prev_chunk_output is prev_output
    assert normal_prefill.model_metas == [{'normal': 1}]

    middle_chunk = SimpleNamespace(is_chunk=True, is_first_chunk=False, is_last_chunk=False, model_metas=None)

    agent._prepare_inputs_prefill(middle_chunk, delta=None)

    assert middle_chunk.model_metas == [{'chunk': 1}]
    assert agent._prev_chunk_output is prev_output


def test_prepare_inputs_prefill_final_chunk_consumes_chunk_model_metas():
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent._prev_chunk_output = {'model_metas': [{'chunk': 1}]}
    final_chunk = SimpleNamespace(is_chunk=True, is_first_chunk=False, is_last_chunk=True, model_metas=None)

    agent._prepare_inputs_prefill(final_chunk, delta=None)

    assert final_chunk.model_metas == [{'chunk': 1}]
    assert agent._prev_chunk_output is None


def test_model_agent_reset_runtime_state_discards_decode_and_chunk_carry():
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

    events = []
    old_step_inputs = object()
    new_step_inputs = object()

    class _StrategyFactory:

        def build_step_inputs(self):
            events.append('build_step_inputs')
            return new_step_inputs

    class _SpecAgent:

        def reset_runtime_state(self):
            events.append('reset_spec')

    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.strategy_factory = _StrategyFactory()
    agent.spec_agent = _SpecAgent()
    agent.step_inputs = old_step_inputs
    agent._prev_chunk_output = {'model_metas': [object()]}
    agent._prev_chunk_last_logit = object()

    agent.reset_runtime_state()

    assert agent.step_inputs is new_step_inputs
    assert agent._prev_chunk_output is None
    assert agent._prev_chunk_last_logit is None
    assert events == ['build_step_inputs', 'reset_spec']


@pytest.mark.parametrize('graph_wrapped', [False, True])
def test_model_agent_builds_and_retains_worker_local_cache_plans(graph_wrapped):
    from lmdeploy.pytorch.config import CacheConfig, ModelConfig
    from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheBinding, BlockCacheRequest
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

    request = BlockCacheRequest('operator_cache', (64, 3), torch.float16)

    class _CacheRequester(torch.nn.Module):

        def get_block_cache_requests(self, context):
            assert context.geometry.logical_block_size == 128
            assert context.geometry.kernel_block_size == 64
            return (request, )

        def bind_block_cache(self, binding: BlockCacheBinding):
            assert binding.cache_name == 'operator_cache'
            self.cache_binding = binding

    class _PatchedModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.requesters = torch.nn.ModuleList([_CacheRequester(), _CacheRequester()])

    class _GraphRunner:

        def __init__(self, model):
            self.model = model

        def get_model(self):
            return self.model

    model_config = ModelConfig(hidden_size=16,
                               num_layers=4,
                               num_attention_heads=2,
                               num_key_value_heads=2,
                               bos_token_id=1,
                               eos_token_id=[2],
                               head_dim=8,
                               use_standard_kv_cache=False)
    cache_config = CacheConfig(max_batches=1,
                               block_size=128,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.all_context = nullcontext
    agent.dist_config = SimpleNamespace(attn_tp=1)
    patched_model = _PatchedModel()
    agent.patched_model = _GraphRunner(patched_model) if graph_wrapped else patched_model
    agent.model_config = model_config
    agent.spec_agent = SimpleNamespace(build_cache_plan=lambda config: 128)
    agent.memdecode_agent = SimpleNamespace(build_cache_plan=lambda config: 64)

    sizes = agent.build_cache_plans(cache_config, spec_cache_config=object())

    assert sizes == (2048, 128, 64)
    assert agent._cache_plan_block_nbytes == (2048, 128)
    assert tuple(spec.name for spec in agent.block_cache_plan.tensor_specs) == ('operator_cache', )
    assert agent.block_cache_plan.tensor_specs[0].consumer_rows == (0, 1)
    assert [requester.cache_binding.consumer_row for requester in patched_model.requesters] == [0, 1]


def test_spec_model_agent_collects_cache_requests_from_graph_runner():
    from lmdeploy.pytorch.config import CacheConfig, ModelConfig
    from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheBinding, BlockCacheRequest
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

    request = BlockCacheRequest('draft_cache', (64, 3), torch.float16)

    class _CacheRequester(torch.nn.Module):

        def get_block_cache_requests(self, context):
            return (request, )

        def bind_block_cache(self, binding: BlockCacheBinding):
            self.cache_binding = binding

    class _DraftModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.requester = _CacheRequester()

    class _GraphRunner:

        def __init__(self, model):
            self.model = model

        def get_model(self):
            return self.model

    model_config = ModelConfig(hidden_size=16,
                               num_layers=1,
                               num_attention_heads=2,
                               num_key_value_heads=2,
                               bos_token_id=1,
                               eos_token_id=[2],
                               head_dim=8,
                               use_standard_kv_cache=False)
    cache_config = CacheConfig(max_batches=1,
                               block_size=128,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    draft_model = _DraftModel()
    agent = SpecModelAgent.__new__(SpecModelAgent)
    agent.draft_context = nullcontext
    agent.draft_dist_ctx = SimpleNamespace(dist_config=SimpleNamespace(attn_tp=1))
    agent.proposer = SimpleNamespace(model=_GraphRunner(draft_model))
    agent.model_config = model_config

    block_nbytes = agent.build_cache_plan(cache_config)

    assert block_nbytes == 1024
    assert tuple(spec.name for spec in agent.block_cache_plan.tensor_specs) == ('draft_cache', )
    assert draft_model.requester.cache_binding.consumer_row == 0


def test_build_spec_agent_allows_guided_spec_followers_without_proposer():
    from lmdeploy.pytorch.config import BackendConfig, DistConfig, SpecDecodeConfig
    from lmdeploy.pytorch.distributed import DistContext
    from lmdeploy.pytorch.spec_decode import build_spec_agent

    guided_manager = object()
    specdecode_config = SpecDecodeConfig(
        model='draft-model',
        method='deepseek_mtp',
        dist_config=DistConfig(),
        num_speculative_tokens=3,
    )
    spec_agent = build_spec_agent(
        specdecode_config,
        backend_config=BackendConfig(),
        dist_ctx=DistContext(rank=1, dist_config=DistConfig(tp=2)),
        inputs_strategy=None,
        agent_strategy=None,
        misc_config=None,
        device='cpu',
        guided_decoding_manager=guided_manager,
    )
    assert spec_agent.is_enabled()
    assert spec_agent.proposer is None
    assert not hasattr(spec_agent, 'rejection_sampler')
    assert not hasattr(spec_agent, 'guided_helper')


def test_build_spec_agent_shares_guided_helper_with_proposer(monkeypatch):
    import lmdeploy.pytorch.spec_decode.spec_agent as spec_agent_mod
    from lmdeploy.pytorch.config import BackendConfig, DistConfig, SpecDecodeConfig
    from lmdeploy.pytorch.distributed import DistContext
    from lmdeploy.pytorch.spec_decode import build_spec_agent

    guided_manager = object()
    proposer = SimpleNamespace(guided_helper=None)
    rejection_sampler = object()
    monkeypatch.setattr(spec_agent_mod, 'build_specdecode_proposer', lambda *args, **kwargs: proposer)
    monkeypatch.setattr(spec_agent_mod, 'RejectionSampler', lambda *args: rejection_sampler)
    inputs_strategy = SimpleNamespace(create_make_dummy_meta=lambda model_config: None)
    specdecode_config = SpecDecodeConfig(
        model='draft-model',
        method='deepseek_mtp',
        dist_config=DistConfig(),
        num_speculative_tokens=3,
    )

    spec_agent = build_spec_agent(
        specdecode_config,
        backend_config=BackendConfig(),
        dist_ctx=DistContext(rank=0, dist_config=DistConfig(tp=2)),
        inputs_strategy=inputs_strategy,
        agent_strategy=None,
        misc_config=None,
        device='cpu',
        guided_decoding_manager=guided_manager,
    )

    assert spec_agent.proposer is proposer
    assert spec_agent.rejection_sampler is rejection_sampler
    assert spec_agent.guided_helper.manager is guided_manager
    assert proposer.guided_helper is spec_agent.guided_helper


def test_spec_agent_reset_runtime_state_discards_chunk_carry():
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

    agent = SpecModelAgent.__new__(SpecModelAgent)
    agent._prev_chunk_last = {'hidden_states': object()}

    agent.reset_runtime_state()

    assert agent._prev_chunk_last == {}


@pytest.mark.parametrize(
    ('is_dummy', 'expected_events'),
    [
        pytest.param(False, [
            'build_context',
            'kv_restore',
            'state_restore',
            'update_model_metas',
            'prepare_inputs',
            'model_forward',
            'kv_save',
            'state_save',
        ], id='real-forward'),
        pytest.param(True, [
            'build_context',
            'update_model_metas',
            'prepare_inputs',
            'model_forward',
        ], id='dummy-forward'),
    ],
)
def test_model_forward_orders_kv_and_state_checkpoint_copies(monkeypatch, is_dummy, expected_events):
    from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    copy_calls = []
    restore_plan = torch.tensor([[1], [2]])
    save_plan = torch.tensor([[3], [4]])

    class _ContextManager:

        def build_context(self, **kwargs):
            events.append('build_context')
            return SimpleNamespace(q_seqlens=torch.tensor([1]),
                                   position_ids=torch.tensor([0]),
                                   is_model_meta_updated=False)

        def context(self, context):
            return nullcontext()

    class _Model:
        ctx_mgr = _ContextManager()

        def update_model_metas(self, **kwargs):
            events.append('update_model_metas')
            return []

        def prepare_inputs_for_generation(self, **kwargs):
            events.append('prepare_inputs')
            return {}

        def __call__(self, **kwargs):
            events.append('model_forward')
            return {'hidden_states': torch.tensor([0])}

    class _CacheEngine:
        cache_config = SimpleNamespace(quant_policy=0)
        gpu_cache = object()
        block_caches = {}

        def copy_logical_blocks(self, plan):
            copy_calls.append(('kv', plan))
            events.append('kv_restore' if plan is restore_plan else 'kv_save')

    class _StateCacheEngine:
        state_caches = object()
        named_state_caches = {}

        def copy_slots(self, src, dst):
            copy_calls.append(('state', src, dst))
            events.append('state_restore' if src == (5, ) else 'state_save')

    inputs = SimpleNamespace(
        is_dummy=is_dummy,
        state_offsets=None,
        seq_length=torch.tensor([1]),
    )
    cache_inputs = CacheCheckpointInputs(
        kv_restore_plan=restore_plan,
        kv_save_plan=save_plan,
        state_restore_plan=((5, ), (6, )),
        state_save_plan=((7, ), (8, )),
    )

    monkeypatch.setattr(agent_module, 'step_ctx_manager', lambda ctx_mgr: nullcontext())
    monkeypatch.setattr(agent_module.torch.cuda, 'stream', lambda stream: nullcontext())

    agent_module.model_forward(_Model(),
                               inputs,
                               object(),
                               _CacheEngine(),
                               _StateCacheEngine(),
                               stream=object(),
                               cache_inputs=cache_inputs)

    assert events == expected_events
    if is_dummy:
        assert copy_calls == []
    else:
        assert copy_calls[0][0] == 'kv'
        assert copy_calls[0][1] is restore_plan
        assert copy_calls[1] == ('state', (5, ), (6, ))
        assert copy_calls[2][0] == 'kv'
        assert copy_calls[2][1] is save_plan
        assert copy_calls[3] == ('state', (7, ), (8, ))


def test_inputs_preprocess_transfers_cache_inputs_and_keeps_host_ref(monkeypatch, event_loop):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    calls = []
    device_cache_inputs = object()

    class _HostCacheInputs:

        def to_device(self, device, non_blocking=False):
            calls.append((device, non_blocking))
            return device_cache_inputs

    class _Event:

        def record(self):
            calls.append('record')

    model_agent = _make_agent_with_queues()
    model_agent.out_stream = object()
    host_cache_inputs = _HostCacheInputs()
    monkeypatch.setattr(agent_module.torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(agent_module.torch.cuda, 'Event', _Event)

    task = event_loop.create_task(model_agent._async_loop_inputs_preprocess())
    model_agent._pre_in_que.put_nowait({'cache_inputs': host_cache_inputs})
    forward_inputs = event_loop.run_until_complete(asyncio.wait_for(model_agent._in_que.get(), timeout=1))
    task.cancel()
    event_loop.run_until_complete(asyncio.gather(task, return_exceptions=True))

    transfer = forward_inputs.pop(agent_module._H2D_TRANSFER_KEY)
    assert calls == [('cuda', True), 'record']
    assert forward_inputs['cache_inputs'] is device_cache_inputs
    assert transfer.refs['cache_inputs'] is host_cache_inputs


def test_record_forward_input_stream_uses_payload_protocol():
    from lmdeploy.pytorch.engine.model_agent.agent import _record_forward_input_stream

    recorded = []

    class _CudaTensor(torch.Tensor):

        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.empty(1), False)

        @property
        def is_cuda(self):
            return True

        def record_stream(self, stream):
            recorded.append(('tensor', id(self), stream))

    class _Payload:

        def record_stream(self, stream):
            recorded.append(('payload', id(self), stream))

    stream = object()
    tensor = _CudaTensor()
    nested_tensor = _CudaTensor()
    payload = _Payload()
    cache_payload = _Payload()
    forward_inputs = {
        'inputs': payload,
        'delta': tensor,
        'cache_inputs': cache_payload,
        'unowned_container': {'tensor': nested_tensor},
    }

    _record_forward_input_stream(forward_inputs, stream)

    assert recorded == [
        ('payload', id(payload), stream),
        ('tensor', id(tensor), stream),
        ('payload', id(cache_payload), stream),
    ]


def test_record_forward_input_stream_requires_payload_protocol():
    from lmdeploy.pytorch.engine.model_agent.agent import _record_forward_input_stream

    with pytest.raises(TypeError, match=r"H2D input 'sampling_inputs'.*record_stream"):
        _record_forward_input_stream({'sampling_inputs': object()}, object())


def test_strategy_inputs_record_stream_tensor_fields():
    from lmdeploy.pytorch.engine.model_agent.agent import _record_forward_input_stream
    from lmdeploy.pytorch.strategies.ar.model_agent import ARStoppingCriteria
    from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs

    recorded = []

    class _CudaTensor(torch.Tensor):

        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.empty(1), False)

        @property
        def is_cuda(self):
            return True

        def record_stream(self, stream):
            recorded.append((id(self), stream))

    stream = object()
    extra_tensor = _CudaTensor()
    stopping_tensor = _CudaTensor()
    extra_inputs = ARSpecExtraInputs(target_logits=extra_tensor)
    stopping_criteria = ARStoppingCriteria(num_appendable_ids=stopping_tensor)

    _record_forward_input_stream(
        {
            'extra_inputs': extra_inputs,
            'stopping_criteria': stopping_criteria,
        }, stream)

    assert recorded == [
        (id(stopping_tensor), stream),
        (id(extra_tensor), stream),
    ]


def test_background_records_h2d_inputs_after_wait_and_before_forward(monkeypatch, event_loop):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    transfer = SimpleNamespace(event=object())
    forward_done = asyncio.Event()

    class _Stream:

        def wait_event(self, event):
            assert event is transfer.event
            events.append('wait')

    class _InputMaker:

        def __init__(self):
            self.sent = False

        async def get(self):
            if not self.sent:
                self.sent = True
                return {
                    agent_module._H2D_TRANSFER_KEY: transfer,
                    'inputs': 'device-inputs',
                }
            await asyncio.Future()

        def step(self):
            events.append('step')

    model_agent = _make_agent_with_queues()
    model_agent.stream = _Stream()
    model_agent.all_context = nullcontext
    model_agent._keep_h2d_transfer = lambda item: events.append('keep')

    async def _async_step(**forward_inputs):
        assert forward_inputs == {'inputs': 'device-inputs'}
        events.append('forward')
        forward_done.set()

    model_agent._async_step = _async_step
    monkeypatch.setattr(agent_module, 'build_inputs_maker', lambda agent: _InputMaker())
    monkeypatch.setattr(agent_module.torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(agent_module, '_record_forward_input_stream',
                        lambda inputs, stream: events.append('record'))

    task = event_loop.create_task(model_agent._async_loop_background())
    event_loop.run_until_complete(asyncio.wait_for(forward_done.wait(), timeout=1))
    task.cancel()
    event_loop.run_until_complete(asyncio.gather(task, return_exceptions=True))

    assert events == ['keep', 'wait', 'record', 'forward', 'step']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA allocator')
def test_record_forward_input_stream_defers_origin_stream_reuse():
    from lmdeploy.pytorch.engine.model_agent.agent import _record_forward_input_stream

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    h2d_stream = torch.cuda.Stream()
    forward_stream = torch.cuda.Stream()
    h2d_event = torch.cuda.Event()
    numel = 1024 * 1024

    with torch.cuda.stream(h2d_stream):
        tensor = torch.full((numel, ), 7, dtype=torch.uint8, device='cuda')
        h2d_event.record()
    original_ptr = tensor.data_ptr()

    forward_stream.wait_event(h2d_event)
    _record_forward_input_stream({'delta': tensor}, forward_stream)
    with torch.cuda.stream(forward_stream):
        torch.cuda._sleep(10_000_000)
        observed = tensor.clone()

    del tensor
    with torch.cuda.stream(h2d_stream):
        replacement = torch.zeros((numel, ), dtype=torch.uint8, device='cuda')

    assert replacement.data_ptr() != original_ptr
    forward_stream.synchronize()
    h2d_stream.synchronize()
    assert torch.all(observed == 7)


def test_async_model_forward_preserves_cache_inputs_through_forward_impl():
    from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

    model_inputs = SimpleNamespace(
        is_dummy=False, is_decoding=False, logits_indices=None, seq_logit_length=None)
    cache_inputs = CacheCheckpointInputs(kv_restore_plan=torch.tensor([[1], [2]]))
    hidden_states = torch.ones(1, 1, 2)
    seen = []

    class _SpecAgent:

        def update_main_model_outputs(self, output, inputs):
            assert inputs is model_inputs
            return output['hidden_states'], output

    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.memdecode_agent = None
    agent.spec_agent = _SpecAgent()
    agent._forward_impl = lambda inputs, cache_inputs=None: (
        seen.append((inputs, cache_inputs)) or {
            'hidden_states': hidden_states
        })
    agent.get_logits = lambda hidden: hidden

    output = asyncio.run(agent._async_model_forward(model_inputs, return_logits=True, cache_inputs=cache_inputs))

    assert seen == [(model_inputs, cache_inputs)]
    assert output['logits'] is hidden_states


class TestDrainQueues:

    def test_drain_empty_queues(self):
        """Draining empty queues should be a no-op."""
        agent = _make_agent_with_queues()
        agent._drain_queues()
        assert agent._pre_in_que.empty()
        assert agent._in_que.empty()
        assert agent._out_que.empty()

    def test_drain_removes_all_items(self):
        """All items in every queue should be discarded."""
        agent = _make_agent_with_queues()
        for i in range(5):
            agent._pre_in_que.put_nowait(f'pre_{i}')
            agent._in_que.put_nowait(f'in_{i}')
            agent._out_que.put_nowait(f'out_{i}')

        agent._drain_queues()

        assert agent._pre_in_que.empty()
        assert agent._in_que.empty()
        assert agent._out_que.empty()

    def test_drain_skips_none_queues(self):
        """Queues that are None (before start()) should be skipped."""
        agent = _make_agent_with_queues()
        agent._pre_in_que = None
        agent._in_que = None
        # _out_que is still a real queue with items
        agent._out_que.put_nowait('stale')

        agent._drain_queues()

        assert agent._out_que.empty()

    def test_drain_prevents_stale_output_after_sleep(self):
        """Stale outputs left in _out_que before sleep must not be returned by
        get_output_async after wakeup.

        This is the exact bug scenario: a prefetch forward completes
        while the engine loop is draining for sleep. The output is put
        into _out_que but never consumed. After wakeup, a new forward
        runs, and get_output_async would return the stale output
        (paired with wrong model_inputs), causing a split size error.
        """
        agent = _make_agent_with_queues()

        # Simulate stale forward data left in queues from before sleep
        agent._pre_in_que.put_nowait('stale_inputs')
        agent._in_que.put_nowait('stale_inputs_cuda')
        agent._out_que.put_nowait('stale_output')

        # Sleep drains the queues
        agent._drain_queues()

        # After wakeup, a new forward is sent
        agent._pre_in_que.put_nowait('new_inputs')

        # The stale output must be gone — only new data should exist
        assert agent._out_que.empty()
        assert agent._pre_in_que.qsize() == 1
        assert agent._pre_in_que.get_nowait() == 'new_inputs'

    def test_get_output_async_returns_new_output_after_drain(self, event_loop):
        """After drain, get_output_async should only return fresh outputs.

        We use a simple wrapper that reads from _out_que directly, since the real get_output_async expects (output,
        cuda_event) tuples which require a GPU.
        """

        async def _read_queue(q):
            return await asyncio.wait_for(q.get(), timeout=1.0)

        agent = _make_agent_with_queues()

        # Stale output from before sleep
        agent._out_que.put_nowait('stale')

        # Drain (simulates sleep)
        agent._drain_queues()

        # Fresh output from post-wakeup forward
        agent._out_que.put_nowait('fresh')

        result = event_loop.run_until_complete(_read_queue(agent._out_que))
        assert result == 'fresh'

    def test_drain_only_removes_current_items(self):
        """Items added after drain should not be affected."""
        agent = _make_agent_with_queues()
        agent._out_que.put_nowait('old')

        agent._drain_queues()

        agent._out_que.put_nowait('new')
        assert agent._out_que.qsize() == 1
        assert agent._out_que.get_nowait() == 'new'


class TestDPForwardInputsMaker:

    @staticmethod
    def _make_ready_event():

        class _ReadyEvent:

            def query(self):
                return True

        return _ReadyEvent()

    @staticmethod
    def _make_maker(is_sleeping=False, dummy_forward_inputs=None):
        from lmdeploy.pytorch.engine.model_agent.inputs_maker import DPForwardInputsMaker

        maker = DPForwardInputsMaker.__new__(DPForwardInputsMaker)
        maker.model_agent = SimpleNamespace(state=SimpleNamespace(is_sleeping=is_sleeping))
        maker._pre_in_que = asyncio.Queue()
        maker._in_que = asyncio.Queue()
        maker._ready_event = TestDPForwardInputsMaker._make_ready_event()

        async def _gather_has_inputs(has_inputs=False):
            return has_inputs

        def _make_dummy_forward_inputs():
            if dummy_forward_inputs is not None:
                return dummy_forward_inputs
            raise AssertionError('pending real input must not be replaced with a dummy')

        maker._gather_has_inputs = _gather_has_inputs
        maker._make_dummy_forward_inputs = _make_dummy_forward_inputs
        return maker

    def test_get_waits_for_queued_preprocess_input(self):
        async def _run():
            maker = self._make_maker()
            maker._pre_in_que.put_nowait({'inputs': 'queued'})

            task = asyncio.create_task(maker.get())
            await asyncio.sleep(0.01)
            assert not task.done()

            real_inputs = {'inputs': 'real'}
            maker._pre_in_que.get_nowait()
            maker._in_que.put_nowait(real_inputs)

            assert await asyncio.wait_for(task, timeout=1.0) is real_inputs

        asyncio.run(_run())

    def test_get_yields_for_worker_forward_rpc_before_dummy(self):

        async def _run():
            maker = self._make_maker()
            real_inputs = {'inputs': 'real'}

            async def _enqueue_after_model_agent_yields():
                await asyncio.sleep(0)
                maker._pre_in_que.put_nowait({'inputs': 'queued'})
                await asyncio.sleep(0)
                maker._pre_in_que.get_nowait()
                maker._in_que.put_nowait(real_inputs)

            enqueue_task = asyncio.create_task(_enqueue_after_model_agent_yields())

            assert await asyncio.wait_for(maker.get(), timeout=1.0) is real_inputs
            await asyncio.wait_for(enqueue_task, timeout=1.0)

        asyncio.run(_run())

    def test_get_uses_dummy_for_sleeping_preprocess_queue(self):
        async def _run():
            dummy_inputs = {'inputs': 'sleep_dummy'}
            maker = self._make_maker(is_sleeping=True, dummy_forward_inputs=dummy_inputs)
            maker._pre_in_que.put_nowait({'inputs': 'stale'})

            assert await asyncio.wait_for(maker.get(), timeout=1.0) is dummy_inputs
            assert maker._pre_in_que.qsize() == 1
            assert maker._in_que.qsize() == 0

        asyncio.run(_run())

    def test_get_uses_dummy_for_sleeping_ready_queue(self):
        async def _run():
            dummy_inputs = {'inputs': 'sleep_dummy'}
            maker = self._make_maker(is_sleeping=True, dummy_forward_inputs=dummy_inputs)
            maker._in_que.put_nowait({'inputs': 'stale_ready'})

            assert await asyncio.wait_for(maker.get(), timeout=1.0) is dummy_inputs
            assert maker._pre_in_que.qsize() == 0
            assert maker._in_que.qsize() == 1

        asyncio.run(_run())

    def test_get_attaches_dummy_inputs_to_connector_only_step(self):
        async def _run():
            metadata = object()
            connector_inputs = {
                'inputs': None,
                'delta': None,
                'extra_inputs': None,
                'return_logits': False,
                'kv_connector_metadata': metadata,
            }
            dummy_inputs = {
                'inputs': 'connector_dummy',
                'extra_inputs': 'dummy_extra',
                'return_logits': True,
            }
            maker = self._make_maker(dummy_forward_inputs=dummy_inputs)
            maker._in_que.put_nowait(connector_inputs)

            result = await asyncio.wait_for(maker.get(), timeout=1.0)

            assert result is connector_inputs
            assert result['inputs'] == 'connector_dummy'
            assert result['extra_inputs'] == 'dummy_extra'
            assert result['return_logits'] is True
            assert result['kv_connector_metadata'] is metadata

        asyncio.run(_run())


class TestDPForwardMeta:

    def test_field_names_follow_enabled_features(self):
        from lmdeploy.pytorch.engine.model_agent.dp_utils import DPForwardMeta

        assert DPForwardMeta.field_names(is_spec_enabled=False, is_microbatch_enabled=False) == (
            'is_decoding',
            'is_dummy',
            'num_tokens',
            'is_sleeping',
            'batch_size',
        )
        assert DPForwardMeta.field_names(is_spec_enabled=True, is_microbatch_enabled=False) == (
            'is_decoding',
            'is_dummy',
            'num_tokens',
            'is_sleeping',
            'batch_size',
            'draft_num_tokens',
        )
        assert DPForwardMeta.field_names(is_spec_enabled=False, is_microbatch_enabled=True) == (
            'is_decoding',
            'is_dummy',
            'num_tokens',
            'is_sleeping',
            'batch_size',
            'enable_microbatch',
        )

    def test_values_omit_disabled_optional_fields(self):
        from lmdeploy.pytorch.engine.model_agent.dp_utils import DPForwardMeta

        meta = DPForwardMeta(is_decoding=True,
                             is_dummy=False,
                             num_tokens=8,
                             is_sleeping=True,
                             batch_size=2,
                             draft_num_tokens=7,
                             enable_microbatch=True)

        assert meta.values(is_spec_enabled=False, is_microbatch_enabled=False) == [1, 0, 8, 1, 2]
        assert meta.values(is_spec_enabled=True, is_microbatch_enabled=False) == [1, 0, 8, 1, 2, 7]
        assert meta.values(is_spec_enabled=False, is_microbatch_enabled=True) == [1, 0, 8, 1, 2, 1]
        assert meta.values(is_spec_enabled=True, is_microbatch_enabled=True) == [1, 0, 8, 1, 2, 7, 1]

    def test_gathered_meta_deserializes_named_columns(self):
        from lmdeploy.pytorch.engine.model_agent.dp_utils import GatheredDPForwardMeta

        values = torch.tensor([
            [1, 0, 8, 0, 2, 7, 1],
            [1, 0, 6, 1, 3, 6, 1],
        ])
        gathered = GatheredDPForwardMeta.from_values(values, is_spec_enabled=True, is_microbatch_enabled=True)

        assert gathered.global_is_decoding is True
        assert gathered.is_all_dummy is False
        assert gathered.is_all_sleeping is False
        assert gathered.all_num_tokens == [8, 6]
        assert gathered.all_batch_sizes == [2, 3]
        assert gathered.all_draft_num_tokens == [7, 6]
        assert gathered.global_enable_microbatch is True

    def test_gathered_meta_supports_base_schema(self):
        from lmdeploy.pytorch.engine.model_agent.dp_utils import GatheredDPForwardMeta

        values = torch.tensor([
            [1, 1, 4, 1, 2],
            [0, 1, 5, 1, 1],
        ])
        gathered = GatheredDPForwardMeta.from_values(values, is_spec_enabled=False, is_microbatch_enabled=False)

        assert gathered.global_is_decoding is False
        assert gathered.is_all_dummy is True
        assert gathered.is_all_sleeping is True
        assert gathered.all_num_tokens == [4, 5]
        assert gathered.all_batch_sizes == [2, 1]
        assert gathered.draft_num_tokens is None
        assert gathered.enable_microbatch is None


class TestResetGraphRunner:

    def test_model_agent_reset_graph_runner_uses_all_context(self):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

        events = []

        class _PatchedModel:

            def reset(self):
                events.append('main_reset')

        class _SpecAgent:

            def reset_graph_runner(self):
                events.append('spec_reset')

        class _MemDecodeAgent:

            def reset_graph_runner(self):
                events.append('memdecode_reset')

        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.patched_model = _PatchedModel()
        agent.spec_agent = _SpecAgent()
        agent.memdecode_agent = _MemDecodeAgent()
        agent._prev_chunk_output = {'model_metas': object()}
        agent._prev_chunk_last_logit = torch.ones(1, 2)

        @contextmanager
        def _all_context():
            events.append('enter_all_context')
            yield
            events.append('exit_all_context')

        agent.all_context = _all_context

        agent.reset_graph_runner()

        assert events == [
            'enter_all_context',
            'main_reset',
            'spec_reset',
            'memdecode_reset',
            'exit_all_context',
        ]
        assert agent._prev_chunk_output is None
        assert agent._prev_chunk_last_logit is None

    def test_spec_agent_reset_graph_runner_uses_draft_context(self):
        from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

        events = []

        class _Model:

            def reset(self):
                events.append('reset')

        agent = SpecModelAgent.__new__(SpecModelAgent)
        agent.proposer = type('Proposer', (), {'model': _Model()})()
        agent._prev_chunk_last = {'hidden_states': torch.ones(1, 1, 2)}

        @contextmanager
        def _draft_context():
            events.append('enter_draft_context')
            yield
            events.append('exit_draft_context')

        agent.draft_context = _draft_context

        agent.reset_graph_runner()

        assert events == [
            'enter_draft_context',
            'reset',
            'exit_draft_context',
        ]
        assert agent._prev_chunk_last == {}


class TestModelAgentWakeup:

    @staticmethod
    def _make_level2_agent(rebuilt_block_nbytes):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

        events = []
        cache_config = object()
        spec_cache_config = object()

        class _Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(0, device='meta'))

        class _GraphRunner:

            def __init__(self, model):
                self.model = model

            def get_model(self):
                return self.model

        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.memdecode_agent = None
        agent.patched_model = _GraphRunner(_Model())
        agent.spec_agent = SimpleNamespace(get_model=lambda: None, cache_config=spec_cache_config)
        agent.cache_config = cache_config
        agent.misc_config = SimpleNamespace(empty_init=False)
        agent._cache_plan_block_nbytes = (256, 128)

        def _build_model():
            events.append('build_model')
            agent.patched_model = _Model()

        def _build_cache_plans(received_cache_config, received_spec_cache_config):
            events.append(('build_cache_plans', received_cache_config, received_spec_cache_config))
            agent._cache_plan_block_nbytes = rebuilt_block_nbytes
            target_nbytes, spec_nbytes = rebuilt_block_nbytes
            return target_nbytes, spec_nbytes, 0

        agent.build_model = _build_model
        agent.build_cache_plans = _build_cache_plans
        agent.build_graph_runner = lambda: events.append('build_graph_runner')
        return agent, events, cache_config, spec_cache_config

    def test_level2_wakeup_rebuilds_cache_plans_before_graph_runner(self):
        agent, events, cache_config, spec_cache_config = self._make_level2_agent((256, 128))

        agent.wakeup(['weights'])

        assert events == [
            'build_model',
            ('build_cache_plans', cache_config, spec_cache_config),
            'build_graph_runner',
        ]
        assert agent.misc_config.empty_init is False

    def test_level2_wakeup_rejects_changed_cache_block_sizes(self):
        agent, events, _, _ = self._make_level2_agent((512, 128))

        with pytest.raises(RuntimeError, match=r'expected target/draft \(256, 128\), got \(512, 128\)'):
            agent.wakeup(['weights'])

        assert 'build_graph_runner' not in events
        assert agent.misc_config.empty_init is False

    def test_sleep_clears_middle_chunk_carryover_state(self, event_loop, monkeypatch):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent, SleepWakeupState
        from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

        events = []

        class _Moveable:

            def __init__(self, name):
                self.name = name

            def to(self, *args, **kwargs):
                events.append((self.name, 'to', args, kwargs))
                return self

        class _PatchedModel:

            def __init__(self):
                self.model = _Moveable('main_model')

            def reset(self):
                events.append('main_reset')

            def get_model(self):
                return self.model

        class _SpecGraphRunner:

            def __init__(self):
                self.model = _Moveable('spec_model')

            def reset(self):
                events.append('spec_reset')

            def get_model(self):
                return self.model

        class _StrategyFactory:

            def build_step_inputs(self):
                events.append('build_step_inputs')
                return {'fresh': 'step_inputs'}

        spec_agent = SpecModelAgent.__new__(SpecModelAgent)
        spec_agent.proposer = type('Proposer', (), {'model': _SpecGraphRunner()})()
        spec_agent._prev_chunk_last = {'hidden_states': torch.ones(1, 1, 2)}
        spec_agent.cache_engine = object()

        @contextmanager
        def _draft_context():
            events.append('enter_draft_context')
            yield
            events.append('exit_draft_context')

        spec_agent.draft_context = _draft_context

        model_agent = BaseModelAgent.__new__(BaseModelAgent)
        model_agent.state = SleepWakeupState()
        model_agent.dist_config = SimpleNamespace(dp=1)
        model_agent.memdecode_agent = None
        model_agent.kv_connector = None
        model_agent.cache_engine = object()
        model_agent.state_cache_engine = object()
        model_agent.patched_model = _PatchedModel()
        model_agent.spec_agent = spec_agent
        model_agent.strategy_factory = _StrategyFactory()
        model_agent.step_inputs = {'stale': 'step_inputs'}
        model_agent._prev_chunk_output = {'model_metas': object()}
        model_agent._prev_chunk_last_logit = torch.ones(1, 2)
        model_agent._pre_in_que = asyncio.Queue()
        model_agent._in_que = asyncio.Queue()
        model_agent._out_que = asyncio.Queue()
        model_agent._pending_h2d_transfers = deque()
        model_agent._pre_in_que.put_nowait('stale_middle_chunk_input')
        model_agent._in_que.put_nowait('stale_middle_chunk_cuda_input')
        model_agent._out_que.put_nowait('stale_middle_chunk_output')
        model_agent._update_params_ipc_tensor = object()
        model_agent._update_params_ipc_event = object()

        @contextmanager
        def _all_context():
            events.append('enter_all_context')
            yield
            events.append('exit_all_context')

        model_agent.all_context = _all_context
        monkeypatch.setattr(torch.cuda, 'synchronize', lambda: events.append('cuda_synchronize'))
        monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: events.append('cuda_empty_cache'))

        event_loop.run_until_complete(model_agent.sleep(level=1))

        assert model_agent._prev_chunk_output is None
        assert model_agent._prev_chunk_last_logit is None
        assert model_agent.step_inputs == {'fresh': 'step_inputs'}
        assert spec_agent._prev_chunk_last == {}
        assert model_agent.cache_engine is None
        assert model_agent.state_cache_engine is None
        assert spec_agent.cache_engine is None
        assert model_agent._pre_in_que.empty()
        assert model_agent._in_que.empty()
        assert model_agent._out_que.empty()
        assert model_agent._update_params_ipc_tensor is None
        assert model_agent._update_params_ipc_event is None
        assert 'main_reset' in events
        assert 'spec_reset' in events
        assert 'build_step_inputs' in events

    def test_dp_kv_cache_wakeup_warms_before_releasing_forward_task(self):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent, SleepWakeupState

        events = []

        model_agent = BaseModelAgent.__new__(BaseModelAgent)
        model_agent.state = SleepWakeupState()
        model_agent.state.is_sleeping = True
        model_agent.dist_config = SimpleNamespace(dp=2)
        model_agent.memdecode_agent = None
        model_agent.build_cache_engine = lambda: events.append('build_cache_engine')

        def _warmup():
            events.append(('warmup', model_agent.state.is_sleeping, model_agent.state.to_wakeup.is_set()))

        model_agent.warmup = _warmup

        model_agent.wakeup(['kv_cache'])

        assert model_agent.state.is_sleeping is False
        assert model_agent.state.to_wakeup.is_set()
        assert events == [
            'build_cache_engine',
            ('warmup', True, False),
        ]


class TestMemDecodeModelAgentLifecycle:

    def _make_agent(self, enabled=True):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent, SleepWakeupState

        events = []

        class _MemDecodeAgent:

            def release(self):
                events.append('memdecode_release')

            def reset_graph_runner(self):
                events.append('memdecode_reset')

        class _SpecAgent:

            def reset_graph_runner(self):
                pass

        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.memdecode_agent = _MemDecodeAgent() if enabled else None
        agent.spec_agent = _SpecAgent()
        agent.state = SleepWakeupState()
        agent.dist_config = SimpleNamespace(dp=1)
        agent.patched_model = object()
        agent.kv_connector = None
        agent.cache_engine = object()
        agent.state_cache_engine = object()

        @contextmanager
        def _all_context():
            yield

        agent.all_context = _all_context
        return agent, events

    def test_sleep_raises_when_memdecode_enabled(self, event_loop):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

        agent, _ = self._make_agent(enabled=True)

        with pytest.raises(NotImplementedError, match='MemDecode sleep/wakeup is not supported yet.'):
            event_loop.run_until_complete(BaseModelAgent.sleep(agent))

    def test_wakeup_raises_when_memdecode_enabled(self):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

        agent, _ = self._make_agent(enabled=True)

        with pytest.raises(NotImplementedError, match='MemDecode sleep/wakeup is not supported yet.'):
            BaseModelAgent.wakeup(agent)

    def test_release_releases_memdecode_and_clears_base_resources(self, monkeypatch):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

        monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
        agent, events = self._make_agent(enabled=True)

        BaseModelAgent.release(agent)

        assert events == ['memdecode_reset', 'memdecode_release']
        assert agent.patched_model is None
        assert agent.cache_engine is None
        assert agent.state_cache_engine is None

    def test_async_model_forward_memdecode_fuses_sliced_logits(self, event_loop):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

        calls = []
        base_hidden = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
        memory_hidden = torch.arange(30, dtype=torch.float32).reshape(1, 5, 6)
        inputs = SimpleNamespace(
            seq_length=torch.tensor([2, 3]),
            is_chunk=False,
            is_dummy=False,
            is_decoding=False,
            logits_indices=None,
            seq_logit_length=None,
        )

        class _MemDecodeAgent:

            async def fuse_with_base(self, inputs, base_output, base_logits, postprocess_output):
                calls.append(('fuse_inputs', inputs))
                calls.append(('fuse_base_hidden_shape', tuple(base_output['hidden_states'].shape)))
                calls.append(('fuse_base_logits_shape', tuple(base_logits.shape)))
                memory_output = {
                    'hidden_states': memory_hidden.clone(),
                    'seq_length': inputs.seq_length,
                }
                memory_output = postprocess_output(memory_output, inputs)
                calls.append(('fuse_memory_hidden_shape', tuple(memory_output['hidden_states'].shape)))
                fused = base_logits + memory_output['hidden_states'].sum(dim=-1, keepdim=True)
                base_output['logits'] = fused
                return base_output

        class _Strategy:

            def slice_outputs(self, hidden_states, seq_length):
                indices = seq_length.cumsum(0) - 1
                return hidden_states[indices]

        async def _base_forward(forward_inputs, cache_inputs=None):
            assert cache_inputs is None
            calls.append(('base_forward', forward_inputs))
            return {'hidden_states': base_hidden.clone(), 'seq_length': forward_inputs.seq_length}

        def _base_logits(hidden_states):
            calls.append(('base_logits_shape', tuple(hidden_states.shape)))
            return hidden_states.sum(dim=-1, keepdim=True)

        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.memdecode_agent = _MemDecodeAgent()
        agent.agent_strategy = _Strategy()
        agent.async_forward = _base_forward
        agent.get_logits = _base_logits

        output = event_loop.run_until_complete(BaseModelAgent._async_model_forward(agent, inputs, return_logits=False))

        assert calls == [
            ('base_forward', inputs),
            ('base_logits_shape', (1, 2, 4)),
            ('fuse_inputs', inputs),
            ('fuse_base_hidden_shape', (1, 2, 4)),
            ('fuse_base_logits_shape', (1, 2, 1)),
            ('fuse_memory_hidden_shape', (1, 2, 6)),
        ]
        assert torch.equal(output['logits'], torch.tensor([[[73.], [229.]]]))
        assert 'all_routed_experts' not in output

    def test_async_model_forward_memdecode_rejects_returned_logits(self, event_loop):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

        class _MemDecodeAgent:
            pass

        async def _base_forward(_inputs):
            raise AssertionError('base forward should not run')

        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.memdecode_agent = _MemDecodeAgent()
        agent.async_forward = _base_forward
        inputs = SimpleNamespace()

        with pytest.raises(RuntimeError, match='MemDecode does not support returned prompt logits yet.'):
            event_loop.run_until_complete(BaseModelAgent._async_model_forward(agent, inputs, return_logits=True))

    def test_async_step_swaps_memdecode_cache_with_base_cache(self, event_loop, monkeypatch):
        import lmdeploy.pytorch.engine.model_agent.agent as agent_module
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

        calls = []
        swap_in_map = {1: 2}
        swap_out_map = {3: 4}

        class _StopAfterSwap(Exception):
            pass

        class _MemDecodeAgent:

            cache_engine = 'memory_cache'

        class _DistManager:

            def current_context(self):
                return SimpleNamespace(dist_config=SimpleNamespace(attn_tp=1, dp=1))

        def _cache_swapping(cache_engine, swap_in_map=None, swap_out_map=None):
            calls.append((cache_engine, swap_in_map, swap_out_map))

        async def _async_model_forward(_inputs, return_logits, cache_inputs=None):
            assert cache_inputs is None
            raise _StopAfterSwap

        monkeypatch.setattr(agent_module, 'get_dist_manager', lambda: _DistManager())
        monkeypatch.setattr(agent_module, 'cache_swapping', _cache_swapping)

        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.rank = 0
        agent.kv_connector = None
        agent.cache_engine = 'base_cache'
        agent.memdecode_agent = _MemDecodeAgent()
        agent._async_model_forward = _async_model_forward
        inputs = SimpleNamespace(is_dummy=True,
                                 is_decoding=False,
                                 input_ids=torch.tensor([1, 2]),
                                 seq_length=torch.tensor([2]),
                                 is_chunk=False,
                                 is_first_chunk=False,
                                 is_last_chunk=False,
                                 dp_meta=None)

        with pytest.raises(_StopAfterSwap):
            event_loop.run_until_complete(
                BaseModelAgent._async_step(agent,
                                           inputs,
                                           swap_in_map=swap_in_map,
                                           swap_out_map=swap_out_map,
                                           return_logits=False))

        assert calls == [
            ('base_cache', swap_in_map, swap_out_map),
            ('memory_cache', swap_in_map, swap_out_map),
        ]
