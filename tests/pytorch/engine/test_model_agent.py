# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch


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


def test_build_spec_agent_allows_guided_spec_followers_without_proposer():
    from lmdeploy.pytorch.config import DistConfig, SpecDecodeConfig
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
        backend_config=None,
        dist_ctx=DistContext(rank=1, dist_config=DistConfig(tp=2)),
        inputs_strategy=None,
        agent_strategy=None,
        misc_config=None,
        device='cpu',
        guided_decoding_manager=guided_manager,
    )
    assert spec_agent.is_enabled()
    assert spec_agent.proposer is None
    assert not hasattr(spec_agent, 'guided_helper')


def test_build_spec_agent_shares_guided_helper_with_proposer(monkeypatch):
    import lmdeploy.pytorch.spec_decode.spec_agent as spec_agent_mod
    from lmdeploy.pytorch.config import DistConfig, SpecDecodeConfig
    from lmdeploy.pytorch.distributed import DistContext
    from lmdeploy.pytorch.spec_decode import build_spec_agent

    guided_manager = object()
    proposer = SimpleNamespace(guided_helper=None)
    monkeypatch.setattr(spec_agent_mod, 'build_specdecode_proposer', lambda *args, **kwargs: proposer)
    inputs_strategy = SimpleNamespace(create_make_dummy_meta=lambda model_config: None)
    specdecode_config = SpecDecodeConfig(
        model='draft-model',
        method='deepseek_mtp',
        dist_config=DistConfig(),
        num_speculative_tokens=3,
    )

    spec_agent = build_spec_agent(
        specdecode_config,
        backend_config=None,
        dist_ctx=DistContext(rank=0, dist_config=DistConfig(tp=2)),
        inputs_strategy=inputs_strategy,
        agent_strategy=None,
        misc_config=None,
        device='cpu',
        guided_decoding_manager=guided_manager,
    )

    assert spec_agent.proposer is proposer
    assert spec_agent.guided_helper.manager is guided_manager
    assert proposer.guided_helper is spec_agent.guided_helper


def test_spec_agent_reset_runtime_state_discards_chunk_carry():
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

    agent = SpecModelAgent.__new__(SpecModelAgent)
    agent._prev_chunk_last = {'hidden_states': object()}

    agent.reset_runtime_state()

    assert agent._prev_chunk_last == {}


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
        inputs = SimpleNamespace(seq_length=torch.tensor([2, 3]), is_chunk=False)

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

        async def _base_forward(forward_inputs):
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

        async def _async_model_forward(_inputs, return_logits):
            raise _StopAfterSwap

        monkeypatch.setattr(agent_module, 'get_dist_manager', lambda: _DistManager())
        monkeypatch.setattr(agent_module, 'cache_swapping', _cache_swapping)

        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.rank = 0
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
