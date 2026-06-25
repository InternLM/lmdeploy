# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
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

    def test_spec_agent_reset_graph_runner_uses_draft_context(self):
        from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent

        events = []

        class _Model:

            def reset(self):
                events.append('reset')

        agent = SpecModelAgent.__new__(SpecModelAgent)
        agent.proposer = type('Proposer', (), {'model': _Model()})()

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


class TestModelAgentWakeup:

    def test_dp_kv_cache_wakeup_warms_before_releasing_forward_task(self):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent, SleepWakeupState

        events = []

        model_agent = BaseModelAgent.__new__(BaseModelAgent)
        model_agent.state = SleepWakeupState()
        model_agent.state.is_sleeping = True
        model_agent.dist_config = SimpleNamespace(dp=2)
        model_agent.memdecode_agent = SimpleNamespace(is_enabled=lambda: False)
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

            def is_enabled(self):
                return enabled

            def release(self):
                events.append('memdecode_release')

            def reset_graph_runner(self):
                events.append('memdecode_reset')

        class _SpecAgent:

            def reset_graph_runner(self):
                pass

        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.memdecode_agent = _MemDecodeAgent()
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

            def is_enabled(self):
                return True

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
                routed = {'selected_experts': torch.tensor([1, 0])}
                base_output['logits'] = fused
                base_output['all_routed_experts'] = routed
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
        assert torch.equal(output['all_routed_experts']['selected_experts'], torch.tensor([1, 0]))

    @pytest.mark.parametrize('is_chunk', [False, True])
    def test_async_model_forward_memdecode_rejects_returned_logits(self, event_loop, is_chunk):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

        class _MemDecodeAgent:

            def is_enabled(self):
                return True

        async def _base_forward(_inputs):
            raise AssertionError('base forward should not run')

        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.memdecode_agent = _MemDecodeAgent()
        agent.async_forward = _base_forward
        inputs = SimpleNamespace(is_chunk=is_chunk)

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

            def is_enabled(self):
                return True

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
