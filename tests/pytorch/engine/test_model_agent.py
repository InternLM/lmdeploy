# Copyright (c) OpenMMLab. All rights reserved.
import asyncio

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
            'has_non_last_chunk',
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
                             has_non_last_chunk=True,
                             draft_num_tokens=7,
                             enable_microbatch=True)

        assert meta.values(is_spec_enabled=False, is_microbatch_enabled=False) == [1, 0, 8, 1, 2]
        assert meta.values(is_spec_enabled=True, is_microbatch_enabled=False) == [1, 0, 8, 1, 2, 1, 7]
        assert meta.values(is_spec_enabled=False, is_microbatch_enabled=True) == [1, 0, 8, 1, 2, 1]
        assert meta.values(is_spec_enabled=True, is_microbatch_enabled=True) == [1, 0, 8, 1, 2, 1, 7, 1]

    def test_gathered_meta_deserializes_named_columns(self):
        from lmdeploy.pytorch.engine.model_agent.dp_utils import GatheredDPForwardMeta

        values = torch.tensor([
            [1, 0, 8, 0, 2, 0, 7, 1],
            [1, 0, 6, 1, 3, 1, 6, 1],
        ])
        gathered = GatheredDPForwardMeta.from_values(values, is_spec_enabled=True, is_microbatch_enabled=True)

        assert gathered.global_is_decoding is True
        assert gathered.is_all_dummy is False
        assert gathered.is_all_sleeping is False
        assert gathered.all_num_tokens == [8, 6]
        assert gathered.all_batch_sizes == [2, 3]
        assert gathered.all_draft_num_tokens == [7, 6]
        assert gathered.dp_has_non_last_chunk is True
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
        assert gathered.has_non_last_chunk is None
        assert gathered.enable_microbatch is None
