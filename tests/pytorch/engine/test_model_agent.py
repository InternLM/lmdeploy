# Copyright (c) OpenMMLab. All rights reserved.
import asyncio

import pytest


@pytest.fixture
def event_loop():
    old_loop = asyncio.get_event_loop()
    new_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(new_loop)
        yield new_loop
    finally:
        new_loop.stop()
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
