# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent


def _agent(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent._out_que = asyncio.Queue()
    agent.out_stream = object()
    agent._release_completed_h2d_transfers = Mock()
    return agent


def test_ready_output_stays_on_the_event_loop(monkeypatch):
    agent = _agent(monkeypatch)
    event = SimpleNamespace(query=lambda: True, synchronize=Mock(), wait=Mock())
    output = SimpleNamespace()
    source = SimpleNamespace(to_cpu=Mock(return_value=output))
    agent._out_que.put_nowait((source, event))
    before = time.time()

    assert asyncio.run(agent.get_output_async()) is output
    assert before <= output.new_token_timestamp <= time.time()
    event.synchronize.assert_not_called()
    event.wait.assert_called_once_with()
    source.to_cpu.assert_called_once_with()
    agent._release_completed_h2d_transfers.assert_called_once_with()


def test_pending_output_wait_allows_other_coroutines_to_run(monkeypatch):
    agent = _agent(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    event_loop_thread = threading.get_ident()
    ready = False

    def synchronize():
        nonlocal ready
        assert threading.get_ident() != event_loop_thread
        entered.set()
        assert release.wait(2)
        ready = True

    def to_cpu():
        assert ready
        assert threading.get_ident() == event_loop_thread
        return output

    event = SimpleNamespace(query=lambda: ready, synchronize=synchronize, wait=Mock())
    output = SimpleNamespace()
    source = SimpleNamespace(to_cpu=Mock(side_effect=to_cpu))
    agent._out_que.put_nowait((source, event))

    async def check():
        task = asyncio.create_task(agent.get_output_async())
        try:
            deadline = time.monotonic() + 2
            while not entered.is_set():
                assert time.monotonic() < deadline
                await asyncio.sleep(0)
            for _ in range(10):
                await asyncio.sleep(0)
                assert not task.done()
            source.to_cpu.assert_not_called()
            agent._release_completed_h2d_transfers.assert_not_called()
            release.set()
            assert await asyncio.wait_for(task, 2) is output
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())
    event.wait.assert_called_once_with()
    agent._release_completed_h2d_transfers.assert_called_once_with()


def test_wait_error_does_not_copy_or_publish_output(monkeypatch):
    agent = _agent(monkeypatch)
    event = SimpleNamespace(query=lambda: False,
                            synchronize=Mock(side_effect=RuntimeError('CUDA event failed')), wait=Mock())
    source = SimpleNamespace(to_cpu=Mock())
    agent._out_que.put_nowait((source, event))

    with pytest.raises(RuntimeError, match='CUDA event failed'):
        asyncio.run(asyncio.wait_for(agent.get_output_async(), 2))
    event.wait.assert_not_called()
    source.to_cpu.assert_not_called()
    agent._release_completed_h2d_transfers.assert_not_called()


def test_cancelled_wait_does_not_copy_output(monkeypatch):
    agent = _agent(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def synchronize():
        entered.set()
        assert release.wait(2)

    event = SimpleNamespace(query=lambda: False, synchronize=synchronize, wait=Mock())
    source = SimpleNamespace(to_cpu=Mock())
    agent._out_que.put_nowait((source, event))

    async def check():
        task = asyncio.create_task(agent.get_output_async())
        try:
            deadline = time.monotonic() + 2
            while not entered.is_set():
                assert time.monotonic() < deadline
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            source.to_cpu.assert_not_called()
            event.wait.assert_not_called()
            agent._release_completed_h2d_transfers.assert_not_called()
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())


def test_output_sentinel_finishes_without_cuda(monkeypatch):
    agent = _agent(monkeypatch)
    agent._out_que.put_nowait(None)
    assert asyncio.run(agent.get_output_async()) == {}
    agent._release_completed_h2d_transfers.assert_not_called()


@pytest.mark.parametrize('device_index', [0, 1])
def test_output_event_wait_uses_the_recording_cuda_device(device_index):
    if torch.cuda.device_count() <= device_index:
        pytest.skip('Requested CUDA device is not available')
    with torch.cuda.device(device_index):
        producer = torch.cuda.Stream()
        output_stream = torch.cuda.Stream()
        values = torch.empty(512, device=f'cuda:{device_index}')
        event = torch.cuda.Event()
        with torch.cuda.stream(producer):
            torch.cuda._sleep(40_000_000)
            values.fill_(7)
            event.record()
        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent._out_que = asyncio.Queue()
        agent.out_stream = output_stream
        agent._release_completed_h2d_transfers = Mock()

        def to_cpu():
            assert event.query()
            return SimpleNamespace(values=values.cpu())

        agent._out_que.put_nowait((SimpleNamespace(to_cpu=to_cpu), event))
        result = asyncio.run(agent.get_output_async())
        torch.testing.assert_close(result.values, torch.full((512, ), 7.))
        agent._release_completed_h2d_transfers.assert_called_once_with()
