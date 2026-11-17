import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.serve.core.async_engine import AsyncEngine
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.managers import SessionManager


@pytest.fixture(autouse=True)
def reset_metrics(monkeypatch):
    from lmdeploy.metrics.metrics_processor import metrics_processor
    from lmdeploy.metrics.stats import SchedulerStats

    monkeypatch.setattr(metrics_processor, 'scheduler_stats', SchedulerStats())


def _engine(handles):
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.epoch = 0
    engine.backend = 'pytorch'
    engine.backend_config = SimpleNamespace(enable_prefix_caching=False)
    engine.speculative_config = None
    engine.session_mgr = SessionManager()
    engine.session_mgr.build_request_handle_pool(SimpleNamespace(create_instance=Mock(side_effect=handles)),
                                                 len(handles))
    return engine


@pytest.mark.parametrize('operation', ['abort', 'abort_all', 'close'])
@pytest.mark.parametrize('api', ['logits', 'logits_with_sessions', 'ppl'])
def test_cancelled_scoring_does_not_dispatch_after_waiting_for_handle(api, operation, monkeypatch):
    async def run():
        handle = SimpleNamespace(async_cancel=AsyncMock(),
                                 async_stream_infer=Mock(side_effect=AssertionError('cancelled scoring dispatched')))
        engine = _engine([handle])
        manager = engine.session_mgr
        pool = manager.request_handle_pool
        held = await pool.get()
        waiting = asyncio.Event()
        original_get = pool.get

        async def get():
            waiting.set()
            return await original_get()

        monkeypatch.setattr(pool, 'get', get)
        if api == 'ppl':
            coro = engine.async_get_ppl([1, 2, 3])
        else:
            sessions = [manager.get()] if api == 'logits_with_sessions' else None
            coro = engine.async_get_logits([[1, 2, 3]], sessions=sessions)
        task = asyncio.create_task(coro)
        try:
            await asyncio.wait_for(waiting.wait(), 5)
            session, = manager.sessions.values()
            if operation == 'abort_all':
                await engine.stop_all_session()
            elif operation == 'close':
                await session.async_close()
            else:
                await session.async_abort()
        finally:
            pool.put(held)
        with pytest.raises(RequestError) as error:
            await asyncio.wait_for(task, 5)
        assert error.value.code is ErrorCode.REQUEST_CANCELLED
        handle.async_stream_infer.assert_not_called()
        handle.async_cancel.assert_not_awaited()
        assert pool.num_dispatched == 0
        expected = {session.session_id: session} if api == 'logits_with_sessions' and operation == 'abort' else {}
        assert manager.sessions == expected

    asyncio.run(run())


def test_logits_failure_cancels_sibling_before_removing_sessions():
    async def run():
        started, stopped = asyncio.Event(), asyncio.Event()

        async def pending_infer(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
                yield EngineOutput(ResponseType.FINISH, [])
            finally:
                stopped.set()

        async def failing_infer(*args, **kwargs):
            await started.wait()
            raise ValueError('scoring failed')
            yield  # This is an async generator, like a backend inference stream.

        handles = [SimpleNamespace(async_cancel=AsyncMock(), async_stream_infer=infer)
                   for infer in (pending_infer, failing_infer)]
        engine = _engine(handles)
        with pytest.raises(RequestError) as error:
            await asyncio.wait_for(engine.async_get_logits([[1, 2], [3, 4]]), 5)
        assert error.value.code is ErrorCode.INTERNAL_ERROR
        assert stopped.is_set()
        assert engine.session_mgr.request_handle_pool.num_dispatched == 0
        assert engine.session_mgr.sessions == {}
        for handle in handles:
            handle.async_cancel.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize('cancel_count', [1, 2, 3])
def test_cancelling_logits_batch_waits_for_cleanup(cancel_count):
    async def run():
        started, cancelling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        finished = asyncio.Event()

        async def infer(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()
            yield EngineOutput(ResponseType.FINISH, [])

        async def cancel(session_id):
            cancelling.set()
            await release.wait()
            finished.set()

        handle = SimpleNamespace(async_cancel=AsyncMock(side_effect=cancel), async_stream_infer=infer)
        engine = _engine([handle])
        task = asyncio.create_task(engine.async_get_logits([[1, 2], [3, 4]]))
        try:
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            await asyncio.wait_for(cancelling.wait(), 5)
            for _ in range(cancel_count - 1):
                task.cancel()
                await asyncio.sleep(0)
            await asyncio.wait({task}, timeout=0.01)
            assert not task.done()
            assert not finished.is_set()
            assert engine.session_mgr.request_handle_pool.num_dispatched == 1
            assert len(engine.session_mgr.sessions) == 2
        finally:
            release.set()
            result, = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
            await asyncio.wait_for(finished.wait(), 5)
        assert isinstance(result, asyncio.CancelledError)
        handle.async_cancel.assert_awaited_once()
        assert engine.session_mgr.request_handle_pool.num_dispatched == 0
        assert engine.session_mgr.sessions == {}

    asyncio.run(run())


@pytest.mark.parametrize('create_sessions', [False, True])
@pytest.mark.parametrize('cancel_batch', [False, True])
def test_logits_failure_waits_for_sibling_already_cancelling(create_sessions, cancel_batch):
    async def run():
        started = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]
        finished = [asyncio.Event(), asyncio.Event()]

        async def infer(*args, **kwargs):
            raise ValueError('scoring failed')
            yield

        def make_handle(index):
            async def cancel(session_id):
                started[index].set()
                await release[index].wait()
                finished[index].set()

            return SimpleNamespace(async_stream_infer=infer, async_cancel=AsyncMock(side_effect=cancel))

        handles = [make_handle(0), make_handle(1)]
        engine = _engine(handles)
        manager = engine.session_mgr
        sessions = None if create_sessions else [manager.get(), manager.get()]
        task = asyncio.create_task(engine.async_get_logits([[1, 2], [3, 4]], sessions=sessions))
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 5)
            sibling = manager.sessions[1]
            # Both children are already cleaning up. Finishing the first child
            # makes the batch cancel the sibling while it awaits async_cancel.
            release[0].set()
            await asyncio.wait_for(finished[0].wait(), 5)
            await asyncio.wait({task}, timeout=0.01)
            assert not task.done()
            assert not finished[1].is_set()
            assert sibling._handle is handles[1]
            assert manager.request_handle_pool.num_dispatched == 1
            assert manager.sessions.get(sibling.session_id) is sibling
            if cancel_batch:
                # Cancel the batch after it has started draining its children.
                task.cancel()
                await asyncio.wait({task}, timeout=0.01)
                assert not task.done()
                assert sibling._handle is handles[1]
                assert manager.sessions.get(sibling.session_id) is sibling
            with pytest.raises(RequestError) as error:
                async with sibling.request_handle():
                    pytest.fail('session acquired a handle before its previous cancellation finished')
            assert error.value.code is ErrorCode.REQUEST_CONFLICT
        finally:
            for event in release:
                event.set()
            result, = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in finished)), 5)

        if cancel_batch:
            assert isinstance(result, asyncio.CancelledError)
        else:
            assert isinstance(result, RequestError)
            assert result.code is ErrorCode.INTERNAL_ERROR
        assert manager.request_handle_pool.num_dispatched == 0
        assert manager.sessions == ({} if create_sessions else {s.session_id: s for s in sessions})
        # The same session becomes reusable once its previous cancellation ends.
        async with sibling.request_handle():
            pass
        for handle in handles:
            handle.async_cancel.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize('cancel_error', [RuntimeError('cancellation failed'), asyncio.CancelledError()])
def test_scoring_preserves_original_error_when_backend_cancellation_fails(cancel_error):
    async def run():
        original_error = ValueError('scoring failed')

        async def infer(*args, **kwargs):
            raise original_error
            yield

        handle = SimpleNamespace(async_stream_infer=infer, async_cancel=AsyncMock(side_effect=cancel_error))
        engine = _engine([handle])
        with pytest.raises(RequestError) as error:
            await asyncio.wait_for(engine.async_get_logits([[1, 2]]), 5)
        assert error.value.code is ErrorCode.INTERNAL_ERROR
        assert error.value.__cause__ is original_error
        handle.async_cancel.assert_awaited_once()
        assert engine.session_mgr.request_handle_pool.num_dispatched == 0
        assert engine.session_mgr.sessions == {}

    asyncio.run(run())


@pytest.mark.parametrize('api', ['logits', 'ppl'])
def test_scoring_returns_results_and_cleans_up_sessions(api):
    async def run():
        logits = torch.tensor([[1.0], [2.0], [3.0]])

        async def infer(*args, **kwargs):
            yield EngineOutput(ResponseType.FINISH, [], logits=logits, ce_loss=3.0)

        handle = SimpleNamespace(async_cancel=AsyncMock(), async_stream_infer=infer)
        engine = _engine([handle])
        if api == 'logits':
            results = await engine.async_get_logits([[1, 2], [1, 2, 3]])
            assert len(results) == 2
            assert torch.equal(results[0], logits[:2])
            assert torch.equal(results[1], logits)
        else:
            assert await engine.async_get_ppl([1, 2, 3]) == 1.5
        assert engine.session_mgr.request_handle_pool.num_dispatched == 0
        assert engine.session_mgr.sessions == {}
        handle.async_cancel.assert_not_awaited()

    asyncio.run(run())
