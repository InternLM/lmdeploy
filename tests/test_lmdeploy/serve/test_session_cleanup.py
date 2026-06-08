import asyncio
from contextlib import suppress

from lmdeploy.serve.core.exceptions import SafeRunException
from lmdeploy.serve.managers import SessionManager


class _FakeHandle:

    async def async_cancel(self, session_id: int):
        return None

    async def async_end(self, session_id: int):
        return None


class _FakeEngine:

    def create_instance(self):
        return _FakeHandle()


async def _run_request_handle_cleanup(raise_safe_run: bool = False):
    session_mgr = SessionManager()
    session_mgr.build_request_handle_pool(_FakeEngine(), 1)
    session_id = session_mgr.map_user_session_id(260606)
    session = session_mgr.get(session_id)
    session._remove_on_request_exit = True

    async with session.request_handle():
        if raise_safe_run:
            raise SafeRunException('cancelled')

    assert session_mgr.sessions == {}
    assert session_mgr.user_session_id_map == {}
    assert session_mgr.session_id_map == {}


def test_terminal_request_handle_exit_removes_session_maps():
    asyncio.run(_run_request_handle_cleanup())


def test_cancelled_request_handle_exit_removes_session_maps():
    asyncio.run(_run_request_handle_cleanup(raise_safe_run=True))


def test_stale_session_cleanup_does_not_remove_reused_session_id():
    session_mgr = SessionManager()
    session_id = 260606
    old_session = session_mgr.get(session_id)
    session_mgr.sessions.pop(session_id)
    new_session = session_mgr.get(session_id)
    session_mgr.session_id_map[session_id] = 42
    session_mgr.user_session_id_map[42] = session_id

    # Simulate a fast session-id reuse while an old request cleanup is still
    # unwinding with a stale Session object.
    session_mgr.remove(old_session)

    assert session_mgr.sessions[session_id] is new_session
    assert session_mgr.session_id_map[session_id] == 42
    assert session_mgr.user_session_id_map[42] == session_id


async def _run_api_wrapper_cleanup_after_cancel():
    from lmdeploy.serve.openai.api_server import VariableInterface, _with_request_cleanup

    session_mgr = SessionManager()
    session = session_mgr.get(260606)
    closed = asyncio.Event()
    never = asyncio.Event()

    class _FakeAsyncEngine:
        pass

    async def result_generator():
        try:
            yield 'engine'
            await never.wait()
        finally:
            closed.set()

    async def response_generator():
        async for item in result:
            yield item

    result = result_generator()

    origin_async_engine = VariableInterface.async_engine
    VariableInterface.async_engine = _FakeAsyncEngine()
    VariableInterface.async_engine.session_mgr = session_mgr
    try:
        wrapped = _with_request_cleanup(response_generator(), [result], [session])
        assert await wrapped.__anext__() == 'engine'

        next_task = asyncio.create_task(wrapped.__anext__())
        await asyncio.sleep(0)
        next_task.cancel()
        with suppress(asyncio.CancelledError):
            await next_task

        await asyncio.wait_for(closed.wait(), timeout=1)
        assert session_mgr.sessions == {}
    finally:
        VariableInterface.async_engine = origin_async_engine


def test_api_wrapper_cleanup_runs_after_cancel():
    asyncio.run(_run_api_wrapper_cleanup_after_cancel())


async def _run_prompt_cancel_updates_metrics():
    from lmdeploy.metrics.metrics_processor import metrics_processor
    from lmdeploy.metrics.stats import SchedulerStats
    from lmdeploy.serve.core.async_engine import AsyncEngine

    class _FakePromptProcessor:

        async def get_prompt_input(self, **kwargs):
            raise asyncio.CancelledError

    class _FakeRequestLogger:

        def log_prompt(self, *args, **kwargs):
            pass

    old_stats = metrics_processor.scheduler_stats
    metrics_processor.scheduler_stats = SchedulerStats()
    try:
        engine = AsyncEngine.__new__(AsyncEngine)
        engine.session_mgr = SessionManager()
        engine.prompt_processor = _FakePromptProcessor()
        engine.request_logger = _FakeRequestLogger()

        generator = engine.generate('hello', 260606)
        with suppress(asyncio.CancelledError):
            await generator.__anext__()

        stats = metrics_processor.scheduler_stats
        assert stats.num_total_reqs == 1
        assert stats.num_cancelled_reqs == 1
        assert stats.num_uncompleted_reqs == 0
        assert engine.session_mgr.sessions == {}
    finally:
        metrics_processor.scheduler_stats = old_stats


def test_prompt_cancel_updates_metrics():
    asyncio.run(_run_prompt_cancel_updates_metrics())
