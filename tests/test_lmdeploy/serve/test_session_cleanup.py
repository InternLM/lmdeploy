import asyncio
from contextlib import aclosing, suppress

from lmdeploy.messages import GenerationConfig
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


async def _run_idle_async_close_removes_session_and_allows_reuse():
    session_mgr = SessionManager()
    session_mgr.build_request_handle_pool(_FakeEngine(), 1)
    session_id = 260606
    session = session_mgr.get(session_id)

    await session.async_close()

    assert session_id not in session_mgr.sessions

    new_session = session_mgr.get(session_id)
    async with new_session.request_handle():
        pass


def test_idle_async_close_removes_session_and_allows_reuse():
    asyncio.run(_run_idle_async_close_removes_session_and_allows_reuse())


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


async def _run_request_cleanup_removes_unstarted_generator_session():
    from lmdeploy.serve.utils.request_cleanup import with_request_cleanup

    session_mgr = SessionManager()
    session = session_mgr.get(260606)

    async def result_generator():
        yield 'engine'

    async def response_generator():
        yield 'header'
        async for item in result:
            yield item

    result = result_generator()
    wrapped = with_request_cleanup(response_generator(), [result], [session], session_mgr)

    assert await wrapped.__anext__() == 'header'
    await wrapped.aclose()

    assert session_mgr.sessions == {}


def test_request_cleanup_removes_session_when_engine_generator_never_started():
    asyncio.run(_run_request_cleanup_removes_unstarted_generator_session())


async def _run_request_cleanup_runs_on_return_inside_loop():
    from lmdeploy.serve.utils.request_cleanup import with_request_cleanup

    session_mgr = SessionManager()
    session = session_mgr.get(260607)
    closed = asyncio.Event()

    async def result_generator():
        try:
            yield 'engine'
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def endpoint_like_return_inside_async_for():
        result = result_generator()
        async with aclosing(with_request_cleanup(result, [result], [session], session_mgr)) as generator:
            async for _ in generator:
                return 'Client disconnected'

    assert await endpoint_like_return_inside_async_for() == 'Client disconnected'

    assert session_mgr.sessions == {}
    assert closed.is_set()


def test_request_cleanup_runs_on_return_inside_loop():
    asyncio.run(_run_request_cleanup_runs_on_return_inside_loop())


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


async def _run_max_new_tokens_zero_cleans_up_session():
    from lmdeploy.metrics.metrics_processor import metrics_processor
    from lmdeploy.metrics.stats import SchedulerStats
    from lmdeploy.serve.core.async_engine import AsyncEngine

    old_stats = metrics_processor.scheduler_stats
    metrics_processor.scheduler_stats = SchedulerStats()
    try:
        engine = AsyncEngine.__new__(AsyncEngine)
        engine.session_mgr = SessionManager()
        engine.session_mgr.build_request_handle_pool(_FakeEngine(), 1)
        engine._determine_gen_config = lambda session, input_ids, gen_config=None: gen_config

        session = engine.session_mgr.get(260606)
        generator = engine.generate(None,
                                    session,
                                    input_ids=[1, 2],
                                    gen_config=GenerationConfig(max_new_tokens=0))

        out = await generator.__anext__()

        assert out.input_token_len == 2
        assert out.finish_reason == 'length'
        assert not hasattr(session, 'step')
        assert engine.session_mgr.sessions == {}
    finally:
        metrics_processor.scheduler_stats = old_stats


def test_max_new_tokens_zero_cleans_up_session():
    asyncio.run(_run_max_new_tokens_zero_cleans_up_session())


async def _run_completions_v1_disconnect_aborts_active_handle():
    from types import SimpleNamespace

    from lmdeploy.serve.core.async_engine import GenOut
    from lmdeploy.serve.openai.api_server import VariableInterface, completions_v1
    from lmdeploy.serve.openai.protocol import CompletionRequest

    class _SpyHandle(_FakeHandle):

        def __init__(self):
            self.cancelled_session_ids = []

        async def async_cancel(self, session_id: int):
            self.cancelled_session_ids.append(session_id)

    class _FakeAsyncEngine:
        model_name = 'test-model'
        epoch = 0

        def __init__(self):
            self.backend_config = SimpleNamespace()
            self.session_mgr = SessionManager()
            self.handle = _SpyHandle()

        async def generate(self, prompt, session, **kwargs):
            # Mirror the real AsyncEngine: a handle is held for the duration of generation.
            session._handle = self.handle
            yield GenOut(response='hi', input_token_len=1, generate_token_len=1, finish_reason=None)
            await asyncio.Event().wait()

    class _FakeRawRequest:

        async def json(self):
            return {}

        async def is_disconnected(self):
            return True

    engine = _FakeAsyncEngine()
    origin_async_engine = VariableInterface.async_engine
    VariableInterface.async_engine = engine
    try:
        request = CompletionRequest(model='test-model', prompt='hello', stream=False)
        await completions_v1(request, _FakeRawRequest())
    finally:
        VariableInterface.async_engine = origin_async_engine

    assert engine.handle.cancelled_session_ids == [0]


def test_completions_v1_disconnect_aborts_active_handle():
    asyncio.run(_run_completions_v1_disconnect_aborts_active_handle())


async def _run_completions_v1_disconnect_returns_client_disconnected_response():
    from types import SimpleNamespace

    from fastapi.responses import JSONResponse

    from lmdeploy.serve.core.async_engine import GenOut
    from lmdeploy.serve.openai.api_server import VariableInterface, completions_v1
    from lmdeploy.serve.openai.protocol import CompletionRequest

    class _FakeAsyncEngine:
        model_name = 'test-model'
        epoch = 0

        def __init__(self):
            self.backend_config = SimpleNamespace()
            self.session_mgr = SessionManager()

        async def generate(self, prompt, session, **kwargs):
            yield GenOut(response='hi', input_token_len=1, generate_token_len=1, finish_reason=None)
            await asyncio.Event().wait()

    class _FakeRawRequest:

        async def json(self):
            return {}

        async def is_disconnected(self):
            return True

    origin_async_engine = VariableInterface.async_engine
    VariableInterface.async_engine = _FakeAsyncEngine()
    try:
        request = CompletionRequest(model='test-model', prompt='hello', stream=False)
        result = await completions_v1(request, _FakeRawRequest())
    finally:
        VariableInterface.async_engine = origin_async_engine

    assert isinstance(result, JSONResponse)
    assert result.status_code == 400
    assert b'Client disconnected' in result.body


def test_completions_v1_disconnect_returns_client_disconnected_response():
    asyncio.run(_run_completions_v1_disconnect_returns_client_disconnected_response())
