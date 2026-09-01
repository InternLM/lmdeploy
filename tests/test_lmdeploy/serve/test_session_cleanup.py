import asyncio
from contextlib import aclosing, suppress
from types import SimpleNamespace

import pytest

from lmdeploy.messages import (
    EngineOutput,
    GenerationConfig,
    RequestMetrics,
    ResponseType,
)
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError, SafeRunException
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

        with suppress(asyncio.CancelledError):
            await engine.preprocess('hello', 260606)

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
        engine.session_len = 4096
        engine._determine_gen_config = lambda input_ids, gen_config=None: gen_config
        engine.backend_config = type('_BackendConfig', (), {'enable_prefix_caching': False})()
        engine.request_logger = type('_RequestLogger', (), {'log_inputs': lambda *args, **kwargs: None})()

        session = engine.session_mgr.get(260606)
        try:
            await engine.preprocess(None,
                                    session,
                                    input_ids=[1, 2],
                                    gen_config=GenerationConfig(max_new_tokens=0))
        except RequestError as error:
            assert error.code is ErrorCode.INVALID_REQUEST
            assert error.message == 'max_new_tokens must be at least 1, got 0.'
        else:
            raise AssertionError('Expected zero max_new_tokens to fail preprocessing.')
        assert not hasattr(session, 'step')
        assert engine.session_mgr.sessions == {}
    finally:
        metrics_processor.scheduler_stats = old_stats


def test_invalid_max_new_tokens_cleans_up_session():
    asyncio.run(_run_max_new_tokens_zero_cleans_up_session())


async def _run_input_logprob_outputs(engine_outputs,
                                     logprob_start_len=0,
                                     queued_metrics=None):
    from lmdeploy.metrics.metrics_processor import metrics_processor
    from lmdeploy.serve.core.async_engine import AsyncEngine

    queued_metrics = [] if queued_metrics is None else queued_metrics

    class _Handle:

        def async_stream_infer(self, *args, **kwargs):

            async def outputs():
                for output in engine_outputs:
                    yield output

            return outputs()

        async def async_cancel(self, session_id):
            raise AssertionError('input-logprob terminal extraction should not cancel')

        async def async_end(self, session_id):
            raise AssertionError('input-logprob terminal extraction should not end early')

    handle = _Handle()
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.session_mgr = SessionManager()
    engine.session_mgr.build_request_handle_pool(
        SimpleNamespace(create_instance=lambda: handle), 1)
    engine.session_len = 4096
    engine._determine_gen_config = lambda input_ids, gen_config=None: gen_config
    engine._if_session_stale = lambda session, input_len: None
    engine.backend = 'pytorch'
    engine.backend_config = SimpleNamespace(
        role=EngineRole.Hybrid,
        logprobs_mode='raw_logprobs',
        enable_prefix_caching=True,
    )
    engine.hf_cfg = SimpleNamespace(vocab_size=10)
    engine.num_spec_token = 0
    engine.request_logger = SimpleNamespace(log_inputs=lambda *args, **kwargs: None,
                                            log_response=lambda *args, **kwargs: None)

    old_queue_update = metrics_processor.queue_update
    metrics_processor.queue_update = queued_metrics.append
    try:
        session = engine.session_mgr.get(260609)
        config = GenerationConfig(max_new_tokens=0,
                                  logprobs=0,
                                  logprob_start_len=logprob_start_len)
        prepared = await engine.preprocess(None,
                                           session,
                                           input_ids=[1, 2, 3],
                                           gen_config=config)
        results = [out async for out in engine.generate(prepared)]
        return results, queued_metrics, engine.session_mgr
    finally:
        metrics_processor.queue_update = old_queue_update


@pytest.mark.parametrize('carrier', [
    [],
    [{2: -0.5}, {3: -0.7}],
])
def test_async_engine_reinterprets_terminal_logprobs_as_input_rows(carrier):
    output = EngineOutput(ResponseType.FINISH,
                          [],
                          logprobs=carrier,
                          req_metrics=RequestMetrics())
    start = 2 if carrier == [] else 0
    results, queued_metrics, manager = asyncio.run(
        _run_input_logprob_outputs([output], logprob_start_len=start))

    assert len(queued_metrics) == 1
    assert manager.sessions == {}
    assert len(results) == 1
    assert results[0].token_ids == []
    assert results[0].generate_token_len == 0
    assert results[0].finish_reason == 'length'
    assert results[0].logprobs == carrier
    assert results[0].logprob_token_ids == ([] if carrier == [] else [2, 3])


def test_async_engine_validates_input_logprob_start_after_preprocess():
    with pytest.raises(RequestError) as exc_info:
        asyncio.run(_run_input_logprob_outputs([], logprob_start_len=3))

    assert exc_info.value.code is ErrorCode.INVALID_REQUEST
    assert 'processed input_ids length(3)' in exc_info.value.message
