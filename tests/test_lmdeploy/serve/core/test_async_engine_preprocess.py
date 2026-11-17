import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.core.async_engine import AsyncEngine, PreprocessedRequest
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.managers import SessionManager
from lmdeploy.vl.media.base import MediaInputError


class _RequestLogger:

    def log_prompt(self, *args, **kwargs):
        pass

    def log_inputs(self, *args, **kwargs):
        pass


def _engine(prompt_processor=None, *, session_len=32):
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.arch = 'Qwen3ForCausalLM'
    engine.session_mgr = SessionManager()
    engine.session_len = session_len
    engine.backend_config = SimpleNamespace(enable_prefix_caching=False)
    engine.request_logger = _RequestLogger()
    engine.prompt_processor = prompt_processor
    engine._determine_gen_config = lambda input_ids, gen_config=None: gen_config or GenerationConfig(max_new_tokens=8)
    return engine


def test_preprocess_returns_single_use_prepared_request():

    class _PromptProcessor:

        async def get_prompt_input(self, **kwargs):
            return {'prompt': 'rendered', 'input_ids': [1, 2, 3]}

    async def _run():
        engine = _engine(_PromptProcessor())
        request = await engine.preprocess('hello', 7)

        assert isinstance(request, PreprocessedRequest)
        assert request.session.session_id == 7
        assert request.inputs == {'prompt': 'rendered', 'input_ids': [1, 2, 3]}
        assert request.input_token_len == 3
        assert request.consumed is False

    asyncio.run(_run())


@pytest.mark.parametrize('active_handle', [False, True])
def test_abort_does_not_cancel_later_requests_on_same_session(active_handle, monkeypatch):
    from lmdeploy.messages import EngineOutput, ResponseType
    from lmdeploy.metrics.metrics_processor import metrics_processor
    from lmdeploy.metrics.stats import SchedulerStats

    monkeypatch.setattr(metrics_processor, 'scheduler_stats', SchedulerStats())
    monkeypatch.setattr(metrics_processor, 'queue_update', Mock())

    async def _run():
        async def infer(*args, **kwargs):
            yield EngineOutput(ResponseType.FINISH, [])

        handle = SimpleNamespace(async_stream_infer=Mock(side_effect=infer), async_cancel=AsyncMock())
        engine = _engine()
        engine.epoch = 0
        engine.num_spec_token = 0
        engine.request_logger.log_response = Mock()
        engine.session_mgr.build_request_handle_pool(SimpleNamespace(create_instance=lambda: handle), 1)
        session = engine.session_mgr.get(7)
        old_request = await engine.preprocess(None, session, input_ids=[1])

        if active_handle:
            async with session.request_handle():
                await session.async_abort()
            handle.async_cancel.assert_awaited_once_with(session.session_id)
        else:
            await session.async_abort()
            handle.async_cancel.assert_not_awaited()

        old_outputs = [out async for out in engine.generate(old_request)]
        assert old_outputs[-1].finish_reason == 'abort'
        handle.async_stream_infer.assert_not_called()

        new_request = await engine.preprocess(None, session, input_ids=[1])
        new_outputs = [out async for out in engine.generate(new_request)]
        assert new_outputs[-1].finish_reason == 'length'
        handle.async_stream_infer.assert_called_once()
        assert engine.session_mgr.request_handle_pool.num_dispatched == 0
        assert metrics_processor.scheduler_stats.num_aborted_reqs == 1
        assert metrics_processor.scheduler_stats.num_succeeded_reqs == 1

    asyncio.run(_run())


@pytest.mark.parametrize('stage', ['preprocess', 'prepared', 'waiting_handle'])
@pytest.mark.parametrize('sync_close', [False, True])
def test_close_invalidates_pending_request(stage, sync_close, monkeypatch):
    from lmdeploy.metrics.metrics_processor import metrics_processor
    from lmdeploy.metrics.stats import SchedulerStats
    from lmdeploy.model import BaseChatTemplate
    from lmdeploy.serve.processors import MultimodalProcessor

    monkeypatch.setattr(metrics_processor, 'scheduler_stats', SchedulerStats())

    async def run():
        loop = asyncio.get_running_loop()
        started, release = asyncio.Event(), threading.Event()

        def encode(prompt, **kwargs):
            if stage == 'preprocess':
                loop.call_soon_threadsafe(started.set)
                assert release.wait(5)
            return [1, 2, 3]

        processor = MultimodalProcessor(SimpleNamespace(encode=encode), BaseChatTemplate())
        engine = _engine(processor)
        engine.epoch = 0
        manager = engine.session_mgr
        manager.attach_event_loop(loop)
        handle = SimpleNamespace(async_cancel=AsyncMock(),
                                 async_stream_infer=Mock(side_effect=AssertionError('closed request dispatched')))
        manager.build_request_handle_pool(SimpleNamespace(create_instance=lambda: handle), 1)
        pool = manager.request_handle_pool
        session = manager.get(42)
        session.epoch = engine.epoch
        prepared = None

        async def collect():
            request = prepared or await engine.preprocess('hello', session, do_preprocess=False)
            return [out async for out in engine.generate(request)]

        if stage != 'preprocess':
            prepared = await engine.preprocess('hello', session, do_preprocess=False)
        if stage == 'waiting_handle':
            held = await pool.get()
            original_get = pool.get

            async def get():
                started.set()
                return await original_get()

            monkeypatch.setattr(pool, 'get', get)
        task = None if stage == 'prepared' else asyncio.create_task(collect())
        try:
            if task is not None:
                await asyncio.wait_for(started.wait(), 5)
            assert session._handle is None
            if sync_close:
                await asyncio.wait_for(asyncio.to_thread(session.close), 5)
            else:
                await session.async_close()
            assert manager.sessions == {}
            # Old request cleanup must leave a reused session ID intact.
            replacement = manager.get(42)
        finally:
            release.set()
            if stage == 'waiting_handle':
                pool.put(held)
            processor.close()
        outputs = await asyncio.wait_for(task if task is not None else collect(), 5)
        assert [out.finish_reason for out in outputs] == ['abort']
        handle.async_stream_infer.assert_not_called()
        handle.async_cancel.assert_not_awaited()
        assert pool.num_dispatched == 0
        assert manager.sessions == {42: replacement}
        assert metrics_processor.scheduler_stats.num_aborted_reqs == 1
        assert metrics_processor.scheduler_stats.num_uncompleted_reqs == 0

    asyncio.run(run())


def test_preprocess_reports_context_error_and_removes_session():

    async def _run():
        engine = _engine(session_len=3)
        with pytest.raises(RequestError) as exc_info:
            await engine.preprocess(None, 8, input_ids=[1, 2, 3])

        assert exc_info.value.code is ErrorCode.CONTEXT_LENGTH_EXCEEDED
        assert engine.session_mgr.sessions == {}

    asyncio.run(_run())


@pytest.mark.parametrize('max_new_tokens', [0, -1])
def test_preprocess_rejects_non_positive_max_new_tokens(max_new_tokens):

    async def _run():
        engine = _engine()
        with pytest.raises(RequestError) as exc_info:
            await engine.preprocess(None,
                                    9,
                                    input_ids=[1],
                                    gen_config=GenerationConfig(max_new_tokens=max_new_tokens))

        assert exc_info.value.code is ErrorCode.INVALID_REQUEST
        assert exc_info.value.message == f'max_new_tokens must be at least 1, got {max_new_tokens}.'
        assert engine.session_mgr.sessions == {}

    asyncio.run(_run())


def test_preprocess_hides_unexpected_exception_details():

    class _PromptProcessor:

        async def get_prompt_input(self, **kwargs):
            raise RuntimeError('private tokenizer detail')

    async def _run():
        engine = _engine(_PromptProcessor())
        with pytest.raises(RequestError) as exc_info:
            await engine.preprocess('hello', 9)

        assert exc_info.value.code is ErrorCode.PREPROCESS_FAILED
        assert exc_info.value.message == 'Request preprocessing failed.'
        assert 'private tokenizer detail' not in exc_info.value.message
        assert engine.session_mgr.sessions == {}

    asyncio.run(_run())


def test_preprocess_reports_media_input_error_detail():

    class _PromptProcessor:

        async def get_prompt_input(self, **kwargs):
            raise MediaInputError('Failed to read media file /data/missing.png: No such file or directory')

    async def _run():
        engine = _engine(_PromptProcessor())
        with pytest.raises(RequestError) as exc_info:
            await engine.preprocess('hello', 9)

        assert exc_info.value.code is ErrorCode.PREPROCESS_FAILED
        assert 'No such file or directory' in exc_info.value.message
        assert '/data/missing.png' in exc_info.value.message
        assert engine.session_mgr.sessions == {}

    asyncio.run(_run())


def test_preprocess_hides_internal_file_errors():
    # Same exception type as a user-path failure, but raised against a
    # server-internal temp file: decoding an HTTP/base64 video stages the bytes
    # in a NamedTemporaryFile, and a vanished temp dir makes it raise
    # FileNotFoundError naming that internal path. The type alone must not make
    # the detail client-visible.
    internal_path = '/tmp/server-internal-video-cache/tmp9f3k.mp4'

    class _PromptProcessor:

        async def get_prompt_input(self, **kwargs):
            raise FileNotFoundError(2, 'No such file or directory', internal_path)

    async def _run():
        engine = _engine(_PromptProcessor())
        with pytest.raises(RequestError) as exc_info:
            await engine.preprocess('hello', 9)

        assert exc_info.value.code is ErrorCode.PREPROCESS_FAILED
        assert exc_info.value.message == 'Request preprocessing failed.'
        assert internal_path not in exc_info.value.message
        assert engine.session_mgr.sessions == {}

    asyncio.run(_run())


def test_preprocess_rejects_active_session_without_removing_it():

    async def _run():
        engine = _engine()
        session = engine.session_mgr.get(11)
        session._handle = object()

        with pytest.raises(RequestError) as exc_info:
            await engine.preprocess(None, session, input_ids=[1])

        assert exc_info.value.code is ErrorCode.REQUEST_CONFLICT
        assert engine.session_mgr.sessions[11] is session

    asyncio.run(_run())


def test_preprocess_rejects_uncompilable_response_format():
    import xgrammar as xgr

    schema = {'type': 'not-a-json-schema-type'}
    response_format = xgr.get_model_structural_tag(
        'qwen_3',
        [{
            'type': 'function',
            'function': {
                'name': 'search',
                'parameters': schema,
            },
        }],
        tool_choice='required',
        reasoning=False,
    ).model_dump(mode='json')

    async def _run():
        engine = _engine()
        gen_config = GenerationConfig(max_new_tokens=8, response_format=response_format)
        with pytest.raises(RequestError) as exc_info:
            await engine.preprocess(None, 9, input_ids=[1], gen_config=gen_config)

        assert exc_info.value.code is ErrorCode.INVALID_REQUEST
        assert engine.session_mgr.sessions == {}

    asyncio.run(_run())


def test_generate_rejects_raw_and_consumed_requests():

    async def _run():
        engine = _engine()
        with pytest.raises(TypeError, match='PreprocessedRequest'):
            await anext(engine.generate('raw input'))

        request = PreprocessedRequest(
            session=engine.session_mgr.get(10),
            inputs={'input_ids': [1]},
            input_token_len=1,
            gen_config=GenerationConfig(max_new_tokens=1),
            adapter_name=None,
            consumed=True,
        )
        with pytest.raises(RequestError) as exc_info:
            await anext(engine.generate(request))
        assert exc_info.value.code is ErrorCode.REQUEST_CONFLICT

    asyncio.run(_run())


@pytest.mark.parametrize('stage', ['preprocess', 'prepared', 'waiting_handle'])
@pytest.mark.parametrize('abort_all', [False, True])
def test_abort_endpoint_prevents_dispatch(stage, abort_all, monkeypatch):
    from fastapi import APIRouter

    from lmdeploy.metrics.metrics_processor import metrics_processor
    from lmdeploy.metrics.stats import SchedulerStats
    from lmdeploy.serve.openai.endpoints.management import register
    from lmdeploy.serve.openai.protocol import AbortRequest

    monkeypatch.setattr(metrics_processor, 'scheduler_stats', SchedulerStats())

    async def _run():
        started, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class _PromptProcessor:

            async def get_prompt_input(self, **kwargs):
                started.set()
                if stage == 'preprocess':
                    await release.wait()
                return {'input_ids': [1, 2, 3]}

        engine = _engine(_PromptProcessor())
        engine.epoch = 0
        manager = engine.session_mgr
        handle = SimpleNamespace(async_cancel=AsyncMock(),
                                 async_stream_infer=Mock(side_effect=AssertionError('aborted request dispatched')))
        manager.build_request_handle_pool(SimpleNamespace(create_instance=lambda: handle), 1)
        pool = manager.request_handle_pool
        session = manager.get(manager.map_user_session_id(42))
        session.epoch = engine.epoch
        context = SimpleNamespace(
            enable_abort_handling=True, async_engine=engine, session_manager=manager,
            find_session=lambda public_id: manager.get(manager.user_session_id_map.get(public_id),
                                                       create_if_not_exists=False))
        router = APIRouter()
        register(router, context)
        abort = next(route.endpoint for route in router.routes if route.path == '/abort_request')

        async def collect():
            request = await engine.preprocess('hello', session)
            if stage == 'prepared':
                await release.wait()
            return [out async for out in engine.generate(request, stream_response=False)]

        if stage == 'waiting_handle':
            held = await pool.get()
            original_get = pool.get

            async def get():
                waiting.set()
                return await original_get()

            monkeypatch.setattr(pool, 'get', get)
        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(waiting.wait() if stage == 'waiting_handle' else started.wait(), 5)
            response = await abort(AbortRequest(session_id=42, abort_all=abort_all))
            assert response.status_code == 200
            assert manager.sessions == {}
            assert manager.user_session_id_map == {}
            assert manager.session_id_map == {}
            # Cleanup of the aborted request must not remove a new request
            # that reuses the same public session ID.
            replacement = manager.get(manager.map_user_session_id(42))
        finally:
            release.set()
            if stage == 'waiting_handle':
                pool.put(held)
        outputs = await asyncio.wait_for(task, 5)
        assert len(outputs) == 1
        assert outputs[0].finish_reason == 'abort'
        assert outputs[0].generate_token_len == 0
        handle.async_stream_infer.assert_not_called()
        handle.async_cancel.assert_not_awaited()
        assert pool.num_dispatched == 0
        assert manager.sessions == {replacement.session_id: replacement}
        assert manager.user_session_id_map == {42: replacement.session_id}
        assert metrics_processor.scheduler_stats.num_aborted_reqs == 1
        assert metrics_processor.scheduler_stats.num_uncompleted_reqs == 0

    asyncio.run(_run())


@pytest.mark.parametrize('abort_all', [False, True])
def test_request_arriving_during_abort_all_remains_cancellable(abort_all, monkeypatch):
    from fastapi import APIRouter

    from lmdeploy.messages import EngineOutput, ResponseType
    from lmdeploy.metrics.metrics_processor import metrics_processor
    from lmdeploy.metrics.stats import SchedulerStats
    from lmdeploy.model import BaseChatTemplate
    from lmdeploy.serve.openai.api_server import ServerContext
    from lmdeploy.serve.openai.endpoints.management import register
    from lmdeploy.serve.openai.protocol import AbortRequest
    from lmdeploy.serve.processors import MultimodalProcessor

    monkeypatch.setattr(metrics_processor, 'scheduler_stats', SchedulerStats())
    monkeypatch.setattr(metrics_processor, 'queue_update', Mock())

    async def run():
        loop = asyncio.get_running_loop()
        abort_started, release_abort = asyncio.Event(), asyncio.Event()
        prompt_started, release_prompt = asyncio.Event(), threading.Event()
        infer_started, cancelled = asyncio.Event(), asyncio.Event()

        def encode(prompt, **kwargs):
            loop.call_soon_threadsafe(prompt_started.set)
            assert release_prompt.wait(5)
            return [1, 2, 3]

        async def cancel(session_id):
            if session_id == old.session_id:
                abort_started.set()
                await release_abort.wait()
            else:
                cancelled.set()

        async def infer(*args, **kwargs):
            infer_started.set()
            await cancelled.wait()
            yield EngineOutput(ResponseType.CANCEL, [])

        processor = MultimodalProcessor(SimpleNamespace(encode=encode), BaseChatTemplate())
        engine = _engine(processor)
        engine.epoch = 0
        engine.num_spec_token = 0
        engine.request_logger.log_response = Mock()
        handle = SimpleNamespace(async_cancel=AsyncMock(side_effect=cancel),
                                 async_stream_infer=Mock(side_effect=infer))
        manager = engine.session_mgr
        manager.build_request_handle_pool(SimpleNamespace(create_instance=lambda: handle), 1)
        context = ServerContext()
        context.async_engine = engine
        context.enable_abort_handling = True
        old = context.create_session(41)
        router = APIRouter()
        register(router, context)
        abort = next(route.endpoint for route in router.routes if route.path == '/abort_request')

        async def collect(session):
            request = await engine.preprocess('hello', session, do_preprocess=False)
            return [out async for out in engine.generate(request)]

        tasks = []
        try:
            async with old.request_handle():
                abort_task = asyncio.create_task(abort(AbortRequest(abort_all=True)))
                tasks.append(abort_task)
                await asyncio.wait_for(abort_started.wait(), 5)
                newcomer = context.create_session(42)
                request_task = asyncio.create_task(collect(newcomer))
                tasks.append(request_task)
                await asyncio.wait_for(prompt_started.wait(), 5)
            release_abort.set()
            assert (await asyncio.wait_for(abort_task, 5)).status_code == 200
            assert context.find_session(41) is None
            assert context.find_session(42) is newcomer
            assert newcomer.epoch == engine.epoch
            assert newcomer.abort_version == 0

            release_prompt.set()
            await asyncio.wait_for(infer_started.wait(), 5)
            response = await abort(AbortRequest(session_id=42, abort_all=abort_all))
            assert response.status_code == 200
            outputs = await asyncio.wait_for(request_task, 5)
            assert [out.finish_reason for out in outputs] == ['abort']
            assert handle.async_stream_infer.call_count == 1
            assert [call.args[0] for call in handle.async_cancel.await_args_list] == [
                old.session_id, newcomer.session_id]
            assert manager.sessions == {}
            assert manager.user_session_id_map == {}
            assert manager.session_id_map == {}
            assert manager.request_handle_pool.num_dispatched == 0
            assert metrics_processor.scheduler_stats.num_uncompleted_reqs == 0
        finally:
            release_abort.set()
            release_prompt.set()
            cancelled.set()
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
            processor.close()

    asyncio.run(run())
