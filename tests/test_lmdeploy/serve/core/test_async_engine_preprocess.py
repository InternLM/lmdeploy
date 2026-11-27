import asyncio
from types import SimpleNamespace

import pytest

from lmdeploy.messages import EngineOutput, GenerationConfig, ResponseType
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest
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


def test_pd_prompt_reuse_never_calls_media_processor():
    class _ForbiddenProcessor:

        async def get_prompt_input(self, **kwargs):
            pytest.fail('Decode must not render or preprocess the imported prompt')

    async def run():
        engine = _engine(_ForbiddenProcessor())
        migration = MigrationRequest(protocol=MigrationProtocol.RDMA, remote_engine_id='prefill',
                                     remote_session_id=1, prompt_text='rendered <image> prompt')
        request = await engine.preprocess(None, 7, input_ids=[1, 2, 2, 3], do_preprocess=False,
                                          gen_config=GenerationConfig(max_new_tokens=8, migration_request=migration))
        assert request.inputs == {'input_ids': [1, 2, 2, 3], 'prompt': 'rendered <image> prompt'}
        assert request.gen_config.migration_request.prompt_text is None

    asyncio.run(run())


@pytest.mark.parametrize('recompute', [False, True])
def test_pd_media_is_materialized_only_on_recompute_without_losing_buffered_output(recompute):
    async def run():
        calls, resumed = [], []
        original = [dict(role='user', content=[dict(type='image_url', image_url=dict(url='stable.png'))])]

        class Processor:
            async def get_prompt_input(self, **kwargs):
                calls.append(kwargs)
                return dict(input_ids=[1, 2, 2, 3], multimodal=['processed pixels'])

        async def resume(session_id, seq_id, inputs):
            resumed.append((session_id, seq_id, inputs))

        migration = MigrationRequest.model_validate(dict(
            protocol=2, remote_engine_id='prefill', remote_session_id=1,
            handoff=dict(transfer_id='transfer', engine_epoch='epoch', cache_schema='schema',
                         prompt_digest='digest', prompt_token_count=4, valid_token_count=4,
                         multimodal_spans=[(1, 3, 'image', 'hash')], first_token_id=9,
                         random_seed=42, random_offset=5, block_size=64, block_ids=[1])))
        engine = _engine(Processor())
        request = await engine.preprocess(original, 7, input_ids=[1, 2, 2, 3],
                                          mm_processor_kwargs={'image': {'max_pixels': 1024}},
                                          gen_config=GenerationConfig(max_new_tokens=8, migration_request=migration))
        assert calls == [] and request.inputs == {'input_ids': [1, 2, 2, 3]}

        async def outputs():
            yield EngineOutput(ResponseType.SUCCESS, [9])
            if recompute:
                yield EngineOutput(ResponseType.SUCCESS, [10], logprobs=[{10: -0.5}],
                                   input_multimodals_request=13)
            yield EngineOutput(ResponseType.FINISH, [11])

        stream = engine._stream_with_multimodal_recompute(
            outputs(), SimpleNamespace(async_resume_multimodal=resume), request.session, request.recompute_prompt)
        assert (await anext(stream)).token_ids == [9]
        assert not calls
        if recompute:
            buffered = await anext(stream)
            assert buffered.token_ids == [10] and buffered.logprobs == [{10: -0.5}]
            assert buffered.input_multimodals_request is None and not calls
        assert (await anext(stream)).token_ids == [11]
        await stream.aclose()
        assert len(calls) == int(recompute)
        if recompute:
            assert calls[0]['prompt'] == original
            assert calls[0]['mm_processor_kwargs'] == {'image': {'max_pixels': 1024}}
            assert resumed == [(7, 13, dict(input_ids=[1, 2, 2, 3], multimodal=['processed pixels']))]

    asyncio.run(run())


def test_cancellation_during_deferred_media_does_not_resume_request():
    async def run():
        started = asyncio.Event()

        class Processor:
            async def get_prompt_input(self, **kwargs):
                started.set()
                await asyncio.Future()

        async def resume(*args):
            pytest.fail('Cancelled preprocessing must not resume the request')

        async def outputs():
            yield EngineOutput(ResponseType.SUCCESS, [], input_multimodals_request=13)

        engine = _engine(Processor())
        stream = engine._stream_with_multimodal_recompute(
            outputs(), SimpleNamespace(async_resume_multimodal=resume), SimpleNamespace(session_id=7), dict(prompt=[]))
        task = asyncio.create_task(anext(stream))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await stream.aclose()

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


@pytest.mark.parametrize('start', [0, 2, 3])
@pytest.mark.parametrize('tokenize', [False, True])
def test_routed_experts_start_uses_processed_prompt_length(start, tokenize):
    class _PromptProcessor:

        async def get_prompt_input(self, **kwargs):
            return {'prompt': 'rendered', 'input_ids': [1, 2]}

    async def _run():
        engine = _engine(_PromptProcessor())
        config = GenerationConfig(return_routed_experts=True, routed_experts_start_len=start)
        kwargs = {'messages': 'render me'} if tokenize else {'messages': None, 'input_ids': [1, 2]}
        if start > 2:
            with pytest.raises(RequestError) as error:
                await engine.preprocess(session_id=17, gen_config=config, **kwargs)
            assert error.value.code is ErrorCode.INVALID_REQUEST
            assert engine.session_mgr.sessions == {}
        else:
            prepared = await engine.preprocess(session_id=17, gen_config=config, **kwargs)
            assert prepared.gen_config.routed_experts_start_len == start

    asyncio.run(_run())
