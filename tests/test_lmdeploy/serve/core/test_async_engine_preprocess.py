import asyncio
from types import SimpleNamespace

import pytest

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.core.async_engine import AsyncEngine, PreprocessedRequest
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.managers import SessionManager


class _RequestLogger:

    def log_prompt(self, *args, **kwargs):
        pass

    def log_inputs(self, *args, **kwargs):
        pass


def _engine(prompt_processor=None, *, session_len=32):
    engine = AsyncEngine.__new__(AsyncEngine)
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


@pytest.mark.parametrize('schema', [
    {'type': 'not-a-json-schema-type'},
    {'type': 'object', 'properties': {'value': {'type': 'string', 'pattern': '(?=a)a'}}, 'required': ['value']},
], ids=['invalid_schema', 'unsupported_grammar'])
def test_preprocess_rejects_uncompilable_response_format(schema):
    import xgrammar as xgr

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
