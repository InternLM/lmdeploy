# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lmdeploy.serve.core.chat_runner import ChatRunner, ChatRunnerOptions
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage


class _FakeSession:

    def __init__(self, session_id=0):
        self.session_id = session_id
        self.aborted = False

    async def async_abort(self):
        self.aborted = True


class _FakeSessionManager:

    def __init__(self):
        self.removed = []
        self.session = _FakeSession()

    def get(self, session_id=None):
        if session_id is not None:
            self.session.session_id = session_id
        return self.session

    def remove(self, session):
        self.removed.append(session)


class _FakeEngine:

    model_name = 'fake-model'
    backend_config = SimpleNamespace(adapters=[])

    def __init__(self, outputs=None):
        self.outputs = outputs or [
            SimpleNamespace(
                response='ok',
                token_ids=[1],
                input_token_len=3,
                generate_token_len=1,
                finish_reason='stop',
                cached_tokens=0,
                logprobs=[{
                    1: -0.1,
                }],
                routed_experts=[[1]],
                cache_block_ids=['cache'],
            )
        ]
        self.session_mgr = _FakeSessionManager()
        self.preprocess_messages = None
        self.preprocess_kwargs = None
        self.generator_closed = False

    async def preprocess(self, messages, session, **kwargs):
        self.preprocess_messages = messages
        self.preprocess_kwargs = kwargs
        return SimpleNamespace(messages=messages, session=session)

    def generate(self, request, **kwargs):
        async def _generator():
            try:
                for output in self.outputs:
                    yield output
            finally:
                self.generator_closed = True

        return _generator()


class _FakeServerContext:

    def __init__(self, parser_cls, outputs=None):
        self.async_engine = _FakeEngine(outputs)
        self.default_gen_config = {}
        self.response_parser_cls = parser_cls

    @property
    def session_manager(self):
        return self.async_engine.session_mgr

    def create_session(self, session_id=None):
        return self.session_manager.get(session_id)


class _Parser:
    supports_required_tool_choice = False
    tool_parser_cls = object()
    tool_parser = object()
    reasoning_tokens = 2

    def __init__(self, request):
        self.request = request

    def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
        return [(DeltaMessage(role='assistant', content=delta_text), False)]

    def parse_complete(self, text: str, token_ids: list[int] | None = None, **kwargs):
        return text, None, None

    def validate_complete(self, text: str | None = None):
        return True


def _request(**kwargs):
    defaults = {
        'model': 'fake-model',
        'messages': [{
            'role': 'user',
            'content': 'hi',
        }],
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _tools():
    return [{
        'type': 'function',
        'function': {
            'name': 'search',
            'parameters': {
                'type': 'object',
            },
        },
    }]


def test_runner_forwards_parser_adjusted_response_format_to_engine():
    response_format = {'type': 'json_object'}

    class _AdjustingParser(_Parser):

        def __init__(self, request):
            super().__init__(request)
            self.request = request.model_copy(update={'response_format': response_format})

    context = _FakeServerContext(_AdjustingParser)

    asyncio.run(ChatRunner.prepare(context, _request()))

    assert context.async_engine.preprocess_kwargs['gen_config'].response_format == response_format


def test_runner_normalizes_string_stop_word_for_engine():
    context = _FakeServerContext(_Parser)

    asyncio.run(ChatRunner.prepare(context, _request(stop='END')))

    assert context.async_engine.preprocess_kwargs['gen_config'].stop_words == ['END']


def test_runner_maps_return_logprob_to_engine_logprobs():
    context = _FakeServerContext(_Parser)

    asyncio.run(ChatRunner.prepare(context, _request(return_logprob=True)))

    assert context.async_engine.preprocess_kwargs['gen_config'].logprobs == 1


def test_runner_maps_chat_logprobs_to_engine_logprobs():
    context = _FakeServerContext(_Parser)

    asyncio.run(ChatRunner.prepare(context, _request(logprobs=True, top_logprobs=3)))

    assert context.async_engine.preprocess_kwargs['gen_config'].logprobs == 3


def test_runner_always_preprocesses_chat_messages():
    context = _FakeServerContext(_Parser)

    asyncio.run(ChatRunner.prepare(context, _request()))

    assert context.async_engine.preprocess_kwargs['do_preprocess'] is True


def test_runner_skips_preprocess_for_raw_input_ids():
    context = _FakeServerContext(_Parser)

    chat_runner = asyncio.run(
        ChatRunner.prepare(
            context,
            _request(messages=[]),
            ChatRunnerOptions(input_ids=[1, 2, 3], do_preprocess=False),
        ))

    assert chat_runner.request.messages == []
    assert context.async_engine.preprocess_messages is None
    assert context.async_engine.preprocess_kwargs['do_preprocess'] is False
    assert context.async_engine.preprocess_kwargs['input_ids'] == [1, 2, 3]


@pytest.mark.parametrize(
    ('request_kwargs', 'finish_reason', 'expected'),
    [
        ({'return_token_ids': True}, 'stop', 'parse_error'),
        ({'return_token_ids': True}, 'length', 'parse_error'),
        ({'return_routed_experts': True}, 'stop', 'parse_error'),
        ({'tool_choice': 'required', 'tools': _tools()}, 'stop', 'stop'),
        ({'tool_choice': 'required', 'tools': _tools()}, 'length', 'length'),
        ({'tool_choice': 'required', 'tools': _tools(), 'return_token_ids': True}, 'stop', 'parse_error'),
        ({'tool_choice': 'required', 'tools': _tools(), 'return_token_ids': True}, 'length', 'parse_error'),
    ],
)
def test_runner_terminal_validation(request_kwargs, finish_reason, expected):
    class _InvalidParser(_Parser):
        supports_required_tool_choice = True

        def validate_complete(self, text: str | None = None):
            return False

    outputs = [
        SimpleNamespace(
            response='plain',
            token_ids=[1],
            input_token_len=3,
            generate_token_len=1,
            finish_reason=finish_reason,
            cached_tokens=0,
            logprobs=None,
            routed_experts=None,
            cache_block_ids=None,
        )
    ]
    context = _FakeServerContext(_InvalidParser, outputs)

    async def _run():
        chat_runner = await ChatRunner.prepare(
            context,
            _request(**request_kwargs),
        )
        return await chat_runner.collect()

    result = asyncio.run(_run())

    assert result.finish_reason == expected


def test_runner_rejects_required_tool_choice_for_unsupported_response_parser():
    context = _FakeServerContext(_Parser)

    with pytest.raises(RequestError) as exc_info:
        asyncio.run(ChatRunner.prepare(
            context,
            _request(tool_choice='required', tools=_tools()),
        ))

    assert exc_info.value.code == ErrorCode.INVALID_REQUEST
    assert 'does not support `tool_choice="required"`' in exc_info.value.message


def test_runner_stream_chunks_preserve_metadata():
    context = _FakeServerContext(_Parser)

    async def _run():
        chat_runner = await ChatRunner.prepare(
            context,
            _request(return_token_ids=True, return_routed_experts=True),
        )
        return [chunk async for chunk in chat_runner.stream()]

    chunks = asyncio.run(_run())

    assert len(chunks) == 1
    assert chunks[0].delta_message.content == 'ok'
    assert chunks[0].token_ids == [1]
    assert chunks[0].logprobs == [{1: -0.1}]
    assert chunks[0].routed_experts == [[1]]
    assert chunks[0].reasoning_tokens == 2
    assert context.async_engine.generator_closed is True
    assert context.session_manager.removed == [context.session_manager.session]


def test_runner_close_cleans_prepared_unconsumed_request():
    context = _FakeServerContext(_Parser)

    async def _run():
        chat_runner = await ChatRunner.prepare(context, _request())
        await chat_runner.close()
        await chat_runner.close()

    asyncio.run(_run())

    assert context.session_manager.removed == [context.session_manager.session]


def test_runner_parser_complete_error_raises_request_error():
    class _FailingParser(_Parser):

        def parse_complete(self, text: str, token_ids: list[int] | None = None, **kwargs):
            raise RuntimeError('bad tool payload')

    async def _run():
        chat_runner = await ChatRunner.prepare(_FakeServerContext(_FailingParser), _request())
        return await chat_runner.collect()

    with pytest.raises(RequestError) as exc_info:
        asyncio.run(_run())

    assert exc_info.value.code == ErrorCode.INVALID_REQUEST
    assert 'bad tool payload' in exc_info.value.message


def test_runner_engine_error_is_not_reclassified_as_request_error():
    class _FailingEngine(_FakeEngine):

        def generate(self, request, **kwargs):
            async def _generator():
                if False:
                    yield None
                raise RuntimeError('engine exploded')

            return _generator()

    context = _FakeServerContext(_Parser)
    context.async_engine = _FailingEngine()

    async def _run():
        chat_runner = await ChatRunner.prepare(context, _request())
        return await chat_runner.collect()

    with pytest.raises(RuntimeError, match='engine exploded'):
        asyncio.run(_run())
