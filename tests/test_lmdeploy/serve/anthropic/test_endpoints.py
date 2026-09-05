from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from lmdeploy.serve.anthropic.protocol import CountTokensRequest, MessagesRequest
from lmdeploy.serve.anthropic.router import create_anthropic_router
from lmdeploy.serve.anthropic.streaming import stream_messages_response
from lmdeploy.serve.core.chat_runner import ChatStreamChunk
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)
from lmdeploy.serve.parsers.response_parser import BaseResponseParser
from lmdeploy.serve.parsers.tool_parser.interns2preview_tool_parser import InternS2PreviewToolParser
from lmdeploy.serve.utils.server_utils import protocol_error_response

ANTHROPIC_HEADERS = {'anthropic-version': '2023-06-01'}
DEFAULT_MESSAGES = [{'role': 'user', 'content': 'Hi there'}]
SEARCH_TOOL = {
    'name': 'search',
    'description': 'demo',
    'input_schema': {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string'
            }
        },
        'required': ['query'],
    },
}


class _FakeSession:

    def __init__(self, session_id: int):
        self.session_id = session_id
        self.aborted = False

    async def async_abort(self):
        self.aborted = True


class _FakeSessionManager:

    def __init__(self):
        self.next_session_id = 0
        self.removed = []

    def get(self):
        session = _FakeSession(self.next_session_id)
        self.next_session_id += 1
        return session

    def remove(self, session):
        self.removed.append(session)


class _FakeTokenizer:

    def encode(self, text: str, add_bos: bool = True, **kwargs):
        tokens = text.split()
        if add_bos:
            return [0] + list(range(1, len(tokens) + 1))
        return list(range(len(tokens)))


class _FakeChatTemplate:

    def messages2prompt(self, messages, **kwargs):
        parts = [f"{item['role']}:{item['content']}" for item in messages]
        return '\n'.join(parts)


class _SystemFirstChatTemplate(_FakeChatTemplate):

    def messages2prompt(self, messages, **kwargs):
        for index, message in enumerate(messages):
            if message['role'] == 'system' and index != 0:
                raise ValueError('System message must be at the beginning.')
        parts = [f"{item['role']}:{item['content']}" for item in messages]
        return '\n'.join(parts)


class _FakeEngine:

    def __init__(
            self,
            *,
            logprobs_mode='raw_logprobs',
            enable_return_routed_experts: bool = True,
            chat_template=None,
    ):
        self.model_name = 'fake-model'
        self.backend_config = SimpleNamespace(adapters=['adapter-model'],
                                             logprobs_mode=logprobs_mode,
                                             enable_return_routed_experts=enable_return_routed_experts)
        self.tokenizer = _FakeTokenizer()
        self.chat_template = chat_template or _FakeChatTemplate()
        self.preprocess_calls = []
        self.generate_calls = []

    async def preprocess(self, *args, **kwargs):
        self.preprocess_calls.append((args, kwargs))
        return SimpleNamespace(args=args, kwargs=kwargs)

    def generate(self, request, **kwargs):
        self.generate_calls.append((request, kwargs))

        async def _gen():
            yield SimpleNamespace(
                response='Hello ',
                token_ids=[101],
                input_token_len=8,
                generate_token_len=1,
                finish_reason=None,
                cached_tokens=0,
                cache_block_ids=None,
                routed_experts=[[[1, 2, 3]]],
                logprobs=[{101: -0.5, 102: -1.2}],
            )
            yield SimpleNamespace(
                response='world!',
                token_ids=[102],
                input_token_len=8,
                generate_token_len=2,
                finish_reason='stop',
                cached_tokens=0,
                cache_block_ids=None,
                routed_experts=[[[1, 2, 3]]],
                logprobs=[{102: -0.3, 103: -2.1}],
            )

        return _gen()


class _BasicParser:
    tool_parser_cls = None

    def __init__(self, request):
        self.request = request
        self.tool_parser = None
        self.reasoning_tokens = None

    def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
        return [(DeltaMessage(role='assistant', content=delta_text), False)]

    def parse_complete(self, text: str, token_ids: list[int] | None = None, **kwargs):
        return text, None, None

    def validate_complete(self, text: str | None = None):
        return True


class _FakeServerContext:
    def __init__(
            self,
            *,
            response_parser_cls=_BasicParser,
            logprobs_mode='raw_logprobs',
            enable_return_routed_experts: bool = True,
            chat_template=None,
    ):
        self.session_mgr = _FakeSessionManager()
        self.async_engine = _FakeEngine(
            logprobs_mode=logprobs_mode,
            enable_return_routed_experts=enable_return_routed_experts,
            chat_template=chat_template,
        )
        self.async_engine.session_mgr = self.session_mgr
        self.default_gen_config = {}
        self.response_parser_cls = response_parser_cls

    @property
    def engine_config(self):
        return self.async_engine.backend_config

    @property
    def session_manager(self):
        return self.async_engine.session_mgr

    def create_session(self, _session_id: int | None = None):
        return self.session_mgr.get()

class _FakeRawRequest:

    def __init__(self, headers):
        self.headers = headers

    async def is_disconnected(self):
        return False


class _TestResponse:

    def __init__(self, status_code: int, payload=None, body: str = ''):
        self.status_code = status_code
        self._payload = jsonable_encoder(payload)
        self._body = body

    def json(self):
        return self._payload

    def iter_lines(self):
        return self._body.splitlines()


class _StreamContext:

    def __init__(self, response: _TestResponse):
        self.response = response

    def __enter__(self):
        return self.response

    def __exit__(self, *args):
        return False


class _AnthropicTestClient:

    def __init__(self, server_context):
        router = create_anthropic_router(server_context)
        self._routes = {route.path: route.endpoint for route in router.routes}

    def post(self, path: str, *, headers, json):
        return asyncio.run(self._post(path, headers=headers, json=json))

    def stream(self, method: str, path: str, *, headers, json):
        assert method == 'POST'
        return _StreamContext(self.post(path, headers=headers, json=json))

    def get(self, path: str):
        return asyncio.run(self._get(path))

    async def _post(self, path: str, *, headers, json):
        path = path.split('?', 1)[0]
        endpoint = self._routes[path]
        request_cls = CountTokensRequest if path == '/v1/messages/count_tokens' else MessagesRequest
        try:
            request = request_cls(**json)
        except ValidationError as exc:
            result = _validation_error_response(path, exc)
            return await self._response_from_result(result)
        result = await endpoint(request, _FakeRawRequest(headers))
        return await self._response_from_result(result)

    async def _get(self, path: str):
        endpoint = self._routes[path.split('?', 1)[0]]
        return await self._response_from_result(await endpoint())

    async def _response_from_result(self, result):
        if isinstance(result, JSONResponse):
            return _TestResponse(result.status_code, json.loads(result.body))
        if isinstance(result, StreamingResponse):
            chunks = []
            async for chunk in result.body_iterator:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode()
                chunks.append(chunk)
            return _TestResponse(result.status_code, body=''.join(chunks))
        return _TestResponse(200, result)


class _ToolAndReasoningParser:
    tool_parser_cls = object

    def __init__(self, request):
        self.request = request
        self.tool_parser = object()
        self.reasoning_tokens = None

    def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
        if delta_text.startswith('Hello'):
            return [(DeltaMessage(role='assistant',
                                  reasoning_content='internal reasoning',
                                  content='visible text'), False)]
        if delta_text.startswith('world'):
            return [(
                DeltaMessage(
                    role='assistant',
                    tool_calls=[
                        DeltaToolCall(
                            index=0,
                            id='toolu_123',
                            function=DeltaFunctionCall(
                                name='search',
                                arguments='{"query":"lmdeploy"}',
                            ),
                        )
                    ],
                ),
                True,
            )]
        return []

    def parse_complete(self, text: str, token_ids: list[int] | None = None, **kwargs):
        return (
            'visible text',
            [
                ToolCall(
                    id='toolu_123',
                    function=FunctionCall(
                        name='search',
                        arguments='{"query":"lmdeploy"}',
                    ),
                )
            ],
            'internal reasoning',
        )

    def validate_complete(self, text: str | None = None):
        return True


class _IncompleteToolParser(_ToolAndReasoningParser):
    validate_calls = 0
    last_text = None

    def validate_complete(self, text: str | None = None):
        type(self).validate_calls += 1
        type(self).last_text = text
        return False


def _make_client(response_parser_cls=_BasicParser,
                 *,
                 server_context=None,
                 logprobs_mode='raw_logprobs',
                 return_context=False):
    context = server_context or _FakeServerContext(response_parser_cls=response_parser_cls,
                                                  logprobs_mode=logprobs_mode)
    client = _AnthropicTestClient(context)
    if return_context:
        return client, context
    return client


def _messages_payload(**overrides):
    payload = {
        'model': 'fake-model',
        'max_tokens': 16,
        'messages': DEFAULT_MESSAGES,
    }
    payload.update(overrides)
    return payload


def _validation_error_response(path: str, exc: ValidationError):
    first_error = next(iter(exc.errors()), {})
    location = '.'.join(str(part) for part in first_error.get('loc', ()))
    detail = first_error.get('msg', 'Invalid request body.')
    message = f'{location}: {detail}' if location else detail
    return protocol_error_response(path, RequestError(ErrorCode.INVALID_REQUEST, message))


def test_messages_non_stream():
    client, context = _make_client(return_context=True)
    response = _post_messages(client)

    assert response.status_code == 200
    data = response.json()
    assert data['type'] == 'message'
    assert data['content'][0]['type'] == 'text'
    assert data['content'][0]['text'] == 'Hello world!'
    assert data['stop_reason'] == 'end_turn'
    assert data['usage']['input_tokens'] == 8
    assert data['usage']['output_tokens'] == 2
    assert len(context.session_mgr.removed) == 1


@pytest.mark.parametrize(
    ('field_name', 'value'),
    [
        pytest.param('temperature', -0.1, id='temperature-below-range'),
        pytest.param('temperature', 1.1, id='temperature-above-range'),
        pytest.param('top_p', -0.1, id='top-p-below-range'),
        pytest.param('top_p', 1.1, id='top-p-above-range'),
        pytest.param('top_k', -1, id='negative-top-k'),
    ],
)
def test_messages_rejects_invalid_sampling_parameter(field_name, value):
    response = _post_messages(_make_client(), **{field_name: value})

    assert response.status_code == 400
    data = response.json()
    assert data['type'] == 'error'
    assert field_name in data['error']['message']


def test_messages_count_tokens_rejects_empty_messages():
    response = _post_count_tokens(_make_client(), messages=[])

    assert response.status_code == 400
    assert response.json() == {
        'type': 'error',
        'error': {
            'type': 'invalid_request_error',
            'message': 'messages: at least one message is required',
        },
    }


def _post_messages(client: _AnthropicTestClient, **overrides):
    return client.post('/v1/messages', headers=ANTHROPIC_HEADERS, json=_messages_payload(**overrides))


def _count_tokens_payload(**overrides):
    payload = {
        'model': 'fake-model',
        'messages': DEFAULT_MESSAGES,
    }
    payload.update(overrides)
    return payload


def _post_count_tokens(client: _AnthropicTestClient, **overrides):
    return client.post('/v1/messages/count_tokens', headers=ANTHROPIC_HEADERS, json=_count_tokens_payload(**overrides))


def _stream_messages_body(client: _AnthropicTestClient, **overrides):
    payload = _messages_payload(**overrides)
    payload['stream'] = True
    with client.stream('POST', '/v1/messages', headers=ANTHROPIC_HEADERS, json=payload) as response:
        return response.status_code, '\n'.join(response.iter_lines())


def _sse_payloads(body: str):
    return [
        json.loads(line.removeprefix('data: '))
        for line in body.splitlines()
        if line.startswith('data: ')
    ]


async def _parsed_stream(result_generator, response_parser):
    streaming_tools = False
    async for res in result_generator:
        token_ids = res.token_ids if getattr(res, 'token_ids', None) is not None else []
        stream_deltas = response_parser.stream_chunk(
            res.response or '',
            token_ids,
            final=res.finish_reason is not None,
        )
        if not stream_deltas:
            if res.finish_reason is None and not token_ids:
                continue
            stream_deltas = [(DeltaMessage(role='assistant', content=''), False)]

        for delta_index, (delta_message, tool_emitted) in enumerate(stream_deltas):
            if tool_emitted:
                streaming_tools = True
            is_last_delta = delta_index == len(stream_deltas) - 1
            finish_reason = res.finish_reason if is_last_delta else None
            if finish_reason == 'stop' and streaming_tools:
                finish_reason = 'tool_calls'
            yield ChatStreamChunk(
                delta_message=delta_message,
                tool_emitted=tool_emitted,
                finish_reason=finish_reason,
                token_ids=token_ids,
                logprobs=getattr(res, 'logprobs', None),
                input_token_len=res.input_token_len,
                generate_token_len=res.generate_token_len,
                cached_tokens=getattr(res, 'cached_tokens', 0),
                routed_experts=getattr(res, 'routed_experts', None) if finish_reason is not None else None,
                reasoning_tokens=getattr(response_parser, 'reasoning_tokens', None),
                is_last_delta=is_last_delta,
            )


def _collect_stream_response_payloads(result_generator, response_parser, **kwargs):
    async def _collect_events():
        return [
            event async for event in stream_messages_response(
                _parsed_stream(result_generator, response_parser),
                request_id='msg_test',
                request=ChatCompletionRequest(
                    model='fake-model',
                    messages=[],
                    **kwargs,
                ),
            )
        ]

    return _sse_payloads('\n'.join(asyncio.run(_collect_events())))


def test_stream_messages_runtime_exception_emits_error_event():

    async def _result_generator():
        if False:
            yield None
        raise RequestError(ErrorCode.INTERNAL_ERROR)

    payloads = _collect_stream_response_payloads(
        _result_generator(),
        _BasicParser(SimpleNamespace()),
    )

    assert payloads == [{
        'type': 'error',
        'error': {
            'type': 'api_error',
            'message': 'An internal server error occurred.',
        },
    }]


def test_messages_return_routed_experts_requires_engine_flag():
    client = _make_client(server_context=_FakeServerContext(enable_return_routed_experts=False))
    response = _post_messages(client, return_routed_experts=True)

    assert response.status_code == 400
    assert 'enable-return-routed-experts' in response.json()['error']['message']


def test_messages_tools_require_tool_parser():
    response = _post_messages(_make_client(), tools=[SEARCH_TOOL])

    assert response.status_code == 400
    assert '--tool-call-parser' in response.json()['error']['message']


@pytest.mark.parametrize('tool_choice', ['any', {'type': 'any'}])
def test_messages_any_tool_choice_requires_tools(tool_choice):
    response = _post_messages(
        _make_client(response_parser_cls=_ToolAndReasoningParser),
        tool_choice=tool_choice,
    )

    assert response.status_code == 400
    assert 'requires at least one tool' in response.json()['error']['message']


@pytest.mark.parametrize(
    ('tools', 'tool_choice', 'error_fragment'),
    [
        (None, {'type': 'tool', 'name': 'search'}, 'requires at least one tool'),
        ([SEARCH_TOOL], {'type': 'tool', 'name': 'missing'}, "not found in `tools`: 'missing'"),
    ],
)
def test_messages_named_tool_choice_validation(tools, tool_choice, error_fragment):
    response = _post_messages(
        _make_client(response_parser_cls=_ToolAndReasoningParser),
        tools=tools,
        tool_choice=tool_choice,
    )

    assert response.status_code == 400
    assert error_fragment in response.json()['error']['message']


def test_messages_beta_accepts_system_role_message():
    context = _FakeServerContext()
    client = _make_client(server_context=context)
    response = client.post(
        '/v1/messages?beta=true',
        headers=ANTHROPIC_HEADERS,
        json=_messages_payload(
            messages=[
                {
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': '<system-reminder>Use the repo instructions.</system-reminder>',
                    }],
                },
                {
                    'role': 'system',
                    'content': [{
                        'type': 'text',
                        'text': 'Project instructions from Claude Code.',
                    }],
                },
                {
                    'role': 'user',
                    'content': 'hi',
                },
            ],
        ),
    )

    assert response.status_code == 200
    args, _kwargs = context.async_engine.preprocess_calls[-1]
    assert args[0] == [
        {
            'role': 'user',
            'content': '<system-reminder>Use the repo instructions.</system-reminder>',
        },
        {
            'role': 'system',
            'content': 'Project instructions from Claude Code.',
        },
        {
            'role': 'user',
            'content': 'hi',
        },
    ]


def test_messages_merges_inline_system_for_system_first_template():
    context = _FakeServerContext(chat_template=_SystemFirstChatTemplate())
    response = _post_messages(
        _make_client(server_context=context),
        system='Top-level.',
        messages=[
            {
                'role': 'user',
                'content': 'first',
            },
            {
                'role': 'system',
                'content': 'Inline.',
            },
            {
                'role': 'user',
                'content': 'second',
            },
        ],
    )

    assert response.status_code == 200
    args, _kwargs = context.async_engine.preprocess_calls[-1]
    assert args[0] == [
        {
            'role': 'system',
            'content': 'Top-level.Inline.',
        },
        {
            'role': 'user',
            'content': 'first',
        },
        {
            'role': 'user',
            'content': 'second',
        },
    ]


def test_messages_count_tokens_merges_inline_system_for_system_first_template():
    context = _FakeServerContext(chat_template=_SystemFirstChatTemplate())
    response = _post_count_tokens(
        _make_client(server_context=context),
        messages=[
            {
                'role': 'user',
                'content': 'first',
            },
            {
                'role': 'system',
                'content': 'Inline.',
            },
            {
                'role': 'user',
                'content': 'second',
            },
        ],
    )

    assert response.status_code == 200


def test_messages_non_stream_with_reasoning_and_tool_use_blocks():
    client = _make_client(response_parser_cls=_ToolAndReasoningParser)
    response = _post_messages(
        client,
        tools=[SEARCH_TOOL],
        tool_choice={
            'type': 'tool',
            'name': 'search',
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data['stop_reason'] == 'tool_use'
    assert data['content'][0] == {
        'type': 'thinking',
        'thinking': 'internal reasoning',
        'signature': 'lmdeploy-local',
    }
    assert data['content'][1] == {'type': 'text', 'text': 'visible text'}
    assert data['content'][2]['type'] == 'tool_use'
    assert data['content'][2]['name'] == 'search'
    assert data['content'][2]['input'] == {'query': 'lmdeploy'}


def test_messages_non_stream_validate_complete_marks_parse_error():
    _IncompleteToolParser.validate_calls = 0
    _IncompleteToolParser.last_text = None
    client = _make_client(response_parser_cls=_IncompleteToolParser)
    response = _post_messages(client, tools=[SEARCH_TOOL], return_token_ids=True)

    assert response.status_code == 200
    assert response.json()['stop_reason'] == 'parse_error'
    assert _IncompleteToolParser.validate_calls == 1
    assert _IncompleteToolParser.last_text == 'Hello world!'


def test_messages_streaming_usage_matches_anthropic_event_spec():
    client, context = _make_client(return_context=True)
    status_code, body = _stream_messages_body(client)
    payloads = _sse_payloads(body)
    message_start = next(item for item in payloads if item['type'] == 'message_start')
    message_delta = next(item for item in payloads if item['type'] == 'message_delta')

    assert status_code == 200
    assert message_start['message']['usage'] == {
        'input_tokens': 8,
        'output_tokens': 1,
    }
    assert message_delta['usage'] == {'output_tokens': 2}
    assert len(context.session_mgr.removed) == 1


def test_messages_unconsumed_streaming_response_cleans_up_session():
    context = _FakeServerContext()
    router = create_anthropic_router(context)
    endpoint = next(route.endpoint for route in router.routes if route.path == '/v1/messages')

    async def _close_without_consuming():
        response = await endpoint(
            MessagesRequest(**_messages_payload(stream=True)),
            _FakeRawRequest(ANTHROPIC_HEADERS),
        )
        await response.close()

    asyncio.run(_close_without_consuming())

    assert len(context.session_mgr.removed) == 1


def test_messages_streaming_with_reasoning_and_tool_use_events():
    client = _make_client(response_parser_cls=_ToolAndReasoningParser)
    status_code, body = _stream_messages_body(client, tools=[SEARCH_TOOL], return_token_ids=True)
    payloads = _sse_payloads(body)
    thinking_events = [payload for payload in payloads if payload.get('index') == 0]

    assert status_code == 200
    assert thinking_events == [
        {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {
                'type': 'thinking',
                'thinking': '',
            },
        },
        {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {
                'type': 'thinking_delta',
                'thinking': 'internal reasoning',
            },
        },
        {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {
                'type': 'signature_delta',
                'signature': 'lmdeploy-local',
            },
        },
        {
            'type': 'content_block_stop',
            'index': 0,
        },
    ]
    assert '"type": "input_json_delta"' in body
    assert '"type": "tool_use"' in body
    assert '"output_ids": [102]' in body


def test_messages_streaming_validate_complete_marks_parse_error():
    _IncompleteToolParser.validate_calls = 0
    _IncompleteToolParser.last_text = None
    client = _make_client(response_parser_cls=_IncompleteToolParser)
    status_code, body = _stream_messages_body(client, tools=[SEARCH_TOOL], return_token_ids=True)
    payloads = _sse_payloads(body)
    message_delta = next(item for item in payloads if item['type'] == 'message_delta')

    assert status_code == 200
    assert message_delta['delta']['stop_reason'] == 'parse_error'
    assert _IncompleteToolParser.validate_calls == 1
    assert _IncompleteToolParser.last_text is None


def test_stream_messages_response_serializes_numpy_routed_experts():
    import numpy as np

    async def _result_generator():
        yield SimpleNamespace(
            response='Hello',
            token_ids=[1],
            input_token_len=2,
            generate_token_len=1,
            finish_reason='stop',
            routed_experts=np.array([[[1, 2]]]),
            logprobs=None,
        )

    payloads = _collect_stream_response_payloads(
        _result_generator(),
        _BasicParser(None),
        return_routed_experts=True,
    )
    message_delta = next(item for item in payloads if item['type'] == 'message_delta')

    assert message_delta['routed_experts'] == [[[1, 2]]]


def test_stream_messages_response_preserves_tool_start_output_ids():
    class _ToolStartParser:

        def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
            return [(
                DeltaMessage(
                    role='assistant',
                    tool_calls=[
                        DeltaToolCall(
                            index=0,
                            id='toolu_123',
                            function=DeltaFunctionCall(name='search', arguments=''),
                        )
                    ],
                ),
                True,
            )]

    async def _result_generator():
        yield SimpleNamespace(
            response='<tool_call><function=search>',
            token_ids=[11, 12, 13],
            input_token_len=8,
            generate_token_len=3,
            finish_reason=None,
            routed_experts=None,
            logprobs=None,
        )

    payloads = _collect_stream_response_payloads(
        _result_generator(),
        _ToolStartParser(),
        return_token_ids=True,
    )
    output_ids = [
        token_id for item in payloads
        if item['type'] == 'content_block_delta'
        for token_id in item.get('output_ids', [])
    ]

    assert output_ids == [11, 12, 13]


def test_stream_messages_response_closes_text_before_resuming_tool_delta():
    class _InterleavedToolParser:
        def __init__(self):
            self.calls = 0

        def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
            self.calls += 1
            if self.calls == 1:
                return [(
                    DeltaMessage(
                        role='assistant',
                        tool_calls=[
                            DeltaToolCall(
                                index=0,
                                id='toolu_123',
                                function=DeltaFunctionCall(
                                    name='search',
                                    arguments='{"query":',
                                ),
                            )
                        ],
                    ),
                    True,
                )]
            if self.calls == 2:
                return [(DeltaMessage(role='assistant', content='interlude'), False)]
            return [(
                DeltaMessage(
                    role='assistant',
                    tool_calls=[
                        DeltaToolCall(
                            index=0,
                            id='toolu_123',
                            function=DeltaFunctionCall(arguments='"lmdeploy"}'),
                        )
                    ],
                ),
                True,
            )]

    async def _result_generator():
        for idx, finish_reason in enumerate([None, None, 'stop'], start=1):
            yield SimpleNamespace(
                response=f'chunk-{idx}',
                token_ids=[idx],
                input_token_len=8,
                generate_token_len=idx,
                finish_reason=finish_reason,
            )

    payloads = _collect_stream_response_payloads(
        _result_generator(),
        _InterleavedToolParser(),
    )

    tool_start = next(
        item for item in payloads
        if item['type'] == 'content_block_start' and item['content_block']['type'] == 'tool_use')
    assert tool_start['content_block']['name'] == 'search'

    resumed_tool_delta_index = next(
        idx for idx, item in enumerate(payloads)
        if item['type'] == 'content_block_delta' and item['delta']['type'] == 'input_json_delta'
        and item['delta']['partial_json'] == '"lmdeploy"}')
    assert any(
        item['type'] == 'content_block_stop' and item['index'] == 1
        for item in payloads[:resumed_tool_delta_index])


def test_stream_messages_response_interns2preview_inter_tool_whitespace_uses_text_block():
    """Keep InternS2Preview's inter-tool newline off open tool-use blocks."""

    class _InternS2PreviewResponseParser(BaseResponseParser):
        reasoning_parser_cls = None
        tool_parser_cls = InternS2PreviewToolParser

    request = ChatCompletionRequest(
        model='fake-model',
        messages=[],
        stream=True,
        tool_choice='auto',
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'city': {
                                'type': 'string'
                            }
                        },
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_news',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'topic': {
                                'type': 'string'
                            }
                        },
                    },
                },
            },
        ],
    )
    response_parser = _InternS2PreviewResponseParser(request)
    raw_response = (
        '<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>'
        '\n'
        '<tool_call>\n<function=get_news>\n<parameter=topic>France</parameter>\n</function>\n</tool_call>')

    async def _result_generator():
        yield SimpleNamespace(
            response=raw_response,
            token_ids=[1],
            input_token_len=8,
            generate_token_len=1,
            finish_reason='stop',
        )

    payloads = _collect_stream_response_payloads(_result_generator(), response_parser)
    block_events = [item for item in payloads if item['type'].startswith('content_block_')]

    assert [(item['type'], item['index']) for item in block_events] == [
        ('content_block_start', 0),
        ('content_block_delta', 0),
        ('content_block_stop', 0),
        ('content_block_start', 1),
        ('content_block_delta', 1),
        ('content_block_stop', 1),
        ('content_block_start', 2),
        ('content_block_delta', 2),
        ('content_block_stop', 2),
    ]
    assert [
        item['content_block']['type'] for item in block_events
        if item['type'] == 'content_block_start'
    ] == ['tool_use', 'text', 'tool_use']
    assert [
        item['delta'] for item in block_events
        if item['type'] == 'content_block_delta'
    ] == [
        {
            'type': 'input_json_delta',
            'partial_json': '{"city": "Paris"}',
        },
        {
            'type': 'text_delta',
            'text': '\n',
        },
        {
            'type': 'input_json_delta',
            'partial_json': '{"topic": "France"}',
        },
    ]


def test_stream_messages_response_maps_stop_to_tool_use_on_empty_terminal_chunk():
    class _ToolParser:
        def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
            if delta_text == 'chunk-1':
                return [(
                    DeltaMessage(
                        role='assistant',
                        tool_calls=[
                            DeltaToolCall(
                                index=0,
                                id='toolu_123',
                                function=DeltaFunctionCall(
                                    name='search',
                                    arguments='{"query":"lmdeploy"}',
                                ),
                            )
                        ],
                    ),
                    True,
                )]
            return []

    async def _result_generator():
        for idx, finish_reason in enumerate([None, 'stop'], start=1):
            yield SimpleNamespace(
                response=f'chunk-{idx}',
                token_ids=[idx],
                input_token_len=8,
                generate_token_len=idx,
                finish_reason=finish_reason,
            )

    payloads = _collect_stream_response_payloads(
        _result_generator(),
        _ToolParser(),
    )

    message_delta = next(item for item in payloads if item['type'] == 'message_delta')
    assert message_delta['delta']['stop_reason'] == 'tool_use'


def test_messages_non_stream_with_output_ids_and_routed_experts():
    client = _make_client()
    response = _post_messages(
        client,
        messages=[{'role': 'user', 'content': 'Hi'}],
        return_token_ids=True,
        return_routed_experts=True,
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get('output_ids') == [101, 102]
    assert data.get('routed_experts') == [[[1, 2, 3]]]


@pytest.mark.parametrize(
    ('overrides', 'error_fragment'),
    [
        pytest.param({'input_ids': [1, 2, 3]}, 'input_ids', id='input_ids-with-messages'),
        pytest.param({'image_data': 'https://example.com/img.png'}, 'image_data', id='image-data-with-messages'),
        pytest.param({
            'messages': [],
            'image_data': 'https://example.com/img.png',
        }, 'input_ids', id='image-data-without-input-ids'),
        pytest.param({
            'messages': [],
            'input_ids': [],
        }, 'input_ids', id='empty-input-ids'),
        pytest.param({'messages': []}, 'messages', id='empty-messages-without-input-ids'),
        pytest.param({
            'messages': [],
            'input_ids': [1, 2, 3],
            'system': 'ignored system prompt',
        }, 'system', id='system-with-input-ids'),
    ],
)
def test_messages_rejects_invalid_input_combinations(overrides, error_fragment):
    response = _post_messages(_make_client(), **overrides)

    assert response.status_code == 400
    assert error_fragment in response.json()['error']['message']


def test_messages_image_data_preserves_input_ids_in_multimodal_content():
    context = _FakeServerContext()
    client = _make_client(server_context=context)
    response = _post_messages(
        client,
        messages=[],
        input_ids=[1, 2, 3],
        image_data='https://example.com/img.png',
    )
    assert response.status_code == 200
    args, kwargs = context.async_engine.preprocess_calls[-1]
    messages_arg = args[0]
    assert messages_arg[0]['content'][0] == {'type': 'text', 'text': [1, 2, 3]}
    assert kwargs['input_ids'] is None


def test_messages_rejects_tools_with_input_ids():
    context = _FakeServerContext(response_parser_cls=_ToolAndReasoningParser)
    client = _make_client(server_context=context)
    response = _post_messages(
        client,
        messages=[],
        input_ids=[1, 2, 3],
        tools=[{
            'name': 'search',
            'description': 'demo',
            'input_schema': {
                'type': 'object',
                'properties': {},
            },
        }],
        tool_choice={
            'type': 'auto',
        },
    )
    assert response.status_code == 400
    assert 'tools cannot be used when input_ids is set' in response.json()['error']['message']


def test_anthropic_model_listing():
    client = _make_client()
    response = client.get('/anthropic/v1/models')
    assert response.status_code == 200
    data = response.json()
    assert data['has_more'] is False
    assert [item['id'] for item in data['data']] == ['fake-model', 'adapter-model']


def test_messages_non_stream_includes_logprobs():
    client = _make_client()
    response = _post_messages(client, messages=[{'role': 'user', 'content': 'Hi'}], return_logprob=True)
    assert response.status_code == 200
    data = response.json()
    # output_token_logprobs should be [(logprob, token_id), ...]
    assert data['output_token_logprobs'] == [[-0.5, 101], [-0.3, 102]]


def test_messages_rejects_logprobs_when_engine_logprobs_mode_disabled():
    client = _make_client(logprobs_mode=None)
    response = _post_messages(client, messages=[{'role': 'user', 'content': 'Hi'}], return_logprob=True)
    assert response.status_code == 400
    assert 'return_logprob' in response.json()['error']['message']
