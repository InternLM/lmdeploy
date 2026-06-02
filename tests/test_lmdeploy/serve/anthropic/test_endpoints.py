from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lmdeploy.serve.anthropic.router import create_anthropic_router
from lmdeploy.serve.anthropic.streaming import stream_messages_response
from lmdeploy.serve.openai.protocol import DeltaFunctionCall, DeltaMessage, DeltaToolCall, FunctionCall, ToolCall

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

    def __init__(self):
        self.aborted = False

    async def async_abort(self):
        self.aborted = True


class _FakeTokenizer:

    def encode(self, text: str, add_bos: bool = True, **kwargs):
        tokens = text.split()
        if add_bos:
            return [0] + list(range(1, len(tokens) + 1))
        return list(range(len(tokens)))


class _FakeChatTemplate:

    def messages2prompt(self, messages, sequence_start: bool = True, **kwargs):
        parts = [f"{item['role']}:{item['content']}" for item in messages]
        return '\n'.join(parts)


class _FakeEngine:

    def __init__(self, *, logprobs_mode='raw_logprobs'):
        self.model_name = 'fake-model'
        self.backend_config = SimpleNamespace(adapters=['adapter-model'], logprobs_mode=logprobs_mode)
        self.tokenizer = _FakeTokenizer()
        self.chat_template = _FakeChatTemplate()
        self.generate_calls = []

    def generate(self, *args, **kwargs):
        self.generate_calls.append((args, kwargs))

        async def _gen():
            yield SimpleNamespace(
                response='Hello ',
                token_ids=[101],
                input_token_len=8,
                generate_token_len=1,
                finish_reason=None,
                routed_experts=[[[1, 2, 3]]],
                logprobs=[{101: -0.5, 102: -1.2}],
            )
            yield SimpleNamespace(
                response='world!',
                token_ids=[102],
                input_token_len=8,
                generate_token_len=2,
                finish_reason='stop',
                routed_experts=[[[1, 2, 3]]],
                logprobs=[{102: -0.3, 103: -2.1}],
            )

        return _gen()


class _BasicParser:
    tool_parser_cls = None

    def __init__(self, request):
        self.request = request

    def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
        return [(DeltaMessage(role='assistant', content=delta_text), False)]

    def parse_complete(self, text: str, token_ids: list[int] | None = None, **kwargs):
        return text, None, None

    def validate_complete(self, text: str | None = None):
        return True


class _FakeServerContext:
    def __init__(self, *, response_parser_cls=_BasicParser, logprobs_mode='raw_logprobs'):
        self.async_engine = _FakeEngine(logprobs_mode=logprobs_mode)
        self.response_parser_cls = response_parser_cls

    def create_session(self, _session_id: int | None = None):
        return _FakeSession()

    def get_engine_config(self):
        return self.async_engine.backend_config


class _ToolAndReasoningParser:
    tool_parser_cls = object

    def __init__(self, request):
        self.request = request

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


def _make_client(response_parser_cls=_BasicParser, *, server_context=None, logprobs_mode='raw_logprobs') -> TestClient:
    app = FastAPI()
    context = server_context or _FakeServerContext(response_parser_cls=response_parser_cls,
                                                  logprobs_mode=logprobs_mode)
    app.include_router(create_anthropic_router(context))
    return TestClient(app)


def _messages_payload(**overrides):
    payload = {
        'model': 'fake-model',
        'max_tokens': 16,
        'messages': DEFAULT_MESSAGES,
    }
    payload.update(overrides)
    return payload


def _post_messages(client: TestClient, **overrides):
    return client.post('/v1/messages', headers=ANTHROPIC_HEADERS, json=_messages_payload(**overrides))


def _stream_messages_body(client: TestClient, **overrides):
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


def _collect_stream_response_payloads(result_generator, response_parser, **kwargs):
    async def _collect_events():
        return [
            event async for event in stream_messages_response(
                result_generator,
                request_id='msg_test',
                model='fake-model',
                response_parser=response_parser,
                **kwargs,
            )
        ]

    return _sse_payloads('\n'.join(asyncio.run(_collect_events())))


def test_messages_non_stream():
    client = _make_client()
    response = _post_messages(client)
    assert response.status_code == 200
    data = response.json()
    assert data['type'] == 'message'
    assert data['content'][0]['type'] == 'text'
    assert data['content'][0]['text'] == 'Hello world!'
    assert data['stop_reason'] == 'end_turn'
    assert data['usage']['input_tokens'] == 8
    assert data['usage']['output_tokens'] == 2


def test_messages_requires_anthropic_version_header():
    client = _make_client()
    response = client.post(
        '/v1/messages',
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{
                'role': 'user',
                'content': 'Hi there',
            }],
        },
    )
    assert response.status_code == 400
    assert response.json()['error']['message'] == 'Missing required header: anthropic-version'


def test_messages_rejects_tools_without_tool_parser():
    client = _make_client()
    response = _post_messages(
        client,
        tools=[{
            'name': 'search',
            'description': 'demo',
            'input_schema': {
                'type': 'object',
                'properties': {},
            },
        }],
    )
    assert response.status_code == 400
    assert '--tool-call-parser' in response.json()['error']['message']


def test_messages_unknown_model():
    client = _make_client()
    response = _post_messages(client, model='missing-model')
    assert response.status_code == 404
    assert response.json()['error']['type'] == 'not_found_error'


def test_messages_beta_accepts_tool_history_blocks():
    client = _make_client()
    response = client.post(
        '/v1/messages?beta=true',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [
                {
                    'role': 'assistant',
                    'content': [{
                        'type': 'tool_use',
                        'id': 'toolu_123',
                        'name': 'search',
                        'input': {
                            'query': 'lmdeploy'
                        },
                    }],
                },
                {
                    'role': 'user',
                    'content': [{
                        'type': 'tool_result',
                        'tool_use_id': 'toolu_123',
                        'content': [{
                            'type': 'text',
                            'text': 'LMDeploy serves LLMs.',
                        }],
                    }],
                },
                {
                    'role': 'user',
                    'content': 'Summarize the tool result.',
                },
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data['type'] == 'message'


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
    assert data['content'][0] == {'type': 'thinking', 'thinking': 'internal reasoning'}
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


def test_messages_streaming_includes_output_ids():
    client = _make_client()
    status_code, body = _stream_messages_body(client, return_token_ids=True)

    assert status_code == 200
    # output_ids should appear in content_block_delta events
    assert 'output_ids' in body


def test_messages_streaming_includes_routed_experts():
    client = _make_client()
    status_code, body = _stream_messages_body(client, return_routed_experts=True)

    assert status_code == 200
    # routed_experts should appear in the message_delta event
    assert 'routed_experts' in body


def test_messages_streaming_usage_matches_anthropic_event_spec():
    client = _make_client()
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


def test_messages_streaming_with_reasoning_and_tool_use_events():
    client = _make_client(response_parser_cls=_ToolAndReasoningParser)
    status_code, body = _stream_messages_body(client, tools=[SEARCH_TOOL], return_token_ids=True)

    assert status_code == 200
    assert '"type": "thinking_delta"' in body
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
    """When return_token_ids is True, output_ids must be populated with the
    generated token IDs.

    When return_routed_experts is True, routed_experts must be populated from the engine result.
    """
    client = _make_client()
    # Test output_ids when return_token_ids is True
    response = _post_messages(
        client,
        messages=[{'role': 'user', 'content': 'Hi'}],
        return_token_ids=True,
        return_routed_experts=True,
    )
    assert response.status_code == 200
    data = response.json()
    # output_ids should be populated from token IDs generated by the fake engine
    assert data.get('output_ids') == [101, 102]
    assert data.get('routed_experts') == [[[1, 2, 3]]]

    # Test that optional metadata is None when not requested.
    response2 = _post_messages(client, messages=[{'role': 'user', 'content': 'Hi'}])
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2.get('output_ids') is None
    assert data2.get('routed_experts') is None


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


def test_messages_input_ids_non_stream():
    client = _make_client()
    response = _post_messages(client, messages=[], input_ids=[1, 2, 3])
    assert response.status_code == 200
    data = response.json()
    assert data['type'] == 'message'
    assert data['content'][0]['type'] == 'text'
    assert data['content'][0]['text'] == 'Hello world!'


def test_messages_input_ids_streaming():
    client = _make_client()
    status_code, body = _stream_messages_body(client, messages=[], input_ids=[1, 2, 3])

    assert status_code == 200
    assert 'event: message_start' in body
    assert 'event: content_block_delta' in body
    assert 'event: message_stop' in body


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
    args, kwargs = context.async_engine.generate_calls[-1]
    messages_arg = args[0]
    assert messages_arg[0]['content'][0] == {'type': 'text', 'text': [1, 2, 3]}
    assert kwargs['input_ids'] is None


def test_messages_accepts_tools_with_input_ids():
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
    assert response.status_code == 200
    args, kwargs = context.async_engine.generate_calls[-1]
    assert args[0] is None
    assert kwargs['input_ids'] == [1, 2, 3]
    assert kwargs['tools'][0].function.name == 'search'


def test_count_tokens():
    client = _make_client()
    response = client.post(
        '/v1/messages/count_tokens',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'messages': [{
                'role': 'user',
                'content': 'count these tokens',
            }],
            'system': 'You are helpful.',
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data['input_tokens'], int)
    assert data['input_tokens'] > 0


def test_count_tokens_accepts_tools():
    client = _make_client()
    response = client.post(
        '/v1/messages/count_tokens',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'messages': [{
                'role': 'user',
                'content': 'count these tokens',
            }],
            'tools': [{
                'name': 'search',
                'description': 'demo',
                'input_schema': {
                    'type': 'object',
                    'properties': {},
                },
            }],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data['input_tokens'], int)
    assert data['input_tokens'] > 0


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


def test_messages_streaming_includes_logprobs():
    client = _make_client()
    status_code, body = _stream_messages_body(client, return_logprob=True)

    assert status_code == 200
    assert 'output_token_logprobs' in body
