from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lmdeploy.serve.anthropic.endpoints import messages, messages_count_tokens, models
from lmdeploy.serve.anthropic.router import create_anthropic_router
from lmdeploy.serve.anthropic.streaming import stream_messages_response
from lmdeploy.serve.openai.protocol import DeltaFunctionCall, DeltaMessage, DeltaToolCall, FunctionCall, ToolCall


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
                routed_experts=[1, 2, 3],
                logprobs=[{101: -0.5, 102: -1.2}],
            )
            yield SimpleNamespace(
                response='world!',
                token_ids=[102],
                input_token_len=8,
                generate_token_len=2,
                finish_reason='stop',
                routed_experts=[1, 2, 3],
                logprobs=[{102: -0.3, 103: -2.1}],
            )

        return _gen()


class _FakeServerContext:
    def __init__(self, *, response_parser_cls=None, logprobs_mode='raw_logprobs'):
        self.async_engine = _FakeEngine(logprobs_mode=logprobs_mode)
        self.response_parser_cls = response_parser_cls

    def create_session(self, _session_id: int):
        return _FakeSession()

    def get_engine_config(self):
        return self.async_engine.backend_config


class _ToolAndReasoningParser:
    tool_parser_cls = object

    def __init__(self, request, tokenizer):
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


def _make_client(response_parser_cls=None, *, server_context=None, logprobs_mode='raw_logprobs') -> TestClient:
    app = FastAPI()
    context = server_context or _FakeServerContext(response_parser_cls=response_parser_cls,
                                                  logprobs_mode=logprobs_mode)
    app.include_router(create_anthropic_router(context))
    return TestClient(app)


def test_endpoint_modules_export_register():
    assert callable(messages.register)
    assert callable(messages_count_tokens.register)
    assert callable(models.register)


def test_messages_non_stream():
    client = _make_client()
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{
                'role': 'user',
                'content': 'Hi there',
            }],
        },
    )
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
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{
                'role': 'user',
                'content': 'Hi there',
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
    assert response.status_code == 400
    assert '--tool-call-parser' in response.json()['error']['message']


def test_messages_unknown_model():
    client = _make_client()
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'missing-model',
            'max_tokens': 16,
            'messages': [{
                'role': 'user',
                'content': 'Hi there',
            }],
        },
    )
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
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{
                'role': 'user',
                'content': 'Hi there',
            }],
            'tools': [{
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
            }],
            'tool_choice': {
                'type': 'tool',
                'name': 'search',
            },
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


def test_messages_streaming_includes_output_ids():
    client = _make_client()
    with client.stream(
            'POST',
            '/v1/messages',
            headers={'anthropic-version': '2023-06-01'},
            json={
                'model': 'fake-model',
                'max_tokens': 16,
                'stream': True,
                'messages': [{'role': 'user', 'content': 'Hi there'}],
                'return_token_ids': True,
            },
    ) as response:
        body = '\n'.join(response.iter_lines())

    assert response.status_code == 200
    # output_ids should appear in content_block_delta events
    assert 'output_ids' in body


def test_messages_streaming_includes_routed_experts():
    client = _make_client()
    with client.stream(
            'POST',
            '/v1/messages',
            headers={'anthropic-version': '2023-06-01'},
            json={
                'model': 'fake-model',
                'max_tokens': 16,
                'stream': True,
                'messages': [{'role': 'user', 'content': 'Hi there'}],
                'return_routed_experts': True,
            },
    ) as response:
        body = '\n'.join(response.iter_lines())

    assert response.status_code == 200
    # routed_experts should appear in the message_delta event
    assert 'routed_experts' in body


def test_messages_streaming_sse_shape():
    client = _make_client()
    with client.stream(
            'POST',
            '/v1/messages',
            headers={'anthropic-version': '2023-06-01'},
            json={
                'model': 'fake-model',
                'max_tokens': 16,
                'stream': True,
                'messages': [{
                    'role': 'user',
                    'content': 'Hi there',
                }],
            },
    ) as response:
        body = '\n'.join(response.iter_lines())

    assert response.status_code == 200
    assert 'event: message_start' in body
    assert 'event: content_block_start' in body
    assert 'event: content_block_delta' in body
    assert 'event: message_delta' in body
    assert 'event: message_stop' in body


def test_messages_streaming_with_reasoning_and_tool_use_events():
    client = _make_client(response_parser_cls=_ToolAndReasoningParser)
    with client.stream(
            'POST',
            '/v1/messages',
            headers={'anthropic-version': '2023-06-01'},
            json={
                'model': 'fake-model',
                'max_tokens': 16,
                'stream': True,
                'messages': [{
                    'role': 'user',
                    'content': 'Hi there',
                }],
                'tools': [{
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
                }],
            },
    ) as response:
        body = '\n'.join(response.iter_lines())

    assert response.status_code == 200
    assert '"type": "thinking_delta"' in body
    assert '"type": "input_json_delta"' in body
    assert '"type": "tool_use"' in body


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

    async def _collect_events():
        return [
            event async for event in stream_messages_response(
                _result_generator(),
                request_id='msg_test',
                model='fake-model',
                response_parser=_InterleavedToolParser(),
            )
        ]

    events = asyncio.run(_collect_events())
    payloads = [
        json.loads(line.removeprefix('data: ')) for event in events for line in event.splitlines()
        if line.startswith('data: ')
    ]

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

    async def _collect_events():
        return [
            event async for event in stream_messages_response(
                _result_generator(),
                request_id='msg_test',
                model='fake-model',
                response_parser=_ToolParser(),
            )
        ]

    events = asyncio.run(_collect_events())
    payloads = [
        json.loads(line.removeprefix('data: ')) for event in events for line in event.splitlines()
        if line.startswith('data: ')
    ]

    message_delta = next(item for item in payloads if item['type'] == 'message_delta')
    assert message_delta['delta']['stop_reason'] == 'tool_use'


def test_messages_return_routed_experts_in_generation_config():
    from lmdeploy.serve.anthropic.adapter import to_generation_config
    from lmdeploy.serve.anthropic.protocol import MessagesRequest
    req = MessagesRequest(
        model='fake-model',
        messages=[{'role': 'user', 'content': 'Hi'}],
        max_tokens=16,
        return_routed_experts=True,
    )
    cfg = to_generation_config(req)
    assert cfg.return_routed_experts is True


def test_messages_accepts_input_ids_and_image_data():
    """Extended fields input_ids, image_data, return_routed_experts, and
    return_token_ids must be accepted by the protocol model."""
    from lmdeploy.serve.anthropic.protocol import MessagesRequest

    req = MessagesRequest(
        model='fake-model',
        messages=[],
        max_tokens=16,
        input_ids=[1, 2, 3],
        image_data='https://example.com/img.png',
        return_routed_experts=True,
        return_token_ids=True,
    )
    assert req.input_ids == [1, 2, 3]
    assert req.image_data == 'https://example.com/img.png'
    assert req.return_routed_experts is True
    assert req.return_token_ids is True

    # Defaults
    req2 = MessagesRequest(model='m', messages=[], max_tokens=16)
    assert req2.input_ids is None
    assert req2.image_data is None
    assert req2.return_routed_experts is False
    assert req2.return_token_ids is False

    # Also verify the endpoint doesn't reject the extended fields via HTTP
    client = _make_client()
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [],
            'input_ids': [1, 2, 3],
        },
    )
    assert response.status_code != 422


def test_messages_non_stream_includes_output_ids_when_return_token_ids():
    """The response should include output_ids in the response-level field."""
    client = _make_client()
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    # output_ids is a new optional field; it should be present (possibly None)
    assert 'output_ids' in data


def test_messages_non_stream_includes_routed_experts():
    """The response should include routed_experts in the response-level
    field."""
    client = _make_client()
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    # routed_experts is a new optional field; it should be present (possibly None)
    assert 'routed_experts' in data


def test_messages_non_stream_with_output_ids_and_routed_experts():
    """When return_token_ids is True, output_ids must be populated with the
    generated token IDs.

    When return_routed_experts is True, routed_experts must be populated from the engine result.
    """
    client = _make_client()
    # Test output_ids when return_token_ids is True
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'return_token_ids': True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    # output_ids should be populated from token IDs generated by the fake engine
    assert data.get('output_ids') == [101, 102]

    # Test that output_ids is None when return_token_ids is False (default)
    response2 = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        },
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2.get('output_ids') is None


def test_messages_rejects_input_ids_with_non_empty_messages():
    client = _make_client()
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'input_ids': [1, 2, 3],
        },
    )
    assert response.status_code == 400
    assert 'input_ids' in response.json()['error']['message']


def test_messages_rejects_image_data_with_non_empty_messages():
    client = _make_client()
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'image_data': 'https://example.com/img.png',
        },
    )
    assert response.status_code == 400
    assert 'image_data' in response.json()['error']['message']


def test_messages_rejects_image_data_without_input_ids():
    client = _make_client()
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [],
            'image_data': 'https://example.com/img.png',
        },
    )
    assert response.status_code == 400
    assert 'input_ids' in response.json()['error']['message']


def test_messages_rejects_empty_input_ids():
    client = _make_client()
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [],
            'input_ids': [],
        },
    )
    assert response.status_code == 400
    assert 'input_ids' in response.json()['error']['message']


def test_messages_input_ids_non_stream():
    client = _make_client()
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [],
            'input_ids': [1, 2, 3],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data['type'] == 'message'
    assert data['content'][0]['type'] == 'text'
    assert data['content'][0]['text'] == 'Hello world!'


def test_messages_input_ids_streaming():
    client = _make_client()
    with client.stream(
            'POST',
            '/v1/messages',
            headers={'anthropic-version': '2023-06-01'},
            json={
                'model': 'fake-model',
                'max_tokens': 16,
                'stream': True,
                'messages': [],
                'input_ids': [1, 2, 3],
            },
    ) as response:
        body = '\n'.join(response.iter_lines())

    assert response.status_code == 200
    assert 'event: message_start' in body
    assert 'event: content_block_delta' in body
    assert 'event: message_stop' in body


def test_messages_image_data_preserves_input_ids_in_multimodal_content():
    context = _FakeServerContext()
    client = _make_client(server_context=context)
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [],
            'input_ids': [1, 2, 3],
            'image_data': 'https://example.com/img.png',
        },
    )
    assert response.status_code == 200
    args, kwargs = context.async_engine.generate_calls[-1]
    messages_arg = args[0]
    assert messages_arg[0]['content'][0] == {'type': 'text', 'text': [1, 2, 3]}
    assert kwargs['input_ids'] is None


def test_messages_rejects_system_with_input_ids():
    client = _make_client()
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [],
            'input_ids': [1, 2, 3],
            'system': 'ignored system prompt',
        },
    )
    assert response.status_code == 400
    assert 'system' in response.json()['error']['message']


def test_messages_rejects_tools_with_input_ids():
    client = _make_client(response_parser_cls=_ToolAndReasoningParser)
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [],
            'input_ids': [1, 2, 3],
            'tools': [{
                'name': 'search',
                'description': 'demo',
                'input_schema': {
                    'type': 'object',
                    'properties': {},
                },
            }],
            'tool_choice': {
                'type': 'auto',
            },
        },
    )
    assert response.status_code == 400
    assert 'tools' in response.json()['error']['message']


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
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'return_logprob': True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    # output_token_logprobs should be [(logprob, token_id), ...]
    assert data['output_token_logprobs'] == [[-0.5, 101], [-0.3, 102]]


def test_messages_non_stream_logprobs_default_off():
    client = _make_client()
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get('output_token_logprobs') is None


def test_messages_rejects_logprobs_when_engine_logprobs_mode_disabled():
    client = _make_client(logprobs_mode=None)
    response = client.post(
        '/v1/messages',
        headers={'anthropic-version': '2023-06-01'},
        json={
            'model': 'fake-model',
            'max_tokens': 16,
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'return_logprob': True,
        },
    )
    assert response.status_code == 400
    assert 'return_logprob' in response.json()['error']['message']


def test_messages_streaming_includes_logprobs():
    client = _make_client()
    with client.stream(
            'POST',
            '/v1/messages',
            headers={'anthropic-version': '2023-06-01'},
            json={
                'model': 'fake-model',
                'max_tokens': 16,
                'stream': True,
                'messages': [{'role': 'user', 'content': 'Hi there'}],
                'return_logprob': True,
            },
    ) as response:
        body = '\n'.join(response.iter_lines())

    assert response.status_code == 200
    assert 'output_token_logprobs' in body
