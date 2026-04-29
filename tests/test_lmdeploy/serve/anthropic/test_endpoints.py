from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lmdeploy.serve.anthropic.endpoints import messages, messages_count_tokens, models
from lmdeploy.serve.anthropic.router import create_anthropic_router
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

    def __init__(self):
        self.model_name = 'fake-model'
        self.backend_config = SimpleNamespace(adapters=['adapter-model'])
        self.tokenizer = _FakeTokenizer()
        self.chat_template = _FakeChatTemplate()

    def generate(self, *args, **kwargs):
        async def _gen():
            yield SimpleNamespace(
                response='Hello ',
                token_ids=[101],
                input_token_len=8,
                generate_token_len=1,
                finish_reason=None,
            )
            yield SimpleNamespace(
                response='world!',
                token_ids=[102],
                input_token_len=8,
                generate_token_len=2,
                finish_reason='stop',
            )

        return _gen()


class _FakeServerContext:
    def __init__(self, *, response_parser_cls=None):
        self.async_engine = _FakeEngine()
        self.response_parser_cls = response_parser_cls

    def get_session(self, _session_id: int):
        return _FakeSession()


class _ToolAndReasoningParser:
    tool_parser_cls = object

    def __init__(self, request, tokenizer):
        self.request = request

    def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
        if delta_text.startswith('Hello'):
            return DeltaMessage(role='assistant', reasoning_content='internal reasoning', content='visible text'), False
        if delta_text.startswith('world'):
            return DeltaMessage(
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
            ), True
        return None, False

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


def _make_client(response_parser_cls=None) -> TestClient:
    app = FastAPI()
    app.include_router(create_anthropic_router(_FakeServerContext(response_parser_cls=response_parser_cls)))
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


def test_anthropic_model_listing():
    client = _make_client()
    response = client.get('/anthropic/v1/models')
    assert response.status_code == 200
    data = response.json()
    assert data['has_more'] is False
    assert [item['id'] for item in data['data']] == ['fake-model', 'adapter-model']
