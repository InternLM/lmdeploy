# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import json
from types import SimpleNamespace

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage


class _FakeRawRequest:

    async def json(self):
        return {}

    async def is_disconnected(self):
        return False


class _FakeSession:

    def __init__(self, session_id=1):
        self.session_id = session_id
        self.epoch = 0
        self.aborted = False

    async def async_abort(self):
        self.aborted = True


class _FakeSessionManager:

    def has(self, session_id):
        return False

    def get(self, *args, **kwargs):
        return _FakeSession()


class _FakeEngine:

    def __init__(self, outputs):
        self.outputs = outputs
        self.epoch = 0
        self.model_name = 'fake-model'
        self.backend_config = SimpleNamespace(
            adapters=None,
            enable_return_routed_experts=False,
            logprobs_mode=None,
        )
        self.session_mgr = _FakeSessionManager()
        self.tokenizer = SimpleNamespace(model=SimpleNamespace(model=object()))

    def generate(self, *args, **kwargs):
        async def _gen():
            for item in self.outputs:
                yield SimpleNamespace(
                    response=item['text'],
                    token_ids=item['token_ids'],
                    input_token_len=3,
                    generate_token_len=item['generate_token_len'],
                    finish_reason=item['finish_reason'],
                    logprobs=None,
                    cache_block_ids=None,
                    routed_experts=None,
                )

        return _gen()


class _EchoParser:
    tool_parser_cls = None

    def __init__(self, request, tokenizer):
        self.request = request
        self.tool_parser = None
        self.accumulated_text = ''

    def stream_chunk(self, delta_text, delta_token_ids, **kwargs):
        self.accumulated_text += delta_text
        return [(DeltaMessage(role='assistant', content=delta_text), False)]

    def parse_complete(self, text, token_ids=None, **kwargs):
        return text, None, None

    def validate_complete(self, text=None):
        return True


class _InvalidResultParser(_EchoParser):

    def validate_complete(self, text=None):
        return False


async def _collect_chat_stream(parser_cls, outputs, **request_kwargs):
    from lmdeploy.serve.openai import api_server

    old_engine = api_server.VariableInterface.async_engine
    old_parser_cls = api_server.VariableInterface.response_parser_cls
    try:
        api_server.VariableInterface.async_engine = _FakeEngine(outputs)
        api_server.VariableInterface.response_parser_cls = parser_cls
        request = ChatCompletionRequest(
            model='fake-model',
            messages=[{
                'role': 'user',
                'content': 'hello',
            }],
            stream=True,
            max_completion_tokens=8,
            **request_kwargs,
        )
        response = await api_server.chat_completions_v1(request, _FakeRawRequest())
        return await _collect_sse_payloads(response)
    finally:
        api_server.VariableInterface.async_engine = old_engine
        api_server.VariableInterface.response_parser_cls = old_parser_cls


async def _collect_sse_payloads(response):
    chunks = []
    async for raw_chunk in response.body_iterator:
        if isinstance(raw_chunk, bytes):
            raw_chunk = raw_chunk.decode('utf-8')
        for line in raw_chunk.splitlines():
            if not line.startswith('data: '):
                continue
            data = line.removeprefix('data: ')
            if data == '[DONE]':
                continue
            chunks.append(json.loads(data))
    return chunks


def test_chat_stream_skips_parser_validation_without_return_fields():
    chunks = asyncio.run(
        _collect_chat_stream(
            _InvalidResultParser,
            [
                dict(text='<tool_call>', token_ids=[1], generate_token_len=1, finish_reason=None),
                dict(text='bad</tool_call>', token_ids=[2], generate_token_len=2, finish_reason='stop'),
            ],
        ))

    assert chunks[-1]['choices'][0]['finish_reason'] == 'stop'


def test_chat_stream_uses_parser_validation_finish_reason_with_return_token_ids():
    chunks = asyncio.run(
        _collect_chat_stream(
            _InvalidResultParser,
            [
                dict(text='<tool_call>', token_ids=[1], generate_token_len=1, finish_reason=None),
                dict(text='bad</tool_call>', token_ids=[2], generate_token_len=2, finish_reason='stop'),
            ],
            return_token_ids=True,
        ))

    assert chunks[-1]['choices'][0]['finish_reason'] == 'parse_error'


def test_chat_non_stream_skips_parser_validation_without_return_fields():
    from lmdeploy.serve.openai import api_server

    old_engine = api_server.VariableInterface.async_engine
    old_parser_cls = api_server.VariableInterface.response_parser_cls
    try:
        api_server.VariableInterface.async_engine = _FakeEngine([
            dict(text='invalid tool', token_ids=[1], generate_token_len=1, finish_reason='stop'),
        ])
        api_server.VariableInterface.response_parser_cls = _InvalidResultParser
        request = ChatCompletionRequest(
            model='fake-model',
            messages=[{
                'role': 'user',
                'content': 'hello',
            }],
            stream=False,
            max_completion_tokens=8,
        )
        response = asyncio.run(api_server.chat_completions_v1(request, _FakeRawRequest()))
        data = response if isinstance(response, dict) else json.loads(response.body)
    finally:
        api_server.VariableInterface.async_engine = old_engine
        api_server.VariableInterface.response_parser_cls = old_parser_cls

    assert data['choices'][0]['finish_reason'] == 'stop'


def test_chat_non_stream_uses_parser_validation_finish_reason_with_return_token_ids():
    from lmdeploy.serve.openai import api_server

    old_engine = api_server.VariableInterface.async_engine
    old_parser_cls = api_server.VariableInterface.response_parser_cls
    try:
        api_server.VariableInterface.async_engine = _FakeEngine([
            dict(text='invalid tool', token_ids=[1], generate_token_len=1, finish_reason='stop'),
        ])
        api_server.VariableInterface.response_parser_cls = _InvalidResultParser
        request = ChatCompletionRequest(
            model='fake-model',
            messages=[{
                'role': 'user',
                'content': 'hello',
            }],
            stream=False,
            max_completion_tokens=8,
            return_token_ids=True,
        )
        response = asyncio.run(api_server.chat_completions_v1(request, _FakeRawRequest()))
        data = response if isinstance(response, dict) else json.loads(response.body)
    finally:
        api_server.VariableInterface.async_engine = old_engine
        api_server.VariableInterface.response_parser_cls = old_parser_cls

    assert data['choices'][0]['finish_reason'] == 'parse_error'
