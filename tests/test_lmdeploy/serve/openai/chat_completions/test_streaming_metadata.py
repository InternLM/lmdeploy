# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import asyncio
import json
from functools import partial
from types import SimpleNamespace

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage


class _FakeTokenizer:

    vocab_size = 1000

    def convert_ids_to_tokens(self, token_id):
        return f'tok{token_id}'


class _FakeSession:

    def __init__(self, session_id):
        self.session_id = session_id
        self.epoch = None

    async def async_abort(self):
        pass


class _FakeSessionManager:

    def __init__(self):
        self.sessions = []
        self.removed = []

    def get(self, session_id=None, create_if_not_exists=True):
        session = _FakeSession(session_id if session_id is not None else len(self.sessions) + 1)
        self.sessions.append(session)
        return session

    def has(self, session_id):
        return False

    def map_user_session_id(self, session_id):
        return session_id

    def remove(self, session):
        self.removed.append(session)


class _FakeAsyncEngine:

    model_name = 'fake-model'
    epoch = 0

    def __init__(self, outputs):
        self.outputs = outputs
        self.backend_config = SimpleNamespace(
            adapters=[],
            logprobs_mode='raw',
            enable_return_routed_experts=False,
        )
        self.session_mgr = _FakeSessionManager()
        self.tokenizer = SimpleNamespace(model=SimpleNamespace(model=_FakeTokenizer()))

    async def preprocess(self, prompt, session, **kwargs):
        return SimpleNamespace(prompt=prompt, session=session)

    def generate(self, *args, **kwargs):

        async def _generator():
            for output in self.outputs:
                yield SimpleNamespace(
                    response=output.get('response', ''),
                    token_ids=output.get('token_ids'),
                    logprobs=output.get('logprobs'),
                    input_token_len=1,
                    generate_token_len=1,
                    cached_tokens=0,
                    finish_reason=output.get('finish_reason'),
                    cache_block_ids=output.get('cache_block_ids'),
                    routed_experts=None,
                )

        return _generator()


class _FakeRawRequest:

    async def json(self):
        return {}

    async def is_disconnected(self):
        return False


class _StreamingMetadataParser:

    tool_parser_cls = None
    seen_token_ids = []

    def __init__(self, request):
        self.request = request
        self.tool_parser = None
        self.reasoning_tokens = 0

    def stream_chunk(self, delta_text, delta_token_ids, **kwargs):
        self.seen_token_ids.append(delta_token_ids)
        if delta_text in ('<hidden>', '<empty>'):
            return []
        if delta_text:
            return [(DeltaMessage(content=delta_text), False)]
        return []

    def parse_complete(self, text, token_ids=None, **kwargs):
        return text, None, None

    def validate_complete(self, text=None):
        return True


@pytest.fixture
def install_fake_chat_server(chat_endpoint):
    endpoint, context = chat_endpoint

    def _install(outputs):
        _StreamingMetadataParser.seen_token_ids.clear()
        engine = _FakeAsyncEngine(outputs)
        context.async_engine = engine
        context.response_parser_cls = _StreamingMetadataParser
        return partial(_chat_stream_payloads, endpoint)

    return _install


def _chat_stream_payloads(endpoint, *, logprobs=True, return_logprob=True, return_token_ids=True):
    request = ChatCompletionRequest(
        model='fake-model',
        messages=[{
            'role': 'user',
            'content': 'hi',
        }],
        stream=True,
        logprobs=logprobs,
        return_logprob=return_logprob,
        return_token_ids=return_token_ids,
    )

    async def _collect():
        response = await endpoint(request, _FakeRawRequest())
        events = [event async for event in response.body_iterator]
        payloads = []
        for event in events:
            if isinstance(event, bytes):
                event = event.decode()
            for line in event.splitlines():
                if not line.startswith('data: '):
                    continue
                data = line.removeprefix('data: ')
                if data != '[DONE]':
                    payloads.append(json.loads(data))
        return payloads

    return asyncio.run(_collect())


def _choice(payload):
    return payload['choices'][0]


def test_empty_parser_result_emits_requested_token_metadata_immediately(install_fake_chat_server):
    chat_stream = install_fake_chat_server([
        {
            'response': '<empty>',
            'token_ids': [101],
            'logprobs': [{
                101: -0.1,
            }],
            'finish_reason': None,
        },
        {
            'response': 'visible',
            'token_ids': [102],
            'logprobs': [{
                102: -0.2,
            }],
            'finish_reason': None,
        },
    ])

    payloads = chat_stream()

    assert len(payloads) == 2
    first_choice = _choice(payloads[0])
    assert first_choice['delta']['content'] == ''
    assert first_choice['output_ids'] == [101]
    assert first_choice['output_token_logprobs'] == [[-0.1, 101]]
    assert [item['token'] for item in first_choice['logprobs']['content']] == ['tok101']

    second_choice = _choice(payloads[1])
    assert second_choice['delta']['content'] == 'visible'
    assert second_choice['output_ids'] == [102]
    assert second_choice['output_token_logprobs'] == [[-0.2, 102]]
    assert [item['token'] for item in second_choice['logprobs']['content']] == ['tok102']


def test_empty_parser_result_keeps_token_ids_and_logprobs_aligned(install_fake_chat_server):
    chat_stream = install_fake_chat_server([
        {
            'response': '<hidden>',
            'token_ids': [101, 102],
            'logprobs': [{
                101: -0.1,
            }, {
                102: -0.2,
            }],
            'finish_reason': None,
        },
        {
            'response': 'visible',
            'token_ids': [103],
            'logprobs': [{
                103: -0.3,
            }],
            'finish_reason': None,
        },
    ])

    payloads = chat_stream()

    assert len(payloads) == 2
    first_choice = _choice(payloads[0])
    assert first_choice['delta']['content'] == ''
    assert first_choice['output_ids'] == [101, 102]
    assert first_choice['output_token_logprobs'] == [[-0.1, 101], [-0.2, 102]]
    assert [item['token'] for item in first_choice['logprobs']['content']] == ['tok101', 'tok102']

    second_choice = _choice(payloads[1])
    assert second_choice['delta']['content'] == 'visible'
    assert second_choice['output_ids'] == [103]
    assert second_choice['output_token_logprobs'] == [[-0.3, 103]]
    assert [item['token'] for item in second_choice['logprobs']['content']] == ['tok103']


def test_terminal_empty_parser_result_emits_finish_reason(install_fake_chat_server):
    chat_stream = install_fake_chat_server([
        {
            'response': '<empty>',
            'token_ids': [101],
            'logprobs': [{
                101: -0.1,
            }],
            'finish_reason': 'stop',
        },
    ])

    payloads = chat_stream()

    assert len(payloads) == 1
    choice = _choice(payloads[0])
    assert choice['delta'] == {'content': '', 'role': 'assistant'}
    assert choice['finish_reason'] == 'stop'
    assert choice['output_ids'] == [101]
    assert choice['output_token_logprobs'] == [[-0.1, 101]]
    assert [item['token'] for item in choice['logprobs']['content']] == ['tok101']


def test_empty_parser_result_without_requested_metadata_is_skipped(install_fake_chat_server):
    chat_stream = install_fake_chat_server([
        {
            'response': '<hidden>',
            'token_ids': [101, 102],
            'logprobs': [{
                101: -0.1,
            }, {
                102: -0.2,
            }],
            'finish_reason': None,
        },
        {
            'response': 'visible',
            'token_ids': [103],
            'logprobs': [{
                103: -0.3,
            }],
            'finish_reason': None,
        },
    ])

    payloads = chat_stream(logprobs=False, return_logprob=False, return_token_ids=False)

    assert _StreamingMetadataParser.seen_token_ids == [[101, 102], [103]]
    choice = _choice(payloads[0])
    assert 'output_ids' not in choice
    assert 'output_token_logprobs' not in choice
    assert 'logprobs' not in choice


@pytest.mark.parametrize(
    ('response', 'finish_reason'),
    [('visible', None), ('<empty>', None), ('<empty>', 'stop')],
)
def test_cache_remote_token_ids_use_current_engine_result(install_fake_chat_server, response, finish_reason):
    chat_stream = install_fake_chat_server([
        {'response': '<hidden>', 'token_ids': [101]},
        {'response': response, 'token_ids': [102], 'finish_reason': finish_reason, 'cache_block_ids': [7]},
    ])

    payloads = chat_stream(logprobs=False, return_logprob=False, return_token_ids=False)

    assert len(payloads) == 1
    assert 'output_ids' not in _choice(payloads[0])
    assert payloads[0]['cache_block_ids'] == [7]
    assert payloads[0]['remote_token_ids'] == [102]
