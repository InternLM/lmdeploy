# Copyright (c) OpenMMLab. All rights reserved.
"""User-image HTTP regressions across Chat, Messages, and Responses.

Generation and the remote HTTP transport are simulated; protocol conversion, HTTP/SSE handling, media loading, and image
decoding use production code.
"""
import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlparse

import pytest
import requests
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from PIL import Image, UnidentifiedImageError

from lmdeploy.model import BaseChatTemplate
from lmdeploy.serve.anthropic.adapter import to_openai_messages
from lmdeploy.serve.anthropic.protocol import MessagesRequest
from lmdeploy.serve.anthropic.router import create_anthropic_router
from lmdeploy.serve.openai.chat_completions import register
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage
from lmdeploy.serve.openai.responses import ResponsesRequest, create_responses_router
from lmdeploy.serve.openai.responses.request import messages_from_input
from lmdeploy.serve.processors.multimodal import MultimodalProcessor

PATHS = {'chat': '/v1/chat/completions', 'messages': '/v1/messages', 'responses': '/v1/responses'}


class _Parser:
    tool_parser_cls = None
    tool_parser = None
    reasoning_tokens = None

    def __init__(self, request):
        self.request = request

    def stream_chunk(self, delta_text, delta_token_ids, **kwargs):
        return [(DeltaMessage(content=delta_text), False)] if delta_text else []

    def parse_complete(self, text, token_ids=None, **kwargs):
        return text, None, None

    def validate_complete(self, text=None):
        return True


class _ImageEngine:
    model_name = 'fake-model'
    backend_config = SimpleNamespace(adapters=[], logprobs_mode=None)
    chat_template = BaseChatTemplate()
    tokenizer = SimpleNamespace(model=SimpleNamespace(model=None))

    def __init__(self):
        self.decoded = None
        self.generate_calls = 0

    async def preprocess(self, prompt, session, **kwargs):
        assert kwargs['do_preprocess'] is True
        self.decoded = await MultimodalProcessor.async_parse_multimodal_item(prompt)
        return SimpleNamespace(session=session)

    def generate(self, request, **kwargs):
        self.generate_calls += 1

        async def generate():
            yield SimpleNamespace(response='ok', token_ids=[1], input_token_len=1, generate_token_len=1,
                                  finish_reason='stop', cached_tokens=0, logprobs=None, routed_experts=None,
                                  cache_block_ids=None)

        return generate()


@pytest.fixture
def image_endpoints():
    engine = _ImageEngine()
    sessions, removed = [], []
    manager = SimpleNamespace(has=lambda _: False, remove=removed.append)

    def create_session(_):
        session = SimpleNamespace(session_id=len(sessions), epoch=0, async_abort=AsyncMock())
        sessions.append(session)
        return session

    context = SimpleNamespace(async_engine=engine, engine_config=engine.backend_config, session_manager=manager,
                              response_parser_cls=_Parser, default_gen_config={}, create_session=create_session)
    app = FastAPI()
    chat_router = APIRouter()
    register(chat_router, context)
    app.include_router(chat_router)
    app.include_router(create_anthropic_router(context))
    app.include_router(create_responses_router(context))
    with TestClient(app) as client:
        yield client, engine, sessions, removed


@pytest.fixture(scope='module')
def image_sources():
    encoded = {}
    for image_format, mime in [('PNG', 'png'), ('JPEG', 'jpeg'), ('WEBP', 'webp')]:
        for color in ('red', 'blue'):
            buffer = BytesIO()
            Image.new('RGB', (32, 32), color).save(buffer, format=image_format)
            encoded[f'/{color}.{image_format}'] = (f'image/{mime}', buffer.getvalue())

    def fetch(_session, url, **kwargs):
        parsed = urlparse(url)
        # A public IP exercises the normal URL validation without external DNS.
        # No network request is made: only the requests transport is replaced.
        assert parsed.netloc == '93.184.216.34'
        mime, data = encoded[parsed.path]
        response = requests.Response()
        response.status_code = 200
        response.headers['Content-Type'] = mime
        response._content = data
        return response

    def source(color, image_format, source_type):
        path = f'/{color}.{image_format}'
        if source_type == 'http':
            return f'http://93.184.216.34{path}'
        mime, data = encoded[path]
        return f'data:{mime};base64,{base64.b64encode(data).decode()}'

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(requests.Session, 'get', fetch)
        yield source


def _history(scenario):
    red, blue = dict(type='image', color='red'), dict(type='image', color='blue')
    before, between, after = [dict(type='text', text=text) for text in ('before', 'between', 'after')]
    if scenario == 'image_only':
        return [dict(role='user', content=[red])]
    if scenario == 'single':
        return [dict(role='user', content=[before, red, after])]
    if scenario == 'multiple':
        return [dict(role='user', content=[before, red, between, blue, after])]
    return [dict(role='user', content=[before, red, after]),
            dict(role='assistant', content='First image received.'),
            dict(role='user', content=[before, blue, after])]


def _payload(protocol, history, source):
    messages = []
    for message in history:
        content = message['content']
        if isinstance(content, list):
            parts = []
            for part in content:
                if part['type'] == 'text':
                    parts.append(dict(type='input_text' if protocol == 'responses' else 'text', text=part['text']))
                    continue
                url = source(part['color'])
                if protocol == 'responses':
                    parts.append(dict(type='input_image', image_url=url))
                elif protocol == 'chat':
                    parts.append(dict(type='image_url', image_url=dict(url=url)))
                else:
                    if url.startswith('data:'):
                        prefix, data = url.split(',', 1)
                        image_source = dict(type='base64', media_type=prefix[5:].split(';')[0], data=data)
                    else:
                        image_source = dict(type='url', url=url)
                    parts.append(dict(type='image', source=image_source))
            content = parts
        messages.append(dict(role=message['role'], content=content))
    if protocol == 'responses':
        return dict(model='fake-model', input=messages, max_output_tokens=16)
    return dict(model='fake-model', messages=messages, max_tokens=16)


def _convert(protocol, payload):
    if protocol == 'chat':
        return ChatCompletionRequest(**payload).messages
    if protocol == 'messages':
        return to_openai_messages(MessagesRequest(**payload))
    return messages_from_input(ResponsesRequest(**payload))


@pytest.mark.parametrize('protocol', PATHS)
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('image_format', ['PNG', 'JPEG', 'WEBP'])
@pytest.mark.parametrize('source_type', ['data', 'http'])
@pytest.mark.parametrize('scenario', ['single', 'image_only', 'multiple', 'multiturn'])
def test_user_images_through_all_http_endpoints(image_endpoints, image_sources, protocol, stream,
                                               image_format, source_type, scenario):
    client, engine, sessions, removed = image_endpoints
    history = _history(scenario)
    payload = _payload(protocol, history, lambda color: image_sources(color, image_format, source_type))
    response = client.post(PATHS[protocol], json=dict(payload, stream=stream),
                           headers={'anthropic-version': '2023-06-01'})
    assert response.status_code == 200, response.text
    if stream:
        assert response.headers['content-type'].startswith('text/event-stream')
        assert {'chat': 'data: [DONE]', 'messages': 'event: message_stop',
                'responses': 'event: response.completed'}[protocol] in response.text
        assert 'event: error' not in response.text
        assert 'event: response.failed' not in response.text
    else:
        result = response.json()
        if protocol == 'chat':
            assert result['choices'][0]['message']['content'] == 'ok'
        elif protocol == 'messages':
            assert result['content'][0]['text'] == 'ok'
        else:
            assert result['output_text'] == 'ok'
            assert result['status'] == 'completed'
    assert engine.generate_calls == 1
    assert [m['role'] for m in engine.decoded] == [m['role'] for m in history]
    for actual, expected in zip(engine.decoded, history):
        if isinstance(expected['content'], str):
            assert actual['content'] == expected['content']
            continue
        assert [p['type'] for p in actual['content']] == [p['type'] for p in expected['content']]
        for actual_part, expected_part in zip(actual['content'], expected['content']):
            if expected_part['type'] == 'text':
                assert actual_part['text'] == expected_part['text']
            else:
                image = actual_part['data']
                assert image.size == (32, 32)
                target = (255, 0, 0) if expected_part['color'] == 'red' else (0, 0, 255)
                assert all(abs(a - b) < 10 for a, b in zip(image.getpixel((0, 0)), target))
    assert len(sessions) == 1
    assert removed == sessions


@pytest.mark.parametrize('protocol', PATHS)
def test_corrupt_images_are_not_silently_dropped(protocol):
    payload = _payload(protocol, _history('image_only'), lambda _: 'data:image/png;base64,bm90IGFuIGltYWdl')
    messages = _convert(protocol, payload)
    with pytest.raises(UnidentifiedImageError):
        asyncio.run(MultimodalProcessor.async_parse_multimodal_item(messages))


@pytest.mark.parametrize('protocol', PATHS)
def test_text_only_model_rejects_images_for_every_protocol(protocol):
    payload = _payload(protocol, _history('single'), lambda _: 'https://example.com/image.png')
    messages = _convert(protocol, payload)
    processor = MultimodalProcessor(None, BaseChatTemplate(), vl_encoder=None)
    with pytest.raises(ValueError, match='Multimodal input is not supported'):
        asyncio.run(processor.get_prompt_input(messages, do_preprocess=True))
