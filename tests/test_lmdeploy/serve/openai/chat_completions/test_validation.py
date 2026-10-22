# Copyright (c) OpenMMLab. All rights reserved.

import sys
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from lmdeploy.serve.core.async_engine import AsyncEngine
from lmdeploy.serve.managers import SessionManager
from lmdeploy.serve.openai.chat_completions import register
from lmdeploy.serve.openai.chat_completions.validation import check_request
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.processors.multimodal import MultimodalProcessor


class _SessionManager:

    @staticmethod
    def has(session_id):
        return False


def _server_context():
    return SimpleNamespace(
        engine_config=SimpleNamespace(
            logprobs_mode='raw_logprobs',
            enable_return_routed_experts=False,
        ),
        session_manager=_SessionManager(),
        response_parser_cls=SimpleNamespace(tool_parser_cls=object()),
    )


def _request(**kwargs):
    request_kwargs = dict(
        model='fake-model',
        messages=[{'role': 'user', 'content': 'hi'}],
    )
    request_kwargs.update(kwargs)
    return ChatCompletionRequest(**request_kwargs)


@pytest.mark.parametrize(
    ('request_kwargs', 'error_fragment'),
    [
        pytest.param({'messages': []}, 'messages', id='empty-messages'),
        pytest.param({'messages': [], 'input_ids': []}, 'input_ids', id='empty-input-ids'),
        pytest.param({'min_p': -0.1}, 'min_p', id='min-p-below-range'),
        pytest.param({'min_p': 1.1}, 'min_p', id='min-p-above-range'),
        pytest.param({'max_completion_tokens': 0}, 'max_completion_tokens', id='zero-max-completion-tokens'),
        pytest.param({'max_tokens': 0}, 'max_tokens', id='zero-max-tokens'),
        pytest.param({'min_new_tokens': -1}, 'min_new_tokens', id='negative-min-new-tokens'),
        pytest.param({'logprobs': True, 'top_logprobs': 21}, 'top_logprobs', id='top-logprobs-above-range'),
    ],
)
def test_chat_completions_rejects_invalid_request(request_kwargs, error_fragment):
    error = check_request(_request(**request_kwargs), _server_context())

    assert error_fragment in error


def test_chat_completions_accepts_raw_input_ids_without_messages():
    error = check_request(
        _request(messages=[], input_ids=[1, 2, 3]),
        _server_context(),
    )

    assert error == ''


def test_chat_completions_accepts_null_n_as_unspecified():
    assert check_request(_request(n=None), _server_context()) == ''


@pytest.fixture
def preprocessing_client(chat_endpoint, monkeypatch):
    """Exercise real preprocessing and error handling without loading a
    model."""
    _, context = chat_endpoint
    engine = context.async_engine
    engine.session_mgr = SessionManager()
    engine.session_len = 128
    engine.backend_config.enable_prefix_caching = False
    engine.request_logger = SimpleNamespace(log_prompt=lambda *a, **kw: None, log_inputs=lambda *a, **kw: None)
    engine._determine_gen_config = lambda input_ids, gen_config: gen_config
    engine.prompt_processor = MultimodalProcessor(
        tokenizer=SimpleNamespace(encode=lambda *a, **kw: [1]),
        chat_template=SimpleNamespace(messages2prompt=lambda messages, **kw: str(messages)),
        vl_encoder=object(),
    )
    engine.preprocess = MethodType(AsyncEngine.preprocess, engine)
    load_media = Mock(side_effect=AssertionError('Invalid content must be rejected before loading media.'))
    monkeypatch.setattr(sys.modules[MultimodalProcessor.__module__], 'load_from_url', load_media)
    app, router = FastAPI(), APIRouter()
    register(router, context)
    app.include_router(router)
    with TestClient(app) as client:
        yield client, engine
    load_media.assert_not_called()


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('messages, location', [
    pytest.param([dict(role='user', content=[dict(type='unsupported_probe')])],
                 'messages[0].content[0].type', id='unknown-only'),
    pytest.param([dict(role='assistant', content='History'),
                  dict(role='user', content=[dict(type='text', text='Keep this'), dict(type='unsupported_probe')])],
                 'messages[1].content[1].type', id='text-and-unknown'),
    pytest.param([dict(role='user', content=[dict(type='image_url', image_url=dict(url='https://example.com/a.png')),
                                           dict(type='unsupported_probe')])],
                 'messages[0].content[1].type', id='image-before-unknown'),
    pytest.param([dict(role='user', content=[dict(type='image_url', image_url=dict(url='https://example.com/a.png'))]),
                  dict(role='user', content=[dict(type='unsupported_probe')])],
                 'messages[1].content[0].type', id='image-in-earlier-message'),
    pytest.param([dict(role='user', content=['not a content object'])],
                 'messages[0].content[0]', id='non-object-block'),
    pytest.param([dict(role='user', content=[dict(text='Missing type')])],
                 'messages[0].content[0].type', id='missing-type'),
])
def test_http_rejects_invalid_content_before_generation(preprocessing_client, messages, location, stream):
    client, engine = preprocessing_client
    response = client.post('/v1/chat/completions',
                           json=dict(model='fake-model', messages=messages, stream=stream, max_completion_tokens=16))

    assert response.status_code == 400, response.text
    assert response.headers['content-type'].startswith('application/json')
    error = response.json()
    assert error['type'] == 'invalid_request_error'
    assert location in error['message']
    assert engine.call_count == 0
    assert engine.session_mgr.sessions == {}


@pytest.mark.parametrize('stream', [False, True])
def test_http_text_blocks_still_generate(preprocessing_client, stream):
    client, engine = preprocessing_client
    messages = [dict(role='user', content=[dict(type='text', text='First'), dict(type='text', text='Second')])]
    response = client.post('/v1/chat/completions',
                           json=dict(model='fake-model', messages=messages, stream=stream, max_completion_tokens=16))

    assert response.status_code == 200, response.text
    if stream:
        assert 'data: [DONE]' in response.text
    else:
        assert response.json()['choices'][0]['message']['content'] == 'choice-1'
    assert engine.call_count == 1
    assert engine.session_mgr.sessions == {}
