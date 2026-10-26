# Copyright (c) OpenMMLab. All rights reserved.

import json
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.pytorch.engine import Engine
from lmdeploy.pytorch.engine.mp_engine.base import MPEngine
from lmdeploy.pytorch.engine.mp_engine.base_worker import EngineWorkerBase
from lmdeploy.serve.core.async_engine import AsyncEngine
from lmdeploy.serve.managers import SessionManager
from lmdeploy.serve.openai.chat_completions import register
from lmdeploy.serve.openai.chat_completions.logprobs import _create_chat_completion_logprobs
from lmdeploy.serve.openai.chat_completions.validation import check_request
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.processors.multimodal import MultimodalProcessor


class _SessionManager:

    @staticmethod
    def has(session_id):
        return False


def _server_context(vocab_size=154880):
    return SimpleNamespace(
        engine_config=PytorchEngineConfig(logprobs_mode='raw_logprobs'),
        async_engine=SimpleNamespace(engine=SimpleNamespace(vocab_size=vocab_size)),
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
        pytest.param({'logprobs': True, 'top_logprobs': 154881}, 'top_logprobs', id='top-logprobs-above-vocab'),
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


@pytest.mark.parametrize('config, limit', [
    (PytorchEngineConfig(logprobs_mode='raw_logprobs'), 154880),
    (PytorchEngineConfig(logprobs_mode='raw_logits'), 16),
    (TurbomindEngineConfig(), 20),
])
def test_chat_logprobs_backend_limits(config, limit):
    context = _server_context(limit)
    context.engine_config = config
    for count in (None, 0, min(21, limit), limit):
        assert check_request(_request(logprobs=True, top_logprobs=count), context) == ''
    for count in (-1, limit + 1):
        assert 'top_logprobs' in check_request(_request(logprobs=True, top_logprobs=count), context)


@pytest.mark.parametrize('use_mp_engine', [False, True])
def test_chat_logprobs_uses_resolved_model_vocab(use_mp_engine):
    vocab_size = 154880
    context = _server_context(vocab_size)
    engine = Engine.__new__(Engine)
    engine.executor = SimpleNamespace(model_config=SimpleNamespace(vocab_size=vocab_size))
    if use_mp_engine:
        worker = EngineWorkerBase.__new__(EngineWorkerBase)
        worker.engine = engine
        engine = MPEngine.__new__(MPEngine)
        engine._collective_rpc = Mock(side_effect=lambda method: getattr(worker, method)())
    context.async_engine.engine = engine
    # Tokenizer and raw HF metadata need not match the resolved LM head.
    context.async_engine.tokenizer = SimpleNamespace(vocab_size=8)
    context.async_engine.hf_cfg = SimpleNamespace(vocab_size=8)
    assert check_request(_request(logprobs=True, top_logprobs=vocab_size), context) == ''
    error = check_request(_request(logprobs=True, top_logprobs=vocab_size + 1), context)
    assert f'[0, {vocab_size}]' in error
    if use_mp_engine:
        engine._collective_rpc.assert_called_once_with('get_vocab_size')


def test_chat_logprobs_still_requires_engine_mode_and_request_flag():
    context = _server_context()
    assert 'Invalid logprobs' in check_request(_request(top_logprobs=21), context)
    context.engine_config.logprobs_mode = None
    assert 'not enabled logprobs_mode' in check_request(_request(logprobs=True, top_logprobs=21), context)


@pytest.mark.parametrize('selected_id', [0, 2])
def test_chat_logprobs_preserves_ids_without_tokenizer_entries(selected_id):
    tokenizer = SimpleNamespace(convert_ids_to_tokens=lambda token_id: {0: 'text', 1: b'\xff'}.get(token_id))
    scores = {0: -0.5, 1: -1.5, 2: -2.5}
    row = _create_chat_completion_logprobs(tokenizer, [selected_id], [scores]).content[0]
    entries = {entry.token: entry for entry in [row, *row.top_logprobs]}
    assert entries['token_id:2'].bytes is None
    assert entries['token_id:2'].logprob == scores[2]
    assert entries['text'].bytes == list(b'text')
    assert entries['\\xff'].bytes == [255]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('top_logprobs', [64, 65])
def test_http_chat_logprobs_vocab_boundary(chat_endpoint, stream, top_logprobs):
    _, context = chat_endpoint
    engine = context.async_engine
    engine.backend_config = PytorchEngineConfig(logprobs_mode='raw_logprobs')
    engine.engine = SimpleNamespace(vocab_size=64)
    engine.tokenizer.model.model = SimpleNamespace(
        convert_ids_to_tokens=lambda token_id: f'token-{token_id}' if token_id < 60 else None)

    original_generate = engine.generate

    async def generate(*args, **kwargs):
        async for result in original_generate(*args, **kwargs):
            result.logprobs = [{i: -float(i) for i in range(top_logprobs)}]
            yield result

    engine.generate = generate
    app, router = FastAPI(), APIRouter()
    register(router, context)
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=dict(
            model='fake-model', messages=[dict(role='user', content='hi')],
            logprobs=True, top_logprobs=top_logprobs, max_completion_tokens=1, stream=stream,
        ))
    if top_logprobs > 64:
        assert response.status_code == 400, response.text
        assert '[0, 64]' in response.json()['message']
        assert engine.call_count == 0
        assert engine.gen_configs == []
    else:
        assert response.status_code == 200, response.text
        if stream:
            assert response.text.rstrip().endswith('data: [DONE]')
            payloads = [json.loads(line[6:]) for line in response.text.splitlines()
                        if line.startswith('data: ') and line != 'data: [DONE]']
        else:
            payloads = [response.json()]
        rows = [row for payload in payloads for choice in payload['choices']
                if choice.get('logprobs') for row in choice['logprobs']['content']]
        assert len(rows) == 1
        # The selected token may be separate from the alternatives in Chat.
        tokens = {rows[0]['token']} | {item['token'] for item in rows[0]['top_logprobs']}
        assert tokens == {f'token-{i}' if i < 60 else f'token_id:{i}' for i in range(top_logprobs)}
        assert engine.gen_configs[0].logprobs == top_logprobs
        assert engine.call_count == 1
    assert engine.session_mgr.sessions == {}


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
        chat_template=SimpleNamespace(messages2prompt=lambda messages, **kw: str(messages),
                                      messages2input_ids=lambda *args, **kwargs: None),
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
