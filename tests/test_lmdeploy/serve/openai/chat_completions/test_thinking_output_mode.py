# Copyright (c) OpenMMLab. All rights reserved.
"""Unsupported thinking controls fail before streaming or generation starts."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lmdeploy.serve.openai.chat_completions import register


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('options', [
    {'thinking_output_mode': 'not_a_valid_mode'},
    {'extra_body': {'thinking_output_mode': 'not_a_valid_mode'}},
    {'thinking_output_mode': None},
    {'extra_body': {'thinking_output_mode': None}},
    {'thinking_output_mode': 'any_value', 'extra_body': {'enable_thinking': False}},
    {'chat_template_kwargs': {'enable_thinking': False}, 'extra_body': {'thinking_output_mode': 'any_value'}},
])
def test_unsupported_thinking_output_mode_returns_400(chat_endpoint, stream, options):
    _, context = chat_endpoint
    app = FastAPI()
    register(app, context)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=dict(
            model='fake-model', messages=[{'role': 'user', 'content': 'hi'}], stream=stream, **options))

    assert response.status_code == 400
    assert response.headers['content-type'] == 'application/json'
    error = response.json()
    assert error['object'] == 'error'
    assert 'thinking_output_mode is not supported' in error['message']
    assert 'chat_template_kwargs.enable_thinking' in error['message']
    assert context.async_engine.gen_configs == []
    assert context.async_engine.call_count == 0
    assert context.session_manager.sessions == {}


@pytest.mark.parametrize('options', [
    {'enable_thinking': False},
    {'extra_body': {'enable_thinking': False}},
    {'chat_template_kwargs': {'enable_thinking': False}},
    {'unrelated_extension': 'value'},
])
def test_supported_controls_and_other_extensions_still_work(chat_endpoint, options):
    _, context = chat_endpoint
    app = FastAPI()
    register(app, context)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=dict(
            model='fake-model', messages=[{'role': 'user', 'content': 'hi'}], **options))

    assert response.status_code == 200
    assert response.json()['choices'][0]['message']['content'] == 'choice-1'
    assert context.async_engine.call_count == 1
