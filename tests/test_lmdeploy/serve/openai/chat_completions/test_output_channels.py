# Copyright (c) OpenMMLab. All rights reserved.
"""HTTP/SSE coverage of answer channels with the real GLM template/parser."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lmdeploy.serve.anthropic import create_anthropic_router
from lmdeploy.serve.openai.chat_completions import register
from lmdeploy.serve.openai.responses import create_responses_router
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.response_parser import BaseResponseParser
from lmdeploy.serve.parsers.tool_parser import ToolParserManager


@pytest.mark.parametrize('api,mode', [
    ('chat/completions', 'json'), ('responses', 'json'),
    ('chat/completions', 'prefill'), ('responses', 'prefill'), ('messages', 'prefill'),
])
@pytest.mark.parametrize('stream', [False, True])
def test_answer_channels(chat_endpoint, hf_media_template, api, mode, stream):
    _, context = chat_endpoint
    template = hf_media_template()
    context.async_engine.chat_template = template

    class Parser(BaseResponseParser):
        reasoning_parser_cls = ReasoningParserManager.get('default')
        tool_parser_cls = ToolParserManager.get('glm47')
        tokenizer = SimpleNamespace(get_vocab=lambda: {'<think>': 1, '</think>': 2})

    context.response_parser_cls = Parser
    original_preprocess = context.async_engine.preprocess
    original_generate = context.async_engine.generate
    expected = '{"species":"rabbit","count":12}' if mode == 'json' else ' Paris'
    prefixes = []

    async def preprocess(messages, session, **kwargs):
        prefixes.append(template.messages2prompt(messages, **(kwargs['chat_template_kwargs'] or {})))
        return await original_preprocess(messages, session, **kwargs)

    def generate(request, **kwargs):
        async def results():
            async for result in original_generate(request, **kwargs):
                result.response = expected
                result.token_ids = [100]
                yield result
        return results()

    context.async_engine.preprocess = preprocess
    context.async_engine.generate = generate
    messages = [{'role': 'user', 'content': 'Return JSON' if mode == 'json' else 'What is the capital of France?'}]
    if mode == 'prefill':
        messages.append({'role': 'assistant', 'content': 'The capital of France is'})
    payload = dict(model='fake-model', stream=stream)
    if api == 'responses':
        payload.update(input=messages, max_output_tokens=64)
        if mode == 'json':
            payload['text'] = {'format': {'type': 'json_object'}}
    else:
        payload.update(messages=messages, max_tokens=64)
        if mode == 'json':
            payload.update(response_format={'type': 'json_object'}, reasoning_effort='high')
    app = FastAPI()
    register(app, context)
    app.include_router(create_responses_router(context))
    app.include_router(create_anthropic_router(context))
    with TestClient(app) as client:
        response = client.post('/v1/' + api, json=payload, headers={'anthropic-version': '2023-06-01'})
    assert response.status_code == 200
    if mode == 'json':
        assert prefixes[0].endswith('<think></think>')
    else:
        assert prefixes[0].endswith('The capital of France is')
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        if api == 'chat/completions':
            text = ''.join(c['delta'].get('content', '') for e in events for c in e.get('choices', []))
            assert not any(c['delta'].get('reasoning_content') for e in events for c in e.get('choices', []))
        elif api == 'responses':
            assert events[-1]['type'] == 'response.completed'
            text = ''.join(e['delta'] for e in events if e['type'] == 'response.output_text.delta')
        else:
            assert events[-1]['type'] == 'message_stop'
            text = ''.join(e.get('delta', {}).get('text', '') for e in events)
            assert not any(e.get('delta', {}).get('thinking') for e in events)
    else:
        data = response.json()
        if api == 'chat/completions':
            text = data['choices'][0]['message']['content']
            assert not data['choices'][0]['message'].get('reasoning_content')
        elif api == 'responses':
            text = data['output_text']
            assert data['usage']['output_tokens_details']['reasoning_tokens'] == 0
        else:
            assert [b['type'] for b in data['content']] == ['text']
            text = ''.join(b['text'] for b in data['content'])
            assert data['usage']['output_tokens_details']['thinking_tokens'] == 0
    assert text == expected
    assert context.session_manager.sessions == {}
