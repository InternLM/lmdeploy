# Copyright (c) OpenMMLab. All rights reserved.
"""Do not advertise callable tools when tool_choice disables them."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lmdeploy.serve.openai.chat_completions import register
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.response_parser import BaseResponseParser
from lmdeploy.serve.parsers.tool_parser import ToolParserManager
from lmdeploy.serve.parsers.tool_parser.tool_parser import dump_tools

TOOLS = [{'type': 'function', 'function': {
    'name': 'get_current_weather', 'description': 'WEATHER_TOOL_DECLARATION',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('parser_name', [None, 'glm47'])
@pytest.mark.parametrize('with_history', [False, True])
def test_none_omits_tool_definitions_but_preserves_history(chat_endpoint, hf_media_template,
                                                        stream, parser_name, with_history):
    _, context = chat_endpoint
    template = hf_media_template()

    class Parser(BaseResponseParser):
        reasoning_parser_cls = None
        tool_parser_cls = ToolParserManager.get(parser_name) if parser_name else None
        tokenizer = SimpleNamespace(get_vocab=lambda: {})

    context.response_parser_cls = Parser
    original_preprocess = context.async_engine.preprocess
    prompts = []

    async def preprocess(messages, session, **kwargs):
        assert kwargs['tools'] is None
        prompts.append(template.messages2prompt(messages, tools=kwargs['tools'],
                                               **(kwargs['chat_template_kwargs'] or {})))
        return await original_preprocess(messages, session, **kwargs)

    context.async_engine.preprocess = preprocess
    messages = [{'role': 'user', 'content': 'What is the weather in Boston?'}]
    if with_history:
        messages.extend([
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_weather', 'type': 'function',
             'function': {'name': 'get_current_weather', 'arguments': '{"city":"Boston"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_weather', 'content': 'Sunny, 23 degrees.'},
            {'role': 'user', 'content': 'Summarize the weather without calling tools.'},
        ])
    app = FastAPI()
    register(app, context)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=dict(
            model='fake-model', messages=messages, tools=TOOLS, tool_choice='none', stream=stream,
            max_completion_tokens=16, chat_template_kwargs={'enable_thinking': False}))

    assert response.status_code == 200, response.text
    assert len(prompts) == 1
    assert 'WEATHER_TOOL_DECLARATION' not in prompts[0]
    if with_history:
        assert 'get_current_weather' in prompts[0]
        assert 'Sunny, 23 degrees.' in prompts[0]
    else:
        assert 'get_current_weather' not in prompts[0]
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        choices = [c for e in events for c in e.get('choices', [])]
        assert ''.join(c['delta'].get('content') or '' for c in choices) == 'choice-1'
        assert not any(c['delta'].get('tool_calls') for c in choices)
    else:
        message = response.json()['choices'][0]['message']
        assert message['content'] == 'choice-1'
        assert not message.get('tool_calls')
    assert context.session_manager.sessions == {}


@pytest.mark.parametrize('choice', ['auto', 'required', 'none'])
def test_dump_tools_preserves_request_and_enabled_choices(choice):
    request = ChatCompletionRequest(model='fake-model', messages=[], tools=TOOLS, tool_choice=choice)
    original = request.model_dump()
    normalized = dump_tools(request)

    assert request.model_dump() == original
    assert normalized.tools == (None if choice == 'none' else [request.tools[0].function.model_dump()])
