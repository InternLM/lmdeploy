# Copyright (c) OpenMMLab. All rights reserved.
"""Do not advertise callable tools when tool_choice disables them."""
from types import SimpleNamespace

import pytest

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
def test_none_omits_tool_definitions_but_preserves_history(chat_endpoint, chat_client, rendered_prompts, sse_events,
                                                        stream, parser_name, with_history):
    _, context = chat_endpoint

    class Parser(BaseResponseParser):
        reasoning_parser_cls = None
        tool_parser_cls = ToolParserManager.get(parser_name) if parser_name else None
        tokenizer = SimpleNamespace(get_vocab=lambda: {})

    context.response_parser_cls = Parser
    prompts = rendered_prompts
    messages = [{'role': 'user', 'content': 'What is the weather in Boston?'}]
    if with_history:
        messages.extend([
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_weather', 'type': 'function',
             'function': {'name': 'get_current_weather', 'arguments': '{"city":"Boston"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_weather', 'content': 'Sunny, 23 degrees.'},
            {'role': 'user', 'content': 'Summarize the weather without calling tools.'},
        ])
    response = chat_client.post('/v1/chat/completions', json=dict(
        model='fake-model', messages=messages, tools=TOOLS, tool_choice='none', stream=stream,
        max_completion_tokens=16, chat_template_kwargs={'enable_thinking': False}))

    assert response.status_code == 200, response.text
    assert len(prompts) == 1
    assert 'WEATHER_TOOL_DECLARATION' not in prompts[0]
    if with_history:
        assert 'get_current_weather' in prompts[0]
        assert 'Sunny, 23 degrees.' in prompts[0]
        assert 'None' not in prompts[0]
    else:
        assert 'get_current_weather' not in prompts[0]
    if stream:
        events = sse_events(response)
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


@pytest.fixture
def harmony_parser(monkeypatch):
    pytest.importorskip('openai_harmony')
    from lmdeploy.serve.parsers import _openai_harmony as harmony

    class ScriptedParser:
        current_channel = 'final'
        current_recipient = None
        last_content_delta = ''

        def __init__(self, events):
            self.events = events

        def process(self, token):
            self.current_channel, self.current_recipient, self.last_content_delta = self.events[token]

    events = {}
    monkeypatch.setattr(harmony, 'get_encoding', lambda: None)
    monkeypatch.setattr(harmony, 'StreamableParser', lambda *a, **kw: ScriptedParser(events))
    monkeypatch.setattr(harmony.GptOssResponseParser, '_build_tool_grammar', lambda *a: None)
    return harmony.GptOssResponseParser, events


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('choice', ['none', 'auto'])
@pytest.mark.parametrize('channel', ['commentary', 'analysis'])
@pytest.mark.parametrize('finish_reason', ['stop', 'length'])
@pytest.mark.parametrize('with_answer', [False, True])
def test_harmony_tool_output_respects_choice(chat_endpoint, chat_client, sse_events, harmony_parser,
                                           stream, choice, channel, finish_reason, with_answer, monkeypatch):
    _, context = chat_endpoint
    context.response_parser_cls, events = harmony_parser
    events.update({
        1: ('analysis', None, 'Checking the weather.'),
        2: (channel, 'functions.get_current_weather', ''),
        3: (channel, 'functions.get_current_weather', '{"city":"'),
        4: (channel, 'functions.get_current_weather', 'Boston"}'),
    })
    if with_answer:
        events[5] = ('final', None, 'Sunny.')
    original_generate = context.async_engine.generate

    def generate(request, **kwargs):
        async def results():
            async for result in original_generate(request, **kwargs):
                for token in events:
                    yield SimpleNamespace(**{**vars(result), 'response': '', 'token_ids': [token],
                                             'finish_reason': None, 'generate_token_len': token})
                # A separate terminal chunk must retain the tool decision.
                yield SimpleNamespace(**{**vars(result), 'response': '', 'token_ids': [],
                                         'finish_reason': finish_reason, 'generate_token_len': len(events)})
        return results()

    monkeypatch.setattr(context.async_engine, 'generate', generate)
    response = chat_client.post('/v1/chat/completions', json=dict(
        model='fake-model', messages=[{'role': 'user', 'content': 'Weather?'}],
        tools=TOOLS, tool_choice=choice, stream=stream))
    assert response.status_code == 200, response.text
    if stream:
        choices = [c for event in sse_events(response) for c in event.get('choices', [])]
        messages = [c['delta'] for c in choices]
        actual_finish = choices[-1]['finish_reason']
    else:
        result = response.json()['choices'][0]
        messages = [result['message']]
        actual_finish = result['finish_reason']
    calls = [call for message in messages for call in message.get('tool_calls') or []]
    assert bool(calls) == (choice == 'auto')
    if calls:
        assert calls[0]['function']['name'] == 'get_current_weather'
        assert ''.join(call['function'].get('arguments') or '' for call in calls) == '{"city":"Boston"}'
    expected_content = ('{"city":"Boston"}' if choice == 'none' else '') + ('Sunny.' if with_answer else '')
    assert ''.join(message.get('content') or '' for message in messages) == expected_content
    assert ''.join(message.get('reasoning_content') or '' for message in messages) == 'Checking the weather.'
    assert actual_finish == ('tool_calls' if choice == 'auto' and finish_reason == 'stop' else finish_reason)
    assert context.session_manager.sessions == {}


@pytest.mark.parametrize('tools', [None, [], TOOLS])
def test_dump_tools_rejects_unknown_named_choice(tools):
    request = ChatCompletionRequest(model='fake-model', messages=[], tools=tools,
                                    tool_choice={'type': 'function', 'function': {'name': 'missing_tool'}})
    original = request.model_dump()
    with pytest.raises(ValueError, match='missing_tool.*not found in request.tools'):
        dump_tools(request)
    assert request.model_dump() == original


def test_dump_tools_preserves_matching_named_choice():
    request = ChatCompletionRequest(
        model='fake-model', messages=[], tools=TOOLS + [{'type': 'function', 'function': {'name': 'other'}}],
        tool_choice={'type': 'function', 'function': {'name': 'get_current_weather'}})
    original = request.model_dump()
    assert dump_tools(request).tools == [request.tools[0].function.model_dump()]
    assert request.model_dump() == original


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('tools', [None, [], TOOLS])
@pytest.mark.parametrize('parser_name', ['glm47', 'gpt-oss'])
def test_unknown_named_choice_returns_400_before_generation(chat_endpoint, chat_client, request,
                                                          stream, tools, parser_name):
    _, context = chat_endpoint
    if parser_name == 'gpt-oss':
        context.response_parser_cls, _ = request.getfixturevalue('harmony_parser')
    else:
        class Parser(BaseResponseParser):
            reasoning_parser_cls = None
            tool_parser_cls = ToolParserManager.get(parser_name)
            tokenizer = SimpleNamespace(get_vocab=lambda: {})

        context.response_parser_cls = Parser
    response = chat_client.post('/v1/chat/completions', json=dict(
        model='fake-model', messages=[{'role': 'user', 'content': 'Weather?'}], tools=tools, stream=stream,
        tool_choice={'type': 'function', 'function': {'name': 'missing_tool'}}))
    assert response.status_code == 400, response.text
    assert 'missing_tool' in response.json()['message']
    assert 'not found in request.tools' in response.json()['message']
    assert context.async_engine.call_count == 0
    assert context.async_engine.gen_configs == []
    assert context.session_manager.sessions == {}
