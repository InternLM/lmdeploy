# Copyright (c) OpenMMLab. All rights reserved.
"""Keep constrained answers and assistant continuations in their channels."""
import json
from types import SimpleNamespace

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.response_parser import BaseResponseParser
from lmdeploy.serve.parsers.tool_parser import ToolParserManager


class Parser(BaseResponseParser):
    reasoning_parser_cls = ReasoningParserManager.get('default')
    tool_parser_cls = ToolParserManager.get('glm47')
    tokenizer = SimpleNamespace(get_vocab=lambda: {'<think>': 1, '</think>': 2})


FORMATS = [
    {'type': 'json_object'},
    {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': {'type': 'object'}}},
    {'type': 'regex_schema', 'regex_schema': '.*'},
]


@pytest.mark.parametrize('response_format', FORMATS)
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('template_kwargs', [{}, {'enable_thinking': True, 'thinking': True}])
def test_constrained_answers_are_literal_content(response_format, stream, template_kwargs):
    request = ChatCompletionRequest(
        model='fake', messages=[{'role': 'user', 'content': 'Return JSON'}], response_format=response_format,
        reasoning_effort='high', chat_template_kwargs=template_kwargs)
    original = request.model_dump()
    parser = Parser(request)
    text = json.dumps({'species': 'rabbit', 'count': 12, 'literal': '<think>data</think><tool_call>data</tool_call>'})
    assert parser.request.chat_template_kwargs['enable_thinking'] is False
    assert parser.request.response_format['type'] == response_format['type']
    assert request.model_dump() == original
    if stream:
        deltas = []
        for i, char in enumerate(text):
            deltas.extend(parser.stream_chunk(char, [100], final=i == len(text) - 1))
        assert ''.join(delta.content or '' for delta, _ in deltas) == text
        assert all(not delta.reasoning_content and not delta.tool_calls and not emitted for delta, emitted in deltas)
    else:
        assert parser.parse_complete(text, [100] * len(text)) == (text, None, None)
    assert parser.reasoning_tokens == 0


@pytest.mark.parametrize('prefill,expected_reasoning', [
    ('The capital of France is', False),
    ('<think>Consider the capital', True),
    ('<think>France</think>The capital of France is', False),
    ('<think>first</think><think>second', True),
    ('', True),
])
@pytest.mark.parametrize('stream', [False, True])
def test_assistant_prefill_keeps_its_channel(prefill, expected_reasoning, stream):
    parser = Parser(ChatCompletionRequest(model='fake', messages=[
        {'role': 'user', 'content': 'What is the capital of France?'},
        {'role': 'assistant', 'content': prefill}], reasoning_effort='high'))
    if stream:
        deltas = parser.stream_chunk(' Paris', [100], final=True)
        content = ''.join(delta.content or '' for delta, _ in deltas) or None
        reasoning = ''.join(delta.reasoning_content or '' for delta, _ in deltas) or None
    else:
        content, tools, reasoning = parser.parse_complete(' Paris', [100])
        assert tools is None
    assert content == (None if expected_reasoning else ' Paris')
    assert reasoning == (' Paris' if expected_reasoning else None)
    assert parser.reasoning_tokens == int(expected_reasoning)


def test_user_turn_still_starts_with_reasoning():
    parser = Parser(ChatCompletionRequest(model='fake', messages=[{'role': 'user', 'content': 'Hi'}]))
    assert parser.parse_complete('analysis</think>answer', [100, 2, 101]) == ('answer', None, 'analysis')
    assert parser.reasoning_tokens == 1
