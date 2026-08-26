# Copyright (c) OpenMMLab. All rights reserved.
import json

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.muse_glimmer_response_parser import MuseGlimmerResponseParser


def _request(**kwargs):
    return ChatCompletionRequest(
        model='meta-models/Muse-Glimmer-30B',
        messages=[{'role': 'user', 'content': 'hello'}],
        **kwargs,
    )


def _response():
    return (
        'to=self<|message|>I should check both places.<|eom|>'
        '<|start|>assistant to=weather<|message|><atem:function_calls>'
        '<atem:invoke name="get_weather">'
        '<atem:parameter name="city">"Paris"</atem:parameter>'
        '<atem:parameter name="days">3</atem:parameter>'
        '</atem:invoke>'
        '<atem:invoke name="get_time">'
        '<atem:parameter name="zone">Europe/Paris</atem:parameter>'
        '</atem:invoke>'
        '</atem:function_calls><|eom|>'
        '<|start|>assistant to=user<|message|>Here is the result.<|eot|>')


def test_request_configuration():
    parser = MuseGlimmerResponseParser(_request(
        reasoning_effort='max',
        tools=[{
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'parameters': {'type': 'object'},
            },
        }],
    ))

    assert parser.request.chat_template_kwargs == {'reasoning_strength': 'xhigh'}
    assert parser.request.skip_special_tokens is False
    assert parser.request.spaces_between_special_tokens is False
    assert parser.request.tools == [{
        'description': None,
        'name': 'get_weather',
        'parameters': {'type': 'object'},
    }]


def test_complete_reasoning_content_and_parallel_tools():
    parser = MuseGlimmerResponseParser(_request())
    content, calls, reasoning = parser.parse_complete(_response())

    assert reasoning == 'I should check both places.'
    assert content == 'Here is the result.'
    assert [call.function.name for call in calls] == ['get_weather', 'get_time']
    assert json.loads(calls[0].function.arguments) == {'city': 'Paris', 'days': 3}
    assert json.loads(calls[1].function.arguments) == {'zone': 'Europe/Paris'}
    assert parser.validate_complete(_response())


def test_streaming_is_independent_of_chunk_boundaries():
    parser = MuseGlimmerResponseParser(_request())
    content = []
    reasoning = []
    calls = []

    text = _response()
    splits = [text[:17], text[17:83], text[83:151], text[151:247], text[247:]]
    for chunk in splits:
        for delta, tool_emitted in parser.stream_chunk(chunk, []):
            if delta.content:
                content.append(delta.content)
            if delta.reasoning_content:
                reasoning.append(delta.reasoning_content)
            if delta.tool_calls:
                calls.extend(delta.tool_calls)
                assert tool_emitted

    assert ''.join(reasoning) == 'I should check both places.'
    assert ''.join(content) == 'Here is the result.'
    assert [call.index for call in calls] == [0, 1]
    assert [call.function.name for call in calls] == ['get_weather', 'get_time']
    assert json.loads(calls[0].function.arguments) == {'city': 'Paris', 'days': 3}
    assert json.loads(calls[1].function.arguments) == {'zone': 'Europe/Paris'}
    assert parser.validate_complete()


def test_validation_rejects_unclosed_protocol_sections():
    parser = MuseGlimmerResponseParser(_request())
    assert not parser.validate_complete('to=self<|message|>unfinished')
    assert not parser.validate_complete(
        'to=weather<|message|><atem:invoke name="get_weather">')
    assert parser.validate_complete('to=user<|message|>visible response')


def test_tool_choice_none_preserves_tool_channel_as_content():
    parser = MuseGlimmerResponseParser(_request(tool_choice='none'))
    text = (
        'to=weather<|message|><atem:function_calls>'
        '<atem:invoke name="get_weather"></atem:invoke>'
        '</atem:function_calls><|eom|>')
    content, calls, reasoning = parser.parse_complete(text)

    assert '<atem:invoke name="get_weather">' in content
    assert calls is None
    assert reasoning is None
