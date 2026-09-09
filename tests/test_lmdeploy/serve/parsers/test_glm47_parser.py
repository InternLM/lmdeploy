import json

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import Glm47ToolParser, ToolParserManager

from .helpers import (
    feed_tool_chunks,
    final_tool_call,
    flatten_stream_deltas,
    stream_tool_arguments,
    stream_tool_arguments_by_chunk,
    tool_arguments,
)

MODEL_ID = 'zai-org/GLM-4.7'
GLM52_MODEL_ID = 'zai-org/GLM-5.2-FP8'
PARSER_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': name,
        },
    } for name in ('get_weather', 'get_current_temperature', 'no_schema_tool')
]


@pytest.fixture()
def response_parser():
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = None
    cls.tool_parser_cls = ToolParserManager.get('glm47')
    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[],
        stream=True,
        tools=PARSER_TOOLS,
        tool_choice='auto',
    )
    return cls(request=request)


@pytest.fixture()
def response_parser_with_reasoning():
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('default')
    cls.tool_parser_cls = ToolParserManager.get('glm47')
    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[],
        stream=True,
        tools=PARSER_TOOLS,
        tool_choice='auto',
        chat_template_kwargs={'enable_thinking': True},
    )
    return cls(request=request)


REFERENCE_CHUNKS = [
    ('prefix ', [{'content': 'prefix ', 'tool_emitted': False}]),
    ('<tool_call>', []),
    ('get_weather', []),
    ('<arg_key>location</arg_key>',
     [{'tool_emitted': True, 'type': 'function', 'name': 'get_weather', 'arguments': None}]),
    ('<arg_value>Bei', [{'tool_emitted': True, 'type': None, 'name': None, 'arguments': '{"location": "Bei'}]),
    ('jing', [{'tool_emitted': True, 'type': None, 'name': None, 'arguments': 'jing'}]),
    ('</arg_value>', [{'tool_emitted': True, 'type': None, 'name': None, 'arguments': '"'}]),
    ('</tool_call>', [{'tool_emitted': True, 'type': None, 'name': None, 'arguments': '}'}]),
]


def _make_response_parser_with_reasoning(chat_template_kwargs=None):
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('default')
    cls.tool_parser_cls = ToolParserManager.get('glm47')
    request = ChatCompletionRequest(
        model=GLM52_MODEL_ID,
        messages=[],
        stream=True,
        tools=PARSER_TOOLS,
        tool_choice='auto',
        chat_template_kwargs=chat_template_kwargs or {},
    )
    return cls(request=request)


def _collect_stream(parser, chunks):
    reasoning_seen = []
    content_seen = []
    emitted_name = None
    emitted_args = ''

    for chunk in chunks:
        for delta, tool_emitted in parser.stream_chunk(delta_text=chunk, delta_token_ids=[]):
            if delta is not None:
                if delta.reasoning_content:
                    reasoning_seen.append(delta.reasoning_content)
                if delta.content:
                    content_seen.append(delta.content)
            if tool_emitted and delta and delta.tool_calls:
                for call in delta.tool_calls:
                    if call.function and call.function.name:
                        emitted_name = call.function.name
                    if call.function and call.function.arguments:
                        emitted_args += call.function.arguments

    return ''.join(reasoning_seen), ''.join(content_seen), emitted_name, emitted_args


class TestGlm47ResponseParserStreaming:
    """Integration tests for ResponseParser.stream_chunk with glm47 tool
    parser."""

    def test_stream_chunk_matches_reference(self, response_parser):
        for delta_text, expected_events in REFERENCE_CHUNKS:
            actual_events = flatten_stream_deltas(
                response_parser.stream_chunk(delta_text=delta_text, delta_token_ids=[]))
            assert actual_events == expected_events

    def test_stream_chunk_function_name_split_before_arg_key(self, response_parser):
        """Callee name streamed in many deltas before ``<arg_key>`` must not
        freeze early."""
        chunks = [
            '<tool_call>',
            'get',
            '_current',
            '_temperature',
            '<arg_key>location</arg_key><arg_value>北京</arg_value>',
            '</tool_call>',
        ]
        emitted_name = None
        emitted_args = ''
        for chunk in chunks:
            for event in flatten_stream_deltas(response_parser.stream_chunk(delta_text=chunk, delta_token_ids=[])):
                if not event.get('tool_emitted'):
                    continue
                if event.get('name'):
                    emitted_name = event['name']
                if event.get('arguments'):
                    emitted_args += event['arguments']
        assert emitted_name == 'get_current_temperature'
        assert json.loads(emitted_args) == {'location': '北京'}

    def test_stream_chunk_mixed_default_reasoning_and_glm47_tool(self, response_parser_with_reasoning):
        chunks = [
            '<think>',
            'first reason',
            '</think>\nAnswer: ',
            '<tool_call>get_weather',
            '<arg_key>location</arg_key><arg_value>Beijing</arg_value>',
            '</tool_call>',
        ]
        reasoning_seen = []
        content_seen = []
        emitted_name = None
        emitted_args = ''

        for chunk in chunks:
            for event in flatten_stream_deltas(
                    response_parser_with_reasoning.stream_chunk(delta_text=chunk, delta_token_ids=[])):
                if event.get('reasoning_content'):
                    reasoning_seen.append(event['reasoning_content'])
                if event.get('content'):
                    content_seen.append(event['content'])
                if event.get('tool_emitted') and event.get('name'):
                    emitted_name = event['name']
                if event.get('tool_emitted') and event.get('arguments'):
                    emitted_args += event['arguments']

        for _ in range(3):
            for event in flatten_stream_deltas(
                    response_parser_with_reasoning.stream_chunk(delta_text='', delta_token_ids=[])):
                if event.get('reasoning_content'):
                    reasoning_seen.append(event['reasoning_content'])
                if event.get('content'):
                    content_seen.append(event['content'])
                if event.get('tool_emitted') and event.get('name'):
                    emitted_name = event['name']
                if event.get('tool_emitted') and event.get('arguments'):
                    emitted_args += event['arguments']

        assert ''.join(reasoning_seen) == 'first reason'
        assert ''.join(content_seen) == '\nAnswer: '
        assert emitted_name == 'get_weather'
        assert emitted_args == '{"location": "Beijing"}'

    def test_stream_chunk_tool_start_ends_reasoning_without_close_tag(self):
        parser = _make_response_parser_with_reasoning()
        chunks = [
            '<think>',
            'first reason',
            '<tool_call>get_weather',
            '<arg_key>location</arg_key><arg_value>Beijing</arg_value>',
            '</tool_call>',
        ]

        reasoning_seen, content_seen, emitted_name, emitted_args = _collect_stream(parser, chunks)

        assert reasoning_seen == 'first reason'
        assert content_seen == ''
        assert emitted_name == 'get_weather'
        assert emitted_args == '{"location": "Beijing"}'

    def test_stream_chunk_split_tool_start_ends_reasoning_without_close_tag(self):
        parser = _make_response_parser_with_reasoning()
        chunks = [
            '<think>first reason<to',
            'ol_call>get_weather<arg_key>location</arg_key>',
            '<arg_value>Beijing</arg_value></tool_call>',
        ]

        reasoning_seen, content_seen, emitted_name, emitted_args = _collect_stream(parser, chunks)

        assert reasoning_seen == 'first reason'
        assert content_seen == ''
        assert emitted_name == 'get_weather'
        assert emitted_args == '{"location": "Beijing"}'

    def test_stream_chunk_resumes_reasoning_after_nested_tool(self):
        parser = _make_response_parser_with_reasoning()
        chunks = [
            '<think>before<tool_call>get_weather',
            '<arg_key>location</arg_key><arg_value>Beijing</arg_value>',
            '</tool_call>after</think>answer',
        ]

        reasoning_seen, content_seen, emitted_name, emitted_args = _collect_stream(parser, chunks)

        assert reasoning_seen == 'beforeafter'
        assert content_seen == 'answer'
        assert emitted_name == 'get_weather'
        assert emitted_args == '{"location": "Beijing"}'

    def test_stream_chunk_reasoning_effort_high_starts_in_reasoning_mode(self):
        parser = _make_response_parser_with_reasoning({'reasoning_effort': 'high'})

        delta, tool_emitted = parser.stream_chunk(delta_text='first reason', delta_token_ids=[])[0]

        assert tool_emitted is False
        assert delta is not None
        assert delta.reasoning_content == 'first reason'
        assert delta.content is None

    def test_stream_chunk_enable_thinking_false_starts_in_plain_mode(self):
        parser = _make_response_parser_with_reasoning({'enable_thinking': False})

        delta, tool_emitted = parser.stream_chunk(delta_text='plain answer', delta_token_ids=[])[0]

        assert tool_emitted is False
        assert delta is not None
        assert delta.content == 'plain answer'
        assert delta.reasoning_content is None

    def test_stream_chunk_keeps_string_without_schema(self, response_parser):
        chunks = [
            '<tool_call>',
            'no_schema_tool',
            '<arg_key>zip</arg_key><arg_value>77004</arg_value>',
            '<arg_key>active</arg_key><arg_value>true</arg_value>',
            '</tool_call>',
        ]
        emitted_name = None
        emitted_args = ''
        for chunk in chunks:
            for event in flatten_stream_deltas(response_parser.stream_chunk(delta_text=chunk, delta_token_ids=[])):
                if not event.get('tool_emitted'):
                    continue
                if event.get('name'):
                    emitted_name = event['name']
                if event.get('arguments'):
                    emitted_args += event['arguments']
        assert emitted_name == 'no_schema_tool'
        assert emitted_args == '{"zip": "77004", "active": "true"}'

    def test_stream_chunk_drops_unavailable_tool(self, response_parser):
        text = ('<tool_call>img_gen'
                '<arg_key>prompt</arg_key><arg_value>edit image</arg_value>'
                '</tool_call>')

        deltas = [delta for delta, _ in response_parser.stream_chunk(text, [])]
        calls = [call for delta in deltas for call in (delta.tool_calls or [])]

        assert calls == []


class TestGlm47ToolParserComplete:
    """Complete-parse tests for glm47 tool payloads."""

    def test_parse_complete_resumes_reasoning_after_nested_tool(self):
        parser = _make_response_parser_with_reasoning()
        text = (
            '<think>before'
            '<tool_call>get_weather'
            '<arg_key>location</arg_key><arg_value>Beijing</arg_value>'
            '</tool_call>after</think>answer'
        )

        content, tool_calls, reasoning = parser.parse_complete(text)

        assert reasoning == 'beforeafter'
        assert content == 'answer'
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == 'get_weather'
        assert json.loads(tool_calls[0].function.arguments) == {'location': 'Beijing'}

    def test_parse_complete_keeps_post_tool_text_in_unclosed_reasoning(self):
        parser = _make_response_parser_with_reasoning()
        text = (
            '<think>before'
            '<tool_call>get_weather'
            '<arg_key>location</arg_key><arg_value>Beijing</arg_value>'
            '</tool_call>after'
        )

        content, tool_calls, reasoning = parser.parse_complete(text)

        assert reasoning == 'beforeafter'
        assert content is None
        assert len(tool_calls) == 1

    def test_parse_complete_tool_start_ends_reasoning_without_close_tag(self):
        parser = _make_response_parser_with_reasoning()
        text = (
            '<think>first reason'
            '<tool_call>get_weather'
            '<arg_key>location</arg_key><arg_value>Beijing</arg_value>'
            '</tool_call>'
        )

        content, tool_calls, reasoning = parser.parse_complete(text)

        assert content is None
        assert reasoning == 'first reason'
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == 'get_weather'
        assert json.loads(tool_calls[0].function.arguments) == {'location': 'Beijing'}

    def test_parse_complete_drops_unavailable_tool(self, response_parser):
        content, tool_calls, _ = response_parser.parse_complete('<tool_call>img_gen</tool_call>')

        assert content is None
        assert tool_calls is None

    def test_final_tool_payload_with_arguments(self):
        parser = Glm47ToolParser()
        payload = (
            'get_weather'
            '<arg_key>location</arg_key><arg_value>Beijing</arg_value>'
            '<arg_key>unit</arg_key><arg_value>celsius</arg_value>'
        )
        tool_call = final_tool_call(parser, payload)
        assert tool_call is not None
        assert tool_call.function.name == 'get_weather'
        assert json.loads(tool_call.function.arguments) == {
            'location': 'Beijing',
            'unit': 'celsius',
        }

    def test_final_tool_payload_without_arguments(self):
        parser = Glm47ToolParser()
        tool_call = final_tool_call(parser, 'get_time')
        assert tool_call is not None
        assert tool_call.function.name == 'get_time'
        assert json.loads(tool_call.function.arguments) == {}

    def test_final_tool_payload_coerces_values_by_schema(self):
        parser = Glm47ToolParser()
        properties = {
            name: {
                'type': schema_type
            }
            for name, schema_type in {
                'name': 'string',
                'age': 'integer',
                'height': 'number',
                'active': 'boolean',
                'meta': 'object',
                'scores': 'array',
                'misc': 'null',
            }.items()
        }
        request = ChatCompletionRequest(
            model=MODEL_ID,
            messages=[],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'typed_tool',
                    'parameters': {
                        'type': 'object',
                        'properties': properties,
                    },
                },
            }],
            tool_choice='auto',
        )
        parser.adjust_request(request)
        payload = (
            'typed_tool'
            '<arg_key>name</arg_key><arg_value>Chen</arg_value>'
            '<arg_key>age</arg_key><arg_value>29</arg_value>'
            '<arg_key>height</arg_key><arg_value>1.75</arg_value>'
            '<arg_key>active</arg_key><arg_value>true</arg_value>'
            '<arg_key>meta</arg_key><arg_value>{"city":"Houston"}</arg_value>'
            '<arg_key>scores</arg_key><arg_value>[98,87]</arg_value>'
            '<arg_key>misc</arg_key><arg_value>null</arg_value>'
        )

        tool_call = final_tool_call(parser, payload)

        assert json.loads(tool_call.function.arguments) == {
            'name': 'Chen',
            'age': 29,
            'height': 1.75,
            'active': True,
            'meta': {
                'city': 'Houston'
            },
            'scores': [98, 87],
            'misc': None,
        }

    def test_duplicate_arguments_preserve_raw_string_whitespace(self):
        payload = (
            'f'
            '<arg_key>a</arg_key><arg_value>  one  </arg_value>'
            '<arg_key>a</arg_key><arg_value>two</arg_value>'
        )
        expected = '{"a": "  one  ", "a": "two"}'

        pending, deltas, _ = feed_tool_chunks(
            Glm47ToolParser(),
            [payload, '</tool_call>tail'],
        )
        complete = final_tool_call(Glm47ToolParser(), payload)

        assert pending == 'tail'
        assert tool_arguments(deltas) == expected
        assert complete.function.arguments == expected

    @pytest.mark.parametrize(
        ('value_chunks', 'expected'),
        [
            (['12'], 12),
            (['2', 'a'], '2a'),
        ],
        ids=['integer', 'invalid-integer'],
    )
    def test_typed_argument_waits_for_value_close(self, value_chunks, expected):
        parser = Glm47ToolParser()
        request = ChatCompletionRequest(
            model=MODEL_ID,
            messages=[],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'typed_tool',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'age': {
                                'type': 'integer'
                            }
                        },
                    },
                },
            }],
            tool_choice='auto',
        )
        parser.adjust_request(request)
        chunks = [
            'typed_tool',
            '<arg_key>age</arg_key><arg_value>',
            *value_chunks,
            '</arg_value>',
            '',
        ]

        streamed_arguments, per_chunk = stream_tool_arguments_by_chunk(parser, chunks)

        assert all(fragment == '' for fragment in per_chunk[1:-2])
        assert json.loads(streamed_arguments) == {'age': expected}

    def test_string_after_typed_argument_uses_its_own_value(self):
        parser = Glm47ToolParser()
        request = ChatCompletionRequest(
            model=MODEL_ID,
            messages=[],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'typed_tool',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'age': {
                                'type': 'integer'
                            },
                            'name': {
                                'type': 'string'
                            },
                        },
                    },
                },
            }],
            tool_choice='auto',
        )
        parser.adjust_request(request)
        chunks = [
            'typed_tool<arg_key>age</arg_key><arg_value>',
            '1',
            '2</arg_value><arg_key>name</arg_key><arg_value>',
            'Alice</arg_value>',
            '',
        ]

        streamed_arguments = stream_tool_arguments(parser, chunks)

        assert json.loads(streamed_arguments) == {'age': 12, 'name': 'Alice'}

    def test_outer_close_text_inside_argument_is_not_a_block_boundary(self):
        payload = (
            'f<arg_key>a</arg_key>'
            '<arg_value>before </tool_call> after</arg_value>'
        )

        pending, deltas, _ = feed_tool_chunks(
            Glm47ToolParser(),
            [payload + '</tool_call>tail'],
        )

        assert pending == 'tail'
        assert tool_arguments(deltas) == '{"a": "before </tool_call> after"}'
