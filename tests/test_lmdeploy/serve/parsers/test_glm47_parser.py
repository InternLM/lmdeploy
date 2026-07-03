import json

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import Glm47ToolParser, ToolParserManager

MODEL_ID = 'zai-org/GLM-4.7'


@pytest.fixture()
def response_parser():
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = None
    cls.tool_parser_cls = ToolParserManager.get('glm47')
    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[],
        stream=True,
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
        tool_choice='auto',
        chat_template_kwargs={'enable_thinking': True},
    )
    return cls(request=request)


def _flatten_stream_deltas(deltas):
    events = []
    for delta_msg, tool_emitted in deltas:
        if delta_msg is None:
            continue
        if delta_msg.reasoning_content is not None:
            events.append({'reasoning_content': delta_msg.reasoning_content, 'tool_emitted': tool_emitted})
        if delta_msg.content is not None:
            events.append({'content': delta_msg.content, 'tool_emitted': tool_emitted})
        if delta_msg.tool_calls:
            for call in delta_msg.tool_calls:
                events.append({
                    'tool_emitted': tool_emitted,
                    'type': call.type,
                    'name': call.function.name if call.function else None,
                    'arguments': call.function.arguments if call.function else None,
                })
    return events


def _stream_tool_arguments(parser, chunks):
    parser.start_tool_call()
    argument_fragments = []
    for chunk, final in chunks:
        for call in parser.decode_tool_incremental(chunk, final=final):
            if call.function and call.function.arguments is not None:
                argument_fragments.append(call.function.arguments)
    parser.finish_tool_call()
    return ''.join(argument_fragments)


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


class TestGlm47ResponseParserStreaming:
    """Integration tests for ResponseParser.stream_chunk with glm47 tool
    parser."""

    def test_stream_chunk_matches_reference(self, response_parser):
        actual = []
        expected = []
        for delta_text, expected_events in REFERENCE_CHUNKS:
            actual.extend(
                _flatten_stream_deltas(response_parser.stream_chunk(delta_text=delta_text, delta_token_ids=[])))
            expected.extend(expected_events)
        assert actual == expected

    def test_stream_chunk_emits_arg_value_before_arg_value_close(self, response_parser):
        chunks = [
            '<tool_call>',
            'get_weather',
            '<arg_key>location</arg_key>',
            '<arg_value>San',
            ' Francisco',
            ', CA',
        ]

        argument_fragments = []
        emitted_before_close = False
        for chunk in chunks:
            for event in _flatten_stream_deltas(response_parser.stream_chunk(delta_text=chunk, delta_token_ids=[])):
                fragment = event.get('arguments')
                if fragment:
                    argument_fragments.append(fragment)
                    emitted_before_close = True

        assert emitted_before_close is True
        assert ''.join(argument_fragments) == '{"location": "San Francisco, CA'

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
            for event in _flatten_stream_deltas(response_parser.stream_chunk(delta_text=chunk, delta_token_ids=[])):
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
            for event in _flatten_stream_deltas(
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
            for event in _flatten_stream_deltas(
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
            for event in _flatten_stream_deltas(response_parser.stream_chunk(delta_text=chunk, delta_token_ids=[])):
                if not event.get('tool_emitted'):
                    continue
                if event.get('name'):
                    emitted_name = event['name']
                if event.get('arguments'):
                    emitted_args += event['arguments']
        assert emitted_name == 'no_schema_tool'
        assert emitted_args == '{"zip": "77004", "active": "true"}'


class TestGlm47ToolParserComplete:
    """Complete-parse tests for glm47 tool payloads."""

    def test_parse_tool_call_complete_with_arguments(self):
        parser = Glm47ToolParser()
        payload = (
            'get_weather'
            '<arg_key>location</arg_key><arg_value>Beijing</arg_value>'
            '<arg_key>unit</arg_key><arg_value>celsius</arg_value>'
        )
        tool_call = parser.parse_tool_call_complete(payload)
        assert tool_call is not None
        assert tool_call.function.name == 'get_weather'
        assert json.loads(tool_call.function.arguments) == {
            'location': 'Beijing',
            'unit': 'celsius',
        }

    def test_parse_tool_call_complete_without_arguments(self):
        parser = Glm47ToolParser()
        tool_call = parser.parse_tool_call_complete('get_time')
        assert tool_call is not None
        assert tool_call.function.name == 'get_time'
        assert json.loads(tool_call.function.arguments) == {}

    def test_parse_tool_call_complete_coerces_types_by_schema(self):
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
                            'name': {
                                'type': 'string'
                            },
                            'age': {
                                'type': 'integer'
                            },
                            'height': {
                                'type': 'number'
                            },
                            'active': {
                                'type': 'boolean'
                            },
                            'meta': {
                                'type': 'object'
                            },
                            'scores': {
                                'type': 'array'
                            },
                            'misc': {
                                'type': 'null'
                            },
                        },
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
        tool_call = parser.parse_tool_call_complete(payload)
        assert tool_call is not None
        assert tool_call.function.name == 'typed_tool'
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

    def test_parse_tool_call_complete_keeps_string_without_schema(self):
        parser = Glm47ToolParser()
        payload = (
            'no_schema_tool'
            '<arg_key>zip</arg_key><arg_value>77004</arg_value>'
            '<arg_key>active</arg_key><arg_value>true</arg_value>'
            '<arg_key>meta</arg_key><arg_value>{"city":"Houston"}</arg_value>'
        )
        tool_call = parser.parse_tool_call_complete(payload)
        assert tool_call is not None
        assert tool_call.function.name == 'no_schema_tool'
        assert json.loads(tool_call.function.arguments) == {
            'zip': '77004',
            'active': 'true',
            'meta': '{"city":"Houston"}',
        }

    def test_streamed_arguments_match_complete_parse_for_quoted_string_value(self):
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
        payload = 'typed_tool<arg_key>name</arg_key><arg_value>"Chen"</arg_value>'

        streamed_arguments = _stream_tool_arguments(
            parser,
            [
                ('typed_tool', False),
                ('<arg_key>name</arg_key>', False),
                ('<arg_value>', False),
                ('"Chen"', False),
                ('</arg_value>', False),
                ('', True),
            ],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments

    def test_streamed_arguments_match_complete_parse_for_invalid_integer_value(self):
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
                        },
                    },
                },
            }],
            tool_choice='auto',
        )
        parser.adjust_request(request)
        payload = 'typed_tool<arg_key>age</arg_key><arg_value>abc</arg_value>'

        streamed_arguments = _stream_tool_arguments(
            parser,
            [
                ('typed_tool', False),
                ('<arg_key>age</arg_key>', False),
                ('<arg_value>', False),
                ('abc', False),
                ('</arg_value>', False),
                ('', True),
            ],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments

    def test_streamed_arguments_match_complete_parse_when_next_arg_starts_with_previous_close(self):
        parser = Glm47ToolParser()
        payload = 'two_args<arg_key>a</arg_key><arg_value>one</arg_value><arg_key>b</arg_key><arg_value>two</arg_value>'

        streamed_arguments = _stream_tool_arguments(
            parser,
            [
                ('two_args', False),
                ('<arg_key>a</arg_key>', False),
                ('<arg_value>one', False),
                ('</arg_value><arg_key>b</arg_key><arg_value>two', False),
                ('</arg_value>', False),
                ('', True),
            ],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments

    def test_streamed_arguments_match_complete_parse_when_close_chunk_has_value_tail(self):
        parser = Glm47ToolParser()
        payload = 'f<arg_key>a</arg_key><arg_value>San Francisco</arg_value>'

        streamed_arguments = _stream_tool_arguments(
            parser,
            [
                ('f', False),
                ('<arg_key>a</arg_key><arg_value>San ', False),
                ('Francisco</arg_value>', False),
                ('', True),
            ],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments
