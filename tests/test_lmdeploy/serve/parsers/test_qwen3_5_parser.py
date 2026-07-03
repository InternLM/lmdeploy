import json

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager
from lmdeploy.serve.parsers.tool_parser.qwen3coder_tool_parser import Qwen3CoderToolParser

MODEL_ID = 'Qwen/Qwen3.5-35B-A3B'


def _build_response_parser():
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('default')
    cls.tool_parser_cls = ToolParserManager.get('qwen3coder')

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
    for chunk in chunks:
        for call in parser.decode_tool_incremental(chunk, final=False):
            if call.function and call.function.arguments is not None:
                argument_fragments.append(call.function.arguments)
    parser.finish_tool_call()
    return ''.join(argument_fragments)


def _stream_tool_arguments_by_chunk(parser, chunks):
    parser.start_tool_call()
    argument_fragments = []
    per_chunk = []
    for chunk in chunks:
        chunk_fragments = []
        for call in parser.decode_tool_incremental(chunk, final=False):
            if call.function and call.function.arguments is not None:
                argument_fragments.append(call.function.arguments)
                chunk_fragments.append(call.function.arguments)
        per_chunk.append(''.join(chunk_fragments))
    parser.finish_tool_call()
    return ''.join(argument_fragments), per_chunk


REFERENCE_CHUNKS = [
    ('计划', [{'reasoning_content': '计划', 'tool_emitted': False}]),
    ('调用', [{'reasoning_content': '调用', 'tool_emitted': False}]),
    ('get', [{'reasoning_content': 'get', 'tool_emitted': False}]),
    ('_current', [{'reasoning_content': '_current', 'tool_emitted': False}]),
    ('_temperature', [{'reasoning_content': '_temperature', 'tool_emitted': False}]),
    ('函数', [{'reasoning_content': '函数', 'tool_emitted': False}]),
    ('并提供', [{'reasoning_content': '并提供', 'tool_emitted': False}]),
    ('location', [{'reasoning_content': 'location', 'tool_emitted': False}]),
    ('参数', [{'reasoning_content': '参数', 'tool_emitted': False}]),
    ('。', [{'reasoning_content': '。', 'tool_emitted': False}]),
    ('\n', [{'reasoning_content': '\n', 'tool_emitted': False}]),
    ('</think>', []),
    ('\n\n', [{'content': '\n\n', 'tool_emitted': False}]),
    ('<tool_call>', []),
    ('\n', []),
    ('<', []),
    ('function', []),
    ('=get', []),
    ('_current', []),
    ('_temperature', []),
    ('>', [{'tool_emitted': True, 'type': 'function', 'name': 'get_current_temperature', 'arguments': None}]),
    ('\n', []),
    ('<', []),
    ('parameter', []),
    ('=location', []),
    ('>', []),
    ('\n', []),
    ('Be', [{'tool_emitted': True, 'type': None, 'name': None, 'arguments': '{"location": "Be'}]),
    ('ijing', [{'tool_emitted': True, 'type': None, 'name': None, 'arguments': 'ijing'}]),
    (',', [{'tool_emitted': True, 'type': None, 'name': None, 'arguments': ','}]),
    (' China', [{'tool_emitted': True, 'type': None, 'name': None, 'arguments': ' China'}]),
    ('\n', []),
    ('</', []),
    ('parameter', []),
    ('>', [{'tool_emitted': True, 'type': None, 'name': None, 'arguments': '"'}]),
    ('\n', []),
    ('</', []),
    ('function', []),
    ('>', [{'tool_emitted': True, 'type': None, 'name': None, 'arguments': '}'}]),
    ('\n', []),
    ('</tool_call>', []),
    ('', [{'content': '', 'tool_emitted': False}]),
]


class TestQwen3_5ResponseParserStreaming:
    """Integration test for ResponseParser.stream_chunk with Qwen3.5 Coder
    parsers."""

    def test_stream_chunk_matches_reference(self):
        response_parser = _build_response_parser()
        actual = []
        expected = []
        for delta_text, expected_events in REFERENCE_CHUNKS:
            actual.extend(
                _flatten_stream_deltas(response_parser.stream_chunk(delta_text=delta_text, delta_token_ids=[])))
            expected.extend(expected_events)
        assert actual == expected

    def test_stream_chunk_emits_parameter_value_before_parameter_close(self):
        response_parser = _build_response_parser()
        chunks = [
            '</think>',
            '<tool_call>',
            '<function=get_current_temperature>',
            '<parameter=location>',
            'San',
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

    def test_parse_complete_parallel_tool_calls_keep_distinct_arguments(self):
        """Regression: parallel tool calls must not reuse the first call's args."""
        response_parser = _build_response_parser()
        text = """
</think>

<tool_call>
<function=get_current_weather>
<parameter=location>
Boston, MA
</parameter>
</function>
</tool_call>
<tool_call>
<function=get_current_weather>
<parameter=location>
San Francisco, CA
</parameter>
</function>
</tool_call>
""".strip()

        content, tool_calls, _ = response_parser.parse_complete(text)

        assert (content or '').strip() == ''
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert json.loads(tool_calls[0].function.arguments) == {'location': 'Boston, MA'}
        assert json.loads(tool_calls[1].function.arguments) == {'location': 'San Francisco, CA'}

    def test_parse_tool_call_complete_treats_params_as_strings(self):
        parser = Qwen3CoderToolParser()
        payload = """
<function=find_user_id_by_name_zip>
<parameter=first_name>
Chen
</parameter>
<parameter=last_name>
Johnson
</parameter>
<parameter=zip>
77004
</parameter>
</function>
""".strip()

        tool_call = parser.parse_tool_call_complete(payload)

        assert tool_call is not None
        assert tool_call.function.name == 'find_user_id_by_name_zip'
        assert json.loads(tool_call.function.arguments) == {
            'first_name': 'Chen',
            'last_name': 'Johnson',
            'zip': '77004',
        }

    def test_parse_tool_call_complete_coerces_types_by_schema(self):
        parser = Qwen3CoderToolParser()
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

        payload = """
<function=typed_tool>
<parameter=name>
Chen
</parameter>
<parameter=age>
29
</parameter>
<parameter=height>
1.75
</parameter>
<parameter=active>
true
</parameter>
<parameter=meta>
{"city":"Houston"}
</parameter>
<parameter=scores>
[98,87]
</parameter>
<parameter=misc>
null
</parameter>
</function>
""".strip()

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

    def test_streamed_arguments_match_complete_parse_for_quoted_string_value(self):
        parser = Qwen3CoderToolParser()
        payload = '<function=find_user><parameter=name>"Chen"</parameter></function>'

        streamed_arguments = _stream_tool_arguments(
            parser,
            ['<function=find_user>', '<parameter=name>', '"Chen"', '</parameter>', '</function>'],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments

    def test_streamed_arguments_match_complete_parse_for_newline_escaped_quoted_string_value(self):
        parser = Qwen3CoderToolParser()
        payload = r'<function=find_user><parameter=name>"A\nB"</parameter></function>'

        streamed_arguments, per_chunk = _stream_tool_arguments_by_chunk(
            parser,
            ['<function=find_user>', '<parameter=name>', '"A\\', 'nB"', '</parameter></function>'],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert per_chunk[2] == ''
        assert per_chunk[3] == ''
        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments

    def test_streamed_arguments_match_complete_parse_for_quote_escaped_quoted_string_value(self):
        parser = Qwen3CoderToolParser()
        payload = r'<function=find_user><parameter=name>"A\"B"</parameter></function>'

        streamed_arguments = _stream_tool_arguments(
            parser,
            ['<function=find_user>', '<parameter=name>', '"A\\', '"B"', '</parameter></function>'],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments

    def test_streamed_arguments_match_complete_parse_for_invalid_integer_value(self):
        parser = Qwen3CoderToolParser()
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
        payload = '<function=typed_tool><parameter=age>abc</parameter></function>'

        streamed_arguments = _stream_tool_arguments(
            parser,
            ['<function=typed_tool>', '<parameter=age>', 'abc', '</parameter>', '</function>'],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments

    def test_streamed_arguments_match_complete_parse_for_invalid_integer_after_numeric_prefix(self):
        parser = Qwen3CoderToolParser()
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
        payload = '<function=typed_tool><parameter=age>2a</parameter></function>'

        streamed_arguments, per_chunk = _stream_tool_arguments_by_chunk(
            parser,
            ['<function=typed_tool>', '<parameter=age>', '2', 'a', '</parameter></function>'],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert per_chunk[2] == ''
        assert per_chunk[3] == ''
        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments
        assert json.loads(streamed_arguments) == {'age': '2a'}

    def test_streamed_arguments_match_complete_parse_when_next_param_starts_with_previous_close(self):
        parser = Qwen3CoderToolParser()
        payload = '<function=two_args><parameter=a>one</parameter><parameter=b>two</parameter></function>'

        streamed_arguments = _stream_tool_arguments(
            parser,
            [
                '<function=two_args>',
                '<parameter=a>',
                'one',
                '</parameter><parameter=b>two',
                '</parameter></function>',
            ],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments

    def test_streamed_arguments_match_complete_parse_when_close_chunk_has_value_tail(self):
        parser = Qwen3CoderToolParser()
        payload = '<function=f><parameter=a>San Francisco</parameter></function>'

        streamed_arguments = _stream_tool_arguments(
            parser,
            ['<function=f>', '<parameter=a>San ', 'Francisco</parameter></function>'],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments

    def test_streamed_arguments_match_complete_parse_for_unquoted_newline_value(self):
        parser = Qwen3CoderToolParser()
        payload = '<function=f><parameter=a>A\nB</parameter></function>'

        streamed_arguments = _stream_tool_arguments(
            parser,
            ['<function=f>', '<parameter=a>A\n', 'B</parameter></function>'],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments

    def test_streamed_arguments_match_complete_parse_for_unquoted_quote_value(self):
        parser = Qwen3CoderToolParser()
        payload = '<function=f><parameter=a>A"B</parameter></function>'

        streamed_arguments = _stream_tool_arguments(
            parser,
            ['<function=f>', '<parameter=a>A"', 'B</parameter></function>'],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments

    def test_streamed_arguments_match_complete_parse_when_value_contains_parameter_like_text(self):
        parser = Qwen3CoderToolParser()
        payload = '<function=f><parameter=a>foo <parameter=bar> baz</parameter></function>'

        streamed_arguments = _stream_tool_arguments(
            parser,
            ['<function=f>', '<parameter=a>foo ', '<parameter=bar> baz', '</parameter></function>'],
        )
        complete_tool_call = parser.parse_tool_call_complete(payload)

        assert complete_tool_call is not None
        assert streamed_arguments == complete_tool_call.function.arguments
