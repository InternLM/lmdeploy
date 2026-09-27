import json

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager
from lmdeploy.serve.parsers.tool_parser.qwen3coder_tool_parser import Qwen3CoderToolParser

from .helpers import (
    feed_tool_chunks,
    feed_tool_payload,
    final_tool_call,
    flatten_stream_deltas,
    stream_tool_arguments,
    tool_arguments,
)

MODEL_ID = 'Qwen/Qwen3.5-35B-A3B'
PARSER_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': name,
        },
    } for name in ('get_current_temperature', 'get_current_weather')
]


def _build_response_parser():
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('default')
    cls.tool_parser_cls = ToolParserManager.get('qwen3coder')

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
    ('', []),
]


class TestQwen3_5ResponseParserStreaming:
    """Integration test for ResponseParser.stream_chunk with Qwen3.5 Coder
    parsers."""

    def test_stream_chunk_matches_reference(self):
        response_parser = _build_response_parser()
        for delta_text, expected_events in REFERENCE_CHUNKS:
            actual_events = flatten_stream_deltas(
                response_parser.stream_chunk(delta_text=delta_text, delta_token_ids=[]))
            assert actual_events == expected_events

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

    def test_final_tool_payload_treats_params_as_strings(self):
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

        tool_call = final_tool_call(parser, payload)

        assert tool_call is not None
        assert tool_call.function.name == 'find_user_id_by_name_zip'
        assert json.loads(tool_call.function.arguments) == {
            'first_name': 'Chen',
            'last_name': 'Johnson',
            'zip': '77004',
        }

    def test_duplicate_parameters_preserve_protocol_newline_semantics(self):
        payload = (
            '<function=f>'
            '<parameter=a>\n one \n\n</parameter>'
            '<parameter=a>two</parameter>'
            '</function>'
        )
        expected = '{"a": " one \\n", "a": "two"}'

        pending, deltas, _ = feed_tool_chunks(
            Qwen3CoderToolParser(),
            [payload[:31], payload[31:] + '</tool_call>tail'],
        )
        complete = final_tool_call(Qwen3CoderToolParser(), payload)

        assert pending == 'tail'
        assert tool_arguments(deltas) == expected
        assert complete.function.arguments == expected

    @pytest.mark.parametrize(
        ('value', 'value_chunks'),
        [
            ('"Chen"', ['"Chen"']),
            (r'"A\nB"', ['"A\\', 'nB"']),
            (r'"A\"B"', ['"A\\', '"B"']),
        ],
        ids=['quoted', 'escaped-newline', 'escaped-quote'],
    )
    def test_quoted_parameter_matches_complete_parse(self, value, value_chunks):
        parser = Qwen3CoderToolParser()
        payload = f'<function=find_user><parameter=name>{value}</parameter></function>'
        chunks = [
            '<function=find_user>',
            '<parameter=name>',
            *value_chunks,
            '</parameter></function>',
        ]

        streamed_arguments = stream_tool_arguments(parser, chunks)
        complete = final_tool_call(parser, payload)

        assert streamed_arguments == complete.function.arguments

    def test_megabyte_string_is_streamed_without_buffering(self):
        parser = Qwen3CoderToolParser()
        parser.begin_tool_block()
        pending, deltas = feed_tool_payload(
            parser,
            '<function=write_file><parameter=content>',
            final=False,
        )
        value_chunk = 'x' * 1024
        for _ in range(1024):
            pending, chunk_deltas = feed_tool_payload(parser, pending + value_chunk, final=False)
            deltas.extend(chunk_deltas)

        assert parser._arg_state.buffered_parts == []
        assert len(pending) < len(parser.param_suffix)

        pending, chunk_deltas = feed_tool_payload(
            parser,
            pending + '</parameter></function></tool_call>',
            final=True,
        )
        deltas.extend(chunk_deltas)

        assert pending == ''
        assert json.loads(tool_arguments(deltas)) == {'content': value_chunk * 1024}

    def test_final_chunk_closes_arguments_without_function_terminator(self):
        parser = Qwen3CoderToolParser()
        parser.begin_tool_block()
        payload = '<function=f><parameter=a>one</parameter>'
        deltas = []

        consumed = parser.feed_tool_block(payload, deltas, final=True)

        assert consumed == len(payload)
        assert parser.block_closed
        assert tool_arguments(deltas) == '{"a": "one"}'
