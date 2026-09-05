import json
from collections import defaultdict

import pytest

from lmdeploy.serve.parsers.tool_parser import (
    DeepSeekV4ToolParser,
    DeepSeekV32ToolParser,
    Glm47ToolParser,
    KimiK2ToolParser,
    Qwen3CoderToolParser,
)


def _arguments_from(calls):
    return ''.join(
        call.function.arguments or ''
        for call in calls
        if call.function is not None
    )


def test_qwen_parameter_markers_follow_token_aligned_boundaries():
    parser = Qwen3CoderToolParser()
    parser.start_tool_call()
    try:
        chunks = [
            '<',
            'function',
            '=f',
            '>',
            '<',
            'parameter',
            '=p',
            '>',
            'value',
            '</',
            'parameter',
            '>',
            '</',
            'function',
            '>',
        ]
        per_chunk = [parser.decode_tool_incremental(chunk, final=False) for chunk in chunks]
    finally:
        parser.finish_tool_call()

    assert per_chunk[3][0].function.name == 'f'
    assert all(not calls for calls in per_chunk[4:8])
    assert _arguments_from(per_chunk[8]) == '{"p": "value'
    assert all(not calls for calls in per_chunk[9:11])
    assert _arguments_from(per_chunk[11]) == '"'
    assert _arguments_from(per_chunk[14]) == '}'


@pytest.mark.parametrize(
    ('parser_cls', 'prefix', 'tail', 'tail_final', 'close_tag'),
    [
        (
            Qwen3CoderToolParser,
            '<function=write_file><parameter=content>',
            '</parameter></function>',
            False,
            '</parameter>',
        ),
        (
            Glm47ToolParser,
            'write_file<arg_key>content</arg_key><arg_value>',
            '</arg_value>',
            True,
            '</arg_value>',
        ),
    ],
)
def test_streamed_megabyte_string_does_not_remain_buffered(parser_cls, prefix, tail, tail_final, close_tag):
    parser = parser_cls()
    parser.start_tool_call()
    fragments = []
    value_chunk = 'x' * 1024
    try:
        fragments.append(_arguments_from(parser.decode_tool_incremental(prefix, final=False)))
        for _ in range(1024):
            fragments.append(_arguments_from(parser.decode_tool_incremental(value_chunk, final=False)))

        assert parser._arg_state.buffered_parts == []
        assert parser._arg_state.pending_ws == ''
        assert len(''.join(parser._payload_parts)) <= len(close_tag) - 1

        fragments.append(_arguments_from(parser.decode_tool_incremental(tail, final=tail_final)))
    finally:
        parser.finish_tool_call()

    assert json.loads(''.join(fragments)) == {'content': value_chunk * 1024}


@pytest.mark.parametrize(
    ('parser_cls', 'payloads'),
    [
        (
            Qwen3CoderToolParser,
            [
                '<function=first><parameter=value>one</parameter></function>',
                '<function=second><parameter=value>two</parameter></function>',
            ],
        ),
        (
            Glm47ToolParser,
            [
                'first<arg_key>value</arg_key><arg_value>one</arg_value>',
                'second<arg_key>value</arg_key><arg_value>two</arg_value>',
            ],
        ),
        (
            KimiK2ToolParser,
            [
                '<|tool_call_begin|>functions.first:0<|tool_call_argument_begin|>{"value":"one"}<|tool_call_end|>',
                '<|tool_call_begin|>functions.second:1<|tool_call_argument_begin|>{"value":"two"}<|tool_call_end|>',
            ],
        ),
    ],
)
def test_tool_parser_lifecycle_resets_stream_state(parser_cls, payloads):
    parser = parser_cls()
    parsed = []
    for payload in payloads:
        parser.start_tool_call()
        calls = parser.decode_tool_incremental(payload, final=True)
        parsed.append((calls[0].index, calls[0].function.name, json.loads(_arguments_from(calls))))
        parser.finish_tool_call()

    assert parsed == [
        (0, 'first', {'value': 'one'}),
        (1, 'second', {'value': 'two'}),
    ]


@pytest.mark.parametrize('parser_cls', [DeepSeekV32ToolParser, DeepSeekV4ToolParser])
def test_dsml_streams_parameter_header_and_string_value_immediately(parser_cls):
    parser = parser_cls()
    token = parser.dsml_token
    parser.start_tool_call()
    try:
        chunks = [
            f'\n<{token}invoke name="search">\n',
            f'<{token}parameter name="query" string="true">',
            'DeepSeek ',
            '"streaming"',
            f'</{token}parameter>\n',
            f'</{token}invoke>\n',
        ]
        per_chunk = [parser.decode_tool_incremental(chunk, final=False) for chunk in chunks]
    finally:
        parser.finish_tool_call()

    assert per_chunk[0][0].function.name == 'search'
    assert _arguments_from(per_chunk[1]) == '{"query": "'
    assert _arguments_from(per_chunk[2]) == 'DeepSeek '
    assert _arguments_from(per_chunk[3]) == '\\"streaming\\"'
    assert _arguments_from(per_chunk[4]) == '"'
    assert _arguments_from(per_chunk[5]) == '}'
    assert json.loads(''.join(_arguments_from(calls) for calls in per_chunk)) == {
        'query': 'DeepSeek "streaming"'
    }


@pytest.mark.parametrize('parser_cls', [DeepSeekV32ToolParser, DeepSeekV4ToolParser])
def test_dsml_streams_non_string_json_and_multiple_invokes(parser_cls):
    parser = parser_cls()
    token = parser.dsml_token
    payload = (
        f'\n<{token}invoke name="rank">\n'
        f'<{token}parameter name="limit" string="false">12</{token}parameter>\n'
        f'<{token}parameter name="filters" string="false">{{"active":true}}</{token}parameter>\n'
        f'</{token}invoke>\n'
        f'<{token}invoke name="lookup">\n'
        f'<{token}parameter name="name" string="true">Ada</{token}parameter>\n'
        f'</{token}invoke>\n'
    )

    parser.start_tool_call()
    try:
        calls = []
        for char in payload:
            calls.extend(parser.decode_tool_incremental(char, final=False))
        calls.extend(parser.decode_tool_incremental('', final=True))
    finally:
        parser.finish_tool_call()

    names = [call for call in calls if call.function and call.function.name]
    assert [(call.index, call.function.name) for call in names] == [(0, 'rank'), (1, 'lookup')]
    assert names[0].id and names[1].id and names[0].id != names[1].id

    arguments_by_index = defaultdict(str)
    for call in calls:
        if call.function and call.function.arguments is not None:
            arguments_by_index[call.index] += call.function.arguments
    assert json.loads(arguments_by_index[0]) == {
        'limit': 12,
        'filters': {
            'active': True
        },
    }
    assert json.loads(arguments_by_index[1]) == {'name': 'Ada'}
