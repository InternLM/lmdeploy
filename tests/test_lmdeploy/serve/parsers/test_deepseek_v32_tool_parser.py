import json
from collections import defaultdict

from lmdeploy.serve.parsers.tool_parser import DeepSeekV32ToolParser

from .helpers import feed_tool_chunks, feed_tool_payload, final_tool_call, tool_arguments


def test_string_parameter_header_and_value_stream_immediately():
    parser = DeepSeekV32ToolParser()
    token = parser.dsml_token
    parser.begin_tool_block()
    reference = [
        (f'\n<{token}invoke name="search">\n', [('function', 'search', None)]),
        (f'<{token}parameter name="query" string="true">', [(None, None, '{"query": "')]),
        ('DeepSeek ', [(None, None, 'DeepSeek ')]),
        ('"streaming"', [(None, None, '\\"streaming\\"')]),
        (f'</{token}parameter>\n', [(None, None, '"')]),
        (f'</{token}invoke>\n', [(None, None, '}')]),
    ]
    pending = ''

    for chunk, expected in reference:
        pending, deltas = feed_tool_payload(parser, pending + chunk, final=False)
        actual = [
            (delta.type, delta.function.name, delta.function.arguments)
            for delta in deltas
        ]
        assert actual == expected


def test_non_string_json_and_multiple_invokes_follow_token_boundaries():
    parser = DeepSeekV32ToolParser()
    token = parser.dsml_token
    chunks = [
        '\n<',
        token,
        'inv',
        'oke name="rank">\n',
        '<',
        token,
        'parameter name="limit" string="false">',
        '12',
        '</',
        token,
        'parameter',
        '>\n<',
        token,
        'parameter name="filters" string="false">',
        '{"active":true}',
        '</',
        token,
        'parameter',
        '>\n</',
        token,
        'inv',
        'oke>\n<',
        token,
        'invoke name="lookup">\n',
        '<',
        token,
        'parameter name="name" string="true">',
        'Ada',
        '</',
        token,
        'parameter',
        '>\n</',
        token,
        'inv',
        'oke>\n</',
        token,
        'function_c',
        'alls',
        '>',
    ]

    pending, deltas, _ = feed_tool_chunks(parser, chunks)

    assert pending == ''
    assert parser.block_closed
    names = [delta for delta in deltas if delta.function and delta.function.name]
    assert [(delta.index, delta.function.name) for delta in names] == [(0, 'rank'), (1, 'lookup')]
    assert names[0].id and names[1].id and names[0].id != names[1].id
    arguments_by_index = defaultdict(str)
    for delta in deltas:
        if delta.function and delta.function.arguments is not None:
            arguments_by_index[delta.index] += delta.function.arguments
    assert json.loads(arguments_by_index[0]) == {
        'limit': 12,
        'filters': {
            'active': True
        },
    }
    assert json.loads(arguments_by_index[1]) == {'name': 'Ada'}


def test_duplicate_parameters_are_preserved_in_streaming_and_complete_paths():
    parser = DeepSeekV32ToolParser()
    token = parser.dsml_token
    payload = (
        f'<{token}invoke name="f">\n'
        f'<{token}parameter name="a" string="true">one</{token}parameter>\n'
        f'<{token}parameter name="a" string="true">two</{token}parameter>\n'
        f'</{token}invoke>'
    )
    expected = '{"a": "one", "a": "two"}'

    pending, deltas, _ = feed_tool_chunks(
        parser,
        [payload[:23], payload[23:], parser.get_tool_close_tag() + 'tail'],
    )
    complete = final_tool_call(DeepSeekV32ToolParser(), payload)

    assert pending == 'tail'
    assert tool_arguments(deltas) == expected
    assert complete.function.arguments == expected


def test_final_payload_emits_incomplete_string_argument():
    parser = DeepSeekV32ToolParser()
    token = parser.dsml_token
    payload = f'<{token}invoke name="f"><{token}parameter name="a" string="true">partial'

    call = final_tool_call(parser, payload)

    assert call.function.name == 'f'
    assert call.function.arguments == '{"a": "partial'
