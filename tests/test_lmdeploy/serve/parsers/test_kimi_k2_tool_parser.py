import json

from lmdeploy.serve.parsers.tool_parser import KimiK2ToolParser

from .helpers import feed_tool_chunks, feed_tool_payload, final_tool_calls, tool_arguments


def test_arguments_stream_immediately_without_growing_pending_payload():
    parser = KimiK2ToolParser()
    parser.begin_tool_block()
    reference = [
        (
            f'{parser.call_begin}functions.search:0{parser.argument_begin}{{"query":"Deep',
            [('function', 'search', None), (None, None, '{"query":"Deep')],
        ),
        ('Seek"', [(None, None, 'Seek"')]),
        ('}', [(None, None, '}')]),
        (parser.call_end, []),
    ]
    pending = ''

    for chunk, expected in reference:
        pending, deltas = feed_tool_payload(parser, pending + chunk, final=False)
        actual = [
            (delta.type, delta.function.name, delta.function.arguments)
            for delta in deltas
        ]
        assert actual == expected

    pending, deltas, max_pending = feed_tool_chunks(
        KimiK2ToolParser(),
        [
            parser.call_begin,
            'functions.f:0',
            parser.argument_begin,
            *'{"text":"inside <|tool_calls_section_end|> marker","n":1}',
            parser.call_end,
            parser.section_end + 'tail',
        ],
    )

    assert pending == 'tail'
    assert max_pending <= len(parser.argument_begin) + len('functions.f:0')
    assert json.loads(tool_arguments(deltas)) == {
        'text': 'inside <|tool_calls_section_end|> marker',
        'n': 1,
    }


def test_complete_payload_preserves_multiple_call_order():
    parser = KimiK2ToolParser()
    payload = (
        f'{parser.call_begin}functions.f:0{parser.argument_begin}{{}}{parser.call_end}'
        f'{parser.call_begin}functions.g:1{parser.argument_begin}{{}}{parser.call_end}'
    )

    calls = final_tool_calls(parser, payload)

    assert [call.function.name for call in calls] == ['f', 'g']
    assert [call.function.arguments for call in calls] == ['{}', '{}']
