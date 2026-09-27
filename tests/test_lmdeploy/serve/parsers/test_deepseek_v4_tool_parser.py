import json

from lmdeploy.serve.parsers.tool_parser import DeepSeekV4ToolParser

from .helpers import feed_tool_chunks, tool_arguments


def test_v4_markers_stream_one_complete_call():
    parser = DeepSeekV4ToolParser()
    token = parser.dsml_token
    chunks = [
        f'<{token}invoke name="search">',
        f'<{token}parameter name="query" string="true">Deep',
        f'Seek</{token}parameter>',
        f'</{token}invoke>',
        f'</{token}tool_c',
        'alls>',
    ]

    pending, deltas, _ = feed_tool_chunks(parser, chunks)

    assert pending == ''
    assert parser.block_closed
    assert [delta.function.name for delta in deltas if delta.function.name] == ['search']
    assert json.loads(tool_arguments(deltas)) == {'query': 'DeepSeek'}
