# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from lmdeploy.serve.parsers.tool_parser import ToolParserManager

from .helpers import feed_tool_payload, tool_arguments


@pytest.mark.parametrize('parser_name', ['qwen3', 'internlm', 'llama3'])
@pytest.mark.parametrize('arguments_first', [False, True])
def test_split_function_name_releases_arguments_only_after_name(parser_name, arguments_first):
    """An incomplete escaped name must retain its input and any earlier
    arguments."""
    parser = ToolParserManager.get(parser_name)()
    parser.begin_tool_block()
    arguments = '{"value":1,"value":2}'
    field = f'"{parser.argument_field}":{arguments}'
    prefix = f'{{{field},"name":' if arguments_first else '{"name":'
    suffix = '}' if arguments_first else f',{field}}}'

    pending, deltas = feed_tool_payload(parser, prefix + ' \n"get\\u0', final=False)
    assert pending == '"get\\u0'
    assert deltas == []

    pending, deltas = feed_tool_payload(parser, pending + '05fweather', final=False)
    assert pending == '"get\\u005fweather'
    assert deltas == []

    text = pending + '"' + suffix + (parser.get_tool_close_tag() or '') + ' tail'
    pending, deltas = feed_tool_payload(parser, text, final=False)

    assert pending == ' tail'
    assert parser.block_closed
    assert [delta.function.name for delta in deltas if delta.function.name is not None] == ['get_weather']
    assert deltas[0].function.arguments is None
    assert tool_arguments(deltas) == arguments


@pytest.mark.parametrize('parser_name', ['qwen3', 'internlm', 'llama3'])
def test_empty_envelope_does_not_leak_arguments_into_next_call(parser_name):
    """Closing an empty envelope must leave trailing input and allow a fresh
    call."""
    parser = ToolParserManager.get(parser_name)()
    close_tag = parser.get_tool_close_tag() or ''
    parser.begin_tool_block()

    pending, deltas = feed_tool_payload(parser, '{}' + close_tag + ' tail', final=False)
    assert pending == ' tail'
    assert parser.block_closed
    assert deltas == []

    parser.begin_tool_block()
    pending, deltas = feed_tool_payload(parser, '{"name":"f"}' + close_tag, final=True)

    assert pending == ''
    assert parser.block_closed
    assert [delta.function.name for delta in deltas if delta.function.name is not None] == ['f']
    assert tool_arguments(deltas) == '{}'
