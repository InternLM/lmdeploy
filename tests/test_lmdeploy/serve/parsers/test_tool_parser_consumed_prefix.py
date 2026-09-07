# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from collections import defaultdict

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.tool_parser import (
    DeepSeekV32ToolParser,
    Glm47ToolParser,
    Internlm2ToolParser,
    KimiK2ToolParser,
    Qwen3CoderToolParser,
    Qwen3ToolParser,
    ToolParserManager,
)

from .helpers import final_tool_calls


def _feed_chunks(parser, chunks):
    parser.begin_tool_block()
    pending = ''
    deltas = []
    max_pending = 0
    for chunk in chunks:
        pending += chunk
        while pending and not parser.block_closed:
            consumed = parser.feed_tool_block(pending, deltas, final=False)
            if not consumed:
                break
            pending = pending[consumed:]
        max_pending = max(max_pending, len(pending))
    return pending, deltas, max_pending


def _arguments(deltas):
    return ''.join(
        delta.function.arguments or ''
        for delta in deltas
        if delta.function is not None
    )


def _final_streamed_call(parser, payload):
    parser.begin_tool_block()
    deltas = []
    consumed = parser.feed_tool_block(payload, deltas, final=True)
    name = next(
        (
            delta.function.name
            for delta in deltas
            if delta.function is not None and delta.function.name is not None
        ),
        None,
    )
    return consumed, name, _arguments(deltas)


def test_json_arguments_before_name_stream_in_source_order_and_preserve_duplicates():
    parser = Qwen3ToolParser()
    raw_arguments = '{"a":"one","a":"two"}'
    pending, deltas, _ = _feed_chunks(
        parser,
        [
            '{"arguments":' + raw_arguments,
            ',"name":"f"}',
            '</tool_call>after',
        ],
    )

    assert pending == 'after'
    assert parser.block_closed
    assert _arguments(deltas) == raw_arguments

    first = deltas[0]
    assert first.id is not None
    assert first.type == 'function'
    assert first.function.name is None
    assert first.function.arguments == raw_arguments
    name_delta = next(delta for delta in deltas if delta.function and delta.function.name)
    assert name_delta.index == first.index
    assert name_delta.id is None
    assert name_delta.function.name == 'f'

    complete = final_tool_calls(
        Qwen3ToolParser(), '{"arguments":' + raw_arguments + ',"name":"f"}')[0]
    assert complete.function.arguments == raw_arguments


@pytest.mark.parametrize(
    ('parser_cls', 'payload'),
    [
        (
            Qwen3CoderToolParser,
            '<function=f><parameter=a>partial',
        ),
        (
            Glm47ToolParser,
            'f<arg_key>a</arg_key><arg_value>partial',
        ),
    ],
)
def test_xml_complete_reuses_final_streaming_semantics_for_incomplete_argument(parser_cls, payload):
    consumed, streamed_name, streamed_arguments = _final_streamed_call(parser_cls(), payload)
    complete = final_tool_calls(parser_cls(), payload)[0]

    assert consumed == len(payload)
    assert complete.function.name == streamed_name
    assert complete.function.arguments == streamed_arguments


def test_complete_and_streaming_drop_malformed_tool_block_without_name():
    parser_cls = ResponseParserManager.get('default')
    old_reasoning_cls = parser_cls.reasoning_parser_cls
    old_tool_cls = parser_cls.tool_parser_cls
    text = '<tool_call>{"arguments":{"x":1}}</tool_call>after'
    request = ChatCompletionRequest(
        model='test',
        messages=[],
        tools=[{'type': 'function', 'function': {'name': 'f'}}],
        tool_choice='auto',
    )
    try:
        parser_cls.reasoning_parser_cls = None
        parser_cls.tool_parser_cls = ToolParserManager.get('qwen3')

        stream_parser = parser_cls(request.model_copy(update={'stream': True}))
        streamed = stream_parser.stream_chunk(text, [])
        streamed_content = ''.join(message.content or '' for message, _ in streamed)
        streamed_calls = [call for message, _ in streamed for call in message.tool_calls or []]

        complete_parser = parser_cls(request.model_copy(update={'stream': False}))
        complete_content, complete_calls, reasoning = complete_parser.parse_complete(text)

        assert streamed_content == 'after'
        assert streamed_calls == []
        assert complete_content == streamed_content
        assert complete_calls is None
        assert reasoning is None
    finally:
        parser_cls.reasoning_parser_cls = old_reasoning_cls
        parser_cls.tool_parser_cls = old_tool_cls


@pytest.mark.parametrize('separator', ['\n', '\n\n'])
@pytest.mark.parametrize(
    ('tool_parser_name', 'first_call', 'second_call'),
    [
        (
            'qwen3',
            '<tool_call>{"name":"f","arguments":{"x":1}}</tool_call>',
            '<tool_call>{"name":"g","arguments":{"x":2}}</tool_call>',
        ),
        (
            'qwen3coder',
            '<tool_call><function=f></function></tool_call>',
            '<tool_call><function=g></function></tool_call>',
        ),
        (
            'glm47',
            '<tool_call>f</tool_call>',
            '<tool_call>g</tool_call>',
        ),
    ],
)
def test_response_parser_drops_newlines_only_between_tool_blocks(
    separator,
    tool_parser_name,
    first_call,
    second_call,
):
    parser_cls = ResponseParserManager.get('default')
    old_reasoning_cls = parser_cls.reasoning_parser_cls
    old_tool_cls = parser_cls.tool_parser_cls
    request = ChatCompletionRequest(
        model='test',
        messages=[],
        tools=[
            {'type': 'function', 'function': {'name': 'f'}},
            {'type': 'function', 'function': {'name': 'g'}},
        ],
        tool_choice='auto',
    )
    try:
        parser_cls.reasoning_parser_cls = None
        parser_cls.tool_parser_cls = ToolParserManager.get(tool_parser_name)

        complete_parser = parser_cls(request.model_copy(update={'stream': False}))
        complete_content, complete_calls, reasoning = complete_parser.parse_complete(
            first_call + separator + second_call)

        stream_parser = parser_cls(request.model_copy(update={'stream': True}))
        streamed = []
        for chunk in (first_call, separator, second_call[:5], second_call[5:]):
            streamed.extend(stream_parser.stream_chunk(chunk, []))
        streamed_content = ''.join(message.content or '' for message, _ in streamed)
        streamed_names = [
            call.function.name
            for message, _ in streamed
            for call in message.tool_calls or []
            if call.function.name is not None
        ]

        assert complete_content is None
        assert reasoning is None
        assert complete_calls is not None
        assert [call.function.name for call in complete_calls] == ['f', 'g']
        assert streamed_content == ''
        assert streamed_names == ['f', 'g']
    finally:
        parser_cls.reasoning_parser_cls = old_reasoning_cls
        parser_cls.tool_parser_cls = old_tool_cls


@pytest.mark.parametrize('tail', ['\nafter', '\n\nafter', '\n', '\n<tool'])
def test_response_parser_preserves_tool_newlines_not_followed_by_another_tool(tail):
    parser_cls = ResponseParserManager.get('default')
    old_reasoning_cls = parser_cls.reasoning_parser_cls
    old_tool_cls = parser_cls.tool_parser_cls
    call = '<tool_call>{"name":"f","arguments":{}}</tool_call>'
    request = ChatCompletionRequest(
        model='test',
        messages=[],
        tools=[{'type': 'function', 'function': {'name': 'f'}}],
        tool_choice='auto',
    )
    try:
        parser_cls.reasoning_parser_cls = None
        parser_cls.tool_parser_cls = ToolParserManager.get('qwen3')

        complete_parser = parser_cls(request.model_copy(update={'stream': False}))
        complete_content, _, _ = complete_parser.parse_complete(call + tail)

        stream_parser = parser_cls(request.model_copy(update={'stream': True}))
        streamed = stream_parser.stream_chunk(call, [])
        streamed.extend(stream_parser.stream_chunk(tail[:1], []))
        streamed.extend(stream_parser.stream_chunk(tail[1:], [], final=True))
        streamed_content = ''.join(message.content or '' for message, _ in streamed)

        assert complete_content == tail
        assert streamed_content == tail
    finally:
        parser_cls.reasoning_parser_cls = old_reasoning_cls
        parser_cls.tool_parser_cls = old_tool_cls


@pytest.mark.parametrize(
    'unknown_chunks',
    [
        (
            '<tool_call>{"arguments":{"secret":1},',
            '"name":"missing"}</tool_call>',
        ),
        (
            '<tool_call>{"name":"missing",',
            '"arguments":{"secret":1}}</tool_call>',
        ),
    ],
)
def test_response_parser_drops_unknown_name_and_keeps_next_valid_call(unknown_chunks):
    parser_cls = ResponseParserManager.get('default')
    old_reasoning_cls = parser_cls.reasoning_parser_cls
    old_tool_cls = parser_cls.tool_parser_cls
    raw_text = (
        '<tool_call>{"arguments":{"secret":1},"name":"missing"}</tool_call>'
        '\n'
        '<tool_call>{"name":"allowed","arguments":{"ok":1}}</tool_call>'
    )
    request = ChatCompletionRequest(
        model='test',
        messages=[],
        tools=[{'type': 'function', 'function': {'name': 'allowed'}}],
        tool_choice='auto',
        stream=True,
    )
    try:
        parser_cls.reasoning_parser_cls = None
        parser_cls.tool_parser_cls = ToolParserManager.get('qwen3')

        stream_parser = parser_cls(request)
        emitted = []
        for chunk in (*unknown_chunks, '\n', '<tool_call>{"name":"allowed","arguments":{"ok":1}}</tool_call>'):
            emitted.extend(stream_parser.stream_chunk(chunk, []))
        streamed_calls = [
            call
            for message, _ in emitted
            for call in message.tool_calls or []
        ]
        assert {call.index for call in streamed_calls} == {0}
        assert [call.function.name for call in streamed_calls if call.function.name] == ['allowed']
        assert ''.join(call.function.arguments or '' for call in streamed_calls) == '{"ok":1}'

        complete_parser = parser_cls(request.model_copy(update={'stream': False}))
        content, complete_calls, reasoning = complete_parser.parse_complete(raw_text)
        assert content is None
        assert reasoning is None
        assert complete_calls is not None
        assert [call.function.name for call in complete_calls] == ['allowed']
        assert complete_calls[0].function.arguments == '{"ok":1}'
    finally:
        parser_cls.reasoning_parser_cls = old_reasoning_cls
        parser_cls.tool_parser_cls = old_tool_cls


@pytest.mark.parametrize(
    ('payload', 'expected_name', 'expected_arguments'),
    [
        ('{"name":"f","name":"g","arguments":{}}', 'f', '{}'),
        ('{"name":"f","arguments":{},"arguments":{}}', 'f', '{}{}'),
        ('{"name":"f","arguments":{},"parameters":{}}', 'f', '{}'),
    ],
)
def test_json_streams_duplicate_envelope_values_without_validating_them(
    payload,
    expected_name,
    expected_arguments,
):
    parser = Qwen3ToolParser()
    pending, deltas, _ = _feed_chunks(parser, [payload + '</tool_call>'])

    assert pending == ''
    assert parser.block_closed
    assert _arguments(deltas) == expected_arguments
    assert next(delta.function.name for delta in deltas if delta.function.name is not None) == expected_name
    complete = final_tool_calls(Qwen3ToolParser(), payload)[0]
    assert complete.function.name == expected_name
    assert complete.function.arguments == expected_arguments


def test_json_streams_malformed_arguments_up_to_protocol_boundary():
    parser = Qwen3ToolParser()
    pending, deltas, _ = _feed_chunks(
        parser,
        [
            '{"name":"f","arguments":{"x":truX',
            '</tool_call>tail',
        ],
    )

    assert pending == 'tail'
    assert parser.block_closed
    assert _arguments(deltas) == '{"x":truX'
    assert next(delta.function.name for delta in deltas if delta.function.name is not None) == 'f'


def test_json_complete_keeps_malformed_arguments_up_to_protocol_boundary():
    parser_cls = ResponseParserManager.get('default')
    old_reasoning_cls = parser_cls.reasoning_parser_cls
    old_tool_cls = parser_cls.tool_parser_cls
    try:
        parser_cls.reasoning_parser_cls = None
        parser_cls.tool_parser_cls = ToolParserManager.get('qwen3')
        parser = parser_cls(
            ChatCompletionRequest(
                model='test',
                messages=[],
                tools=[{'type': 'function', 'function': {'name': 'f'}}],
                tool_choice='auto',
            ))

        content, tool_calls, reasoning = parser.parse_complete(
            '<tool_call>{"name":"f","arguments":{"x":truX</tool_call>after')

        assert content == 'after'
        assert reasoning is None
        assert tool_calls is not None
        assert tool_calls[0].function.name == 'f'
        assert tool_calls[0].function.arguments == '{"x":truX'
    finally:
        parser_cls.reasoning_parser_cls = old_reasoning_cls
        parser_cls.tool_parser_cls = old_tool_cls


def test_json_keeps_protocol_marker_inside_argument_string():
    parser = Qwen3ToolParser()
    raw_arguments = '{"text":"inside </tool_call> marker"}'
    pending, deltas, _ = _feed_chunks(
        parser,
        ['{"name":"f","arguments":' + raw_arguments + '}</tool_call>tail'],
    )

    assert pending == 'tail'
    assert parser.block_closed
    assert _arguments(deltas) == raw_arguments

    complete_parser = Qwen3ToolParser()
    complete_parser.begin_tool_block()
    complete_deltas = []
    complete_text = '{"name":"f","arguments":' + raw_arguments + '}</tool_call>tail'
    complete_end = complete_parser.feed_tool_block(complete_text, complete_deltas, final=True)
    complete_calls = complete_parser.build_tool_calls(complete_deltas)
    assert complete_end == len('{"name":"f","arguments":' + raw_arguments + '}</tool_call>')
    assert complete_calls[0].function.arguments == raw_arguments


def test_each_json_protocol_ignores_the_other_argument_field():
    arguments_payload = '{"name":"f","arguments":{"x":1}}'
    parameters_payload = '{"name":"f","parameters":{"x":1}}'

    assert final_tool_calls(Qwen3ToolParser(), arguments_payload)
    assert final_tool_calls(Qwen3ToolParser(), parameters_payload)[0].function.arguments == '{}'
    assert final_tool_calls(Internlm2ToolParser(), arguments_payload)[0].function.arguments == '{}'
    assert final_tool_calls(Internlm2ToolParser(), parameters_payload)


def test_qwen_xml_preserves_duplicate_parameters_and_only_protocol_newlines():
    payload = (
        '<function=f>'
        '<parameter=a>\n one \n\n</parameter>'
        '<parameter=a>two</parameter>'
        '</function>'
    )
    expected = '{"a": " one \\n", "a": "two"}'
    parser = Qwen3CoderToolParser()
    pending, deltas, _ = _feed_chunks(
        parser,
        [payload[:31], payload[31:] + '</tool_call>tail'],
    )

    assert pending == 'tail'
    assert _arguments(deltas) == expected
    complete = final_tool_calls(Qwen3CoderToolParser(), payload)[0]
    assert complete.function.arguments == expected


def test_glm_preserves_raw_string_whitespace_and_duplicate_parameters():
    payload = (
        'f'
        '<arg_key>a</arg_key><arg_value>  one  </arg_value>'
        '<arg_key>a</arg_key><arg_value>two</arg_value>'
    )
    expected = '{"a": "  one  ", "a": "two"}'
    parser = Glm47ToolParser()
    pending, deltas, _ = _feed_chunks(parser, [payload, '</tool_call>tail'])

    assert pending == 'tail'
    assert _arguments(deltas) == expected
    complete = final_tool_calls(Glm47ToolParser(), payload)[0]
    assert complete.function.arguments == expected


@pytest.mark.parametrize(
    ('parser_cls', 'payload'),
    [
        (
            Qwen3CoderToolParser,
            '<function=f><parameter=a>one</parameter><parameter=b>two</parameter></function>',
        ),
        (
            Glm47ToolParser,
            'f<arg_key>a</arg_key><arg_value>one</arg_value>'
            '<arg_key>b</arg_key><arg_value>two</arg_value>',
        ),
    ],
)
def test_xml_consumes_multiple_arguments_and_outer_close_in_one_chunk(parser_cls, payload):
    parser = parser_cls()
    pending, deltas, _ = _feed_chunks(parser, [payload + '</tool_call>tail'])

    assert pending == 'tail'
    assert parser.block_closed
    assert _arguments(deltas) == '{"a": "one", "b": "two"}'


def test_xml_outer_close_text_inside_parameter_value_is_not_a_block_boundary():
    payload = (
        'f<arg_key>a</arg_key>'
        '<arg_value>before </tool_call> after</arg_value>'
    )
    parser = Glm47ToolParser()
    pending, deltas, _ = _feed_chunks(parser, [payload + '</tool_call>tail'])

    assert pending == 'tail'
    assert _arguments(deltas) == '{"a": "before </tool_call> after"}'


def test_dsml_preserves_duplicate_parameters_in_complete_and_streaming_paths():
    parser = DeepSeekV32ToolParser()
    token = parser.dsml_token
    payload = (
        f'<{token}invoke name="f">\n'
        f'<{token}parameter name="a" string="true">one</{token}parameter>\n'
        f'<{token}parameter name="a" string="true">two</{token}parameter>\n'
        f'</{token}invoke>'
    )
    expected = '{"a": "one", "a": "two"}'
    pending, deltas, _ = _feed_chunks(
        parser,
        [payload[:23], payload[23:], parser.get_tool_close_tag() + 'tail'],
    )

    assert pending == 'tail'
    assert _arguments(deltas) == expected
    complete = final_tool_calls(DeepSeekV32ToolParser(), payload)
    assert complete[0].function.arguments == expected


def test_multi_call_complete_adapter_preserves_call_order():
    kimi = KimiK2ToolParser()
    kimi_payload = (
        f'{kimi.call_begin}functions.f:0{kimi.argument_begin}{{}}{kimi.call_end}'
        f'{kimi.call_begin}functions.g:1{kimi.argument_begin}{{}}{kimi.call_end}'
    )
    dsml = DeepSeekV32ToolParser()
    token = dsml.dsml_token
    dsml_payload = (
        f'<{token}invoke name="f"></{token}invoke>'
        f'<{token}invoke name="g"></{token}invoke>'
    )

    for parser, payload in ((kimi, kimi_payload), (dsml, dsml_payload)):
        calls = final_tool_calls(parser, payload)
        assert [call.function.name for call in calls] == ['f', 'g']
        assert [call.function.arguments for call in calls] == ['{}', '{}']


def test_dsml_complete_emits_incomplete_string_argument():
    parser = DeepSeekV32ToolParser()
    token = parser.dsml_token
    payload = f'<{token}invoke name="f"><{token}parameter name="a" string="true">partial'

    call = final_tool_calls(parser, payload)[0]

    assert call.function.name == 'f'
    assert call.function.arguments == '{"a": "partial'


def test_kimi_streams_arguments_without_accumulating_payload_and_handles_atomic_markers():
    parser = KimiK2ToolParser()
    raw_arguments = '{"text":"inside <|tool_calls_section_end|> marker","n":1}'
    chunks = [
        parser.call_begin,
        'functions.f:0',
        parser.argument_begin,
        *raw_arguments,
        parser.call_end,
        parser.section_end + 'tail',
    ]
    pending, deltas, max_pending = _feed_chunks(parser, chunks)

    assert pending == 'tail'
    assert parser.block_closed
    assert _arguments(deltas) == raw_arguments
    assert not hasattr(parser, '_tool_payload')
    assert max_pending <= len(parser.argument_begin) + len('functions.f:0')
    complete = final_tool_calls(KimiK2ToolParser(), ''.join(chunks[:-1]))
    assert complete[0].function.arguments == raw_arguments


def test_final_missing_tool_close_does_not_delay_emitted_results():
    parser = Qwen3ToolParser()
    parser.begin_tool_block()
    text = '{"name":"f","arguments":{}}'
    deltas = []

    consumed = parser.feed_tool_block(text, deltas, final=True)

    assert consumed == len(text)
    assert parser.block_closed
    assert _arguments(deltas) == '{}'


def test_response_parser_emits_trailing_text_after_atomic_tool_close():
    parser_cls = ResponseParserManager.get('default')
    old_reasoning_cls = parser_cls.reasoning_parser_cls
    old_tool_cls = parser_cls.tool_parser_cls
    try:
        parser_cls.reasoning_parser_cls = None
        parser_cls.tool_parser_cls = ToolParserManager.get('qwen3')
        parser = parser_cls(
            ChatCompletionRequest(
                model='test',
                messages=[],
                tools=[{'type': 'function', 'function': {'name': 'get_weather'}}],
                tool_choice='auto',
                stream=True,
            ))
        chunks = [
            '<tool_call>{"arguments":{"city":"Bei',
            'jing"},"name":"get_weather"}',
            '</tool_call>after',
        ]
        emitted = []
        for chunk in chunks:
            emitted.extend(parser.stream_chunk(chunk, []))

        arguments_by_index = defaultdict(str)
        names = {}
        content = []
        ordered_calls = []
        for message, _ in emitted:
            content.append(message.content or '')
            for call in message.tool_calls or []:
                ordered_calls.append(call)
                if call.function and call.function.name:
                    names[call.index] = call.function.name
                if call.function and call.function.arguments:
                    arguments_by_index[call.index] += call.function.arguments

        assert ''.join(content) == 'after'
        assert names == {0: 'get_weather'}
        assert arguments_by_index == {0: '{"city":"Beijing"}'}
        assert ordered_calls[0].function.arguments
        assert ordered_calls[0].function.name is None
        assert ordered_calls[-1].function.name == 'get_weather'
        assert parser._pending == ''
        assert not hasattr(parser, '_accumulated_chunks')
    finally:
        parser_cls.reasoning_parser_cls = old_reasoning_cls
        parser_cls.tool_parser_cls = old_tool_cls
