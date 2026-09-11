import json

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import MiMoToolParser, ToolParserManager

from .helpers import final_tool_call, final_tool_calls, first_stream_delta

MODEL_ID = 'XiaomiMiMo/MiMo-V2-Flash'


def _make_request(*, stream: bool = False):
    return ChatCompletionRequest(
        model=MODEL_ID,
        messages=[],
        stream=stream,
        tools=[{
            'type': 'function',
            'function': {
                'name': 'run_command',
                'description': 'Run a command.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'command': {
                            'type': 'string'
                        },
                        'timeout': {
                            'type': 'integer'
                        },
                        'options': {
                            'type': 'object'
                        },
                    },
                },
            },
        }],
        tool_choice='auto',
        chat_template_kwargs={'enable_thinking': True},
    )


def _make_response_parser(*, stream: bool = False):
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('default')
    cls.tool_parser_cls = ToolParserManager.get('mimo')
    return cls(request=_make_request(stream=stream))


def test_mimo_complete_preserves_string_and_coerces_typed_values():
    parser = MiMoToolParser()
    parser.adjust_request(_make_request())
    payload = (
        '<function=run_command>'
        '<parameter=command>  printf &quot;a&amp;b&quot;\nnext  </parameter>'
        '<parameter=timeout>30</parameter>'
        '<parameter=options>{"check":true}</parameter>'
        '</function>'
    )

    call = final_tool_call(parser, payload)

    assert call is not None
    assert call.function.name == 'run_command'
    assert json.loads(call.function.arguments) == {
        'command': '  printf "a&b"\nnext  ',
        'timeout': 30,
        'options': {
            'check': True
        },
    }

def test_mimo_streaming_handles_split_tags_and_multiline_value():
    parser = _make_response_parser(stream=True)
    chunks = [
        '<think>Need a tool.</think>',
        '<tool_',
        'call><function=run_',
        'command><parameter=command>line 1\n',
        '  line 2</parameter><parameter=timeout>',
        '10</parameter></function></tool_call>',
    ]
    reasoning_parts = []
    name = None
    arguments = ''

    for chunk in chunks:
        for delta, emitted in parser.stream_chunk(delta_text=chunk, delta_token_ids=[]):
            if delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
            if not emitted or not delta.tool_calls:
                continue
            for call in delta.tool_calls:
                if call.function and call.function.name:
                    name = call.function.name
                if call.function and call.function.arguments:
                    arguments += call.function.arguments

    for _ in range(3):
        delta, emitted = first_stream_delta(parser.stream_chunk(delta_text='', delta_token_ids=[]))
        if emitted and delta and delta.tool_calls:
            for call in delta.tool_calls:
                if call.function and call.function.name:
                    name = call.function.name
                if call.function and call.function.arguments:
                    arguments += call.function.arguments

    assert ''.join(reasoning_parts) == 'Need a tool.'
    assert name == 'run_command'
    assert json.loads(arguments) == {
        'command': 'line 1\n  line 2',
        'timeout': 10,
    }
    assert parser.tool_parser is not None
    assert parser.tool_parser.block_closed is True


def test_mimo_streaming_decodes_split_html_entity():
    parser = MiMoToolParser()
    parser.adjust_request(_make_request(stream=True))
    parser.begin_tool_block()
    pending = ''
    deltas = []

    for chunk in (
        '<function=run_command><parameter=command>A &quo',
        't;b&amp;c</parameter></function></tool_call>',
    ):
        text = pending + chunk
        consumed = parser.feed_tool_block(text, deltas, final=False)
        pending = text[consumed:]

    assert pending == ''
    calls = parser.build_tool_calls(deltas)
    assert json.loads(calls[0].function.arguments) == {'command': 'A "b&c'}


def test_mimo_filters_unknown_tool_names():
    parser = MiMoToolParser()
    parser.adjust_request(_make_request())

    calls = final_tool_calls(parser, '<function=unknown></function>')
    assert calls == []
