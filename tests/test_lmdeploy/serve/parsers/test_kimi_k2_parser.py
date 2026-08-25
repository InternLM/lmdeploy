import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import KimiK2ReasoningParser, ReasoningParserManager
from lmdeploy.serve.parsers.response_parser import validate_parser_names
from lmdeploy.serve.parsers.tool_parser import KimiK2ToolParser, ToolParserManager

MODEL_ID = 'moonshotai/Kimi-K2.6'
SECTION_BEGIN = '<|tool_calls_section_begin|>'
SECTION_END = '<|tool_calls_section_end|>'
CALL_BEGIN = '<|tool_call_begin|>'
ARGUMENT_BEGIN = '<|tool_call_argument_begin|>'
CALL_END = '<|tool_call_end|>'

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'weather.lookup',
            'description': 'Look up the weather.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string'
                    }
                },
                'required': ['city'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'clock-now',
            'description': 'Look up the current time.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'zone': {
                        'type': 'string'
                    }
                },
                'required': ['zone'],
            },
        },
    },
]


def _tool_call(call_id: str, arguments: str) -> str:
    return f'{CALL_BEGIN}{call_id}{ARGUMENT_BEGIN}{arguments}{CALL_END}'


@pytest.fixture()
def make_parser():
    cls = ResponseParserManager.get('default')
    old_reasoning_cls = cls.reasoning_parser_cls
    old_tool_cls = cls.tool_parser_cls
    old_tokenizer = cls.tokenizer

    def _make(*,
              chat_template_kwargs=None,
              enable_thinking=None,
              stream=False,
              with_reasoning=True,
              with_tools=False):
        cls.reasoning_parser_cls = ReasoningParserManager.get('kimi_k2') if with_reasoning else None
        cls.tool_parser_cls = ToolParserManager.get('kimi_k2') if with_tools else None
        request = ChatCompletionRequest(
            model=MODEL_ID,
            messages=[],
            stream=stream,
            tools=TOOLS if with_tools else None,
            tool_choice='auto' if with_tools else 'none',
            chat_template_kwargs=chat_template_kwargs,
            enable_thinking=enable_thinking,
        )
        return cls(request=request)

    yield _make

    cls.reasoning_parser_cls = old_reasoning_cls
    cls.tool_parser_cls = old_tool_cls
    cls.tokenizer = old_tokenizer


@pytest.mark.parametrize('name', ['kimi_k2', 'kimi-k2'])
def test_kimi_parser_aliases_are_registered(name):
    assert ReasoningParserManager.get(name) is KimiK2ReasoningParser
    assert ToolParserManager.get(name) is KimiK2ToolParser
    assert validate_parser_names(reasoning_parser_name=name, tool_parser_name=name) == (name, name)


@pytest.mark.parametrize(
    ('kwargs', 'starts_in_reasoning'),
    [
        ({}, True),
        ({'thinking': True}, True),
        ({'thinking': False}, False),
    ],
)
def test_kimi_reasoning_mode_switches(kwargs, starts_in_reasoning):
    parser = KimiK2ReasoningParser(**kwargs)
    assert parser.starts_in_reasoning_mode() is starts_in_reasoning


@pytest.mark.parametrize(
    ('chat_template_kwargs', 'expected'),
    [
        (None, True),
        ({'thinking': True}, True),
        ({'thinking': False}, False),
    ],
)
def test_request_template_mode_matches_parser(make_parser, chat_template_kwargs,
                                              expected):
    parser = make_parser(chat_template_kwargs=chat_template_kwargs, with_tools=True)

    assert parser.profile.starts_in_reasoning_mode is expected
    assert parser.request.skip_special_tokens is False
    assert parser.request.spaces_between_special_tokens is False


def test_default_thinking_non_streaming(make_parser):
    parser = make_parser()
    content, tool_calls, reasoning = parser.parse_complete('reasoning steps</think>final answer')

    assert reasoning == 'reasoning steps'
    assert content == 'final answer'
    assert tool_calls is None


def test_explicit_think_tag_and_unclosed_reasoning(make_parser):
    parser = make_parser()
    content, tool_calls, reasoning = parser.parse_complete(
        '<think>reasoning steps</think>final answer')

    assert reasoning == 'reasoning steps'
    assert content == 'final answer'
    assert tool_calls is None

    parser = make_parser()
    content, tool_calls, reasoning = parser.parse_complete('reasoning only')

    assert reasoning == 'reasoning only'
    assert content is None
    assert tool_calls is None


def test_thinking_false_streaming_and_non_streaming(make_parser):
    kwargs = {'thinking': False}
    parser = make_parser(chat_template_kwargs=kwargs)
    content, tool_calls, reasoning = parser.parse_complete('instant answer')

    assert content == 'instant answer'
    assert tool_calls is None
    assert reasoning is None

    parser = make_parser(chat_template_kwargs=kwargs, stream=True)
    deltas = []
    for chunk in ['instant ', 'answer']:
        deltas.extend(parser.stream_chunk(chunk, []))

    assert ''.join(delta.content or '' for delta, _ in deltas) == 'instant answer'
    assert all(delta.reasoning_content is None for delta, _ in deltas)


def test_tool_section_implicitly_ends_reasoning(make_parser):
    tool_section = (
        SECTION_BEGIN + _tool_call('functions.weather.lookup:3', '{"city":"Paris"}') +
        SECTION_END)
    parser = make_parser(with_tools=True)

    content, tool_calls, reasoning = parser.parse_complete(f'need weather{tool_section}')

    assert reasoning == 'need weather'
    assert content is None
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].function.name == 'weather.lookup'


def test_complete_response_with_multiple_tool_calls(make_parser):
    arguments_0 = '{"city": "北京", "units": ["c", "f"]}'
    arguments_1 = '{"zone":"UTC"}'
    tool_section = (
        SECTION_BEGIN + _tool_call('functions.weather.lookup:41', arguments_0) +
        _tool_call('clock-now:99', arguments_1) + SECTION_END)
    parser = make_parser(with_tools=True)

    content, tool_calls, reasoning = parser.parse_complete(
        f'check both tools</think>I will check. {tool_section}')

    assert reasoning == 'check both tools'
    assert content == 'I will check. '
    assert tool_calls is not None
    assert [call.id for call in tool_calls] == ['functions.weather.lookup:41', 'clock-now:99']
    assert [call.function.name for call in tool_calls] == ['weather.lookup', 'clock-now']
    assert [call.function.arguments for call in tool_calls] == [arguments_0, arguments_1]
    assert parser.request.skip_special_tokens is False
    assert parser.request.spaces_between_special_tokens is False
    assert parser.validate_complete(
        f'check both tools</think>I will check. {tool_section}') is True


def test_streaming_multiple_calls_with_split_delimiters(make_parser):
    parser = make_parser(stream=True, with_tools=True)
    chunks = [
        'reasoning</thi',
        'nk>Answer <|tool_calls_sec',
        'tion_begin|><|tool_call_be',
        'gin|>functions.weather.lookup:41<|tool_call_argument_be',
        'gin|>{"city": "北京"}<|tool_call_',
        'end|><|tool_call_begin|>clock-now:99<|tool_call_argument_begin|>',
        '{"zone":"UTC"}<|tool_call_end|><|tool_calls_section_',
        'end|>',
    ]

    deltas = []
    for chunk in chunks:
        deltas.extend(parser.stream_chunk(chunk, []))

    assert ''.join(delta.reasoning_content or '' for delta, _ in deltas) == 'reasoning'
    assert ''.join(delta.content or '' for delta, _ in deltas) == 'Answer '

    tool_deltas = [call for delta, emitted in deltas if emitted for call in delta.tool_calls or []]
    name_deltas = [call for call in tool_deltas if call.function and call.function.name]
    argument_deltas = [call for call in tool_deltas if call.function and call.function.arguments]
    assert [call.id for call in name_deltas] == ['functions.weather.lookup:41', 'clock-now:99']
    assert [call.index for call in name_deltas] == [0, 1]
    assert [call.function.name for call in name_deltas] == ['weather.lookup', 'clock-now']
    assert [call.index for call in argument_deltas] == [0, 1]
    assert [call.function.arguments for call in argument_deltas] == [
        '{"city": "北京"}',
        '{"zone":"UTC"}',
    ]
    assert parser.validate_complete() is True


@pytest.mark.parametrize('call_id', ['0', 'functions.weather.lookup', 'weather.lookup:x'])
def test_tool_parser_rejects_non_native_call_ids(call_id):
    parser = KimiK2ToolParser()
    payload = _tool_call(call_id, '{"city":"Paris"}')
    assert parser.parse_tool_call_complete(payload) is None


@pytest.mark.parametrize(
    ('raw_arguments', 'expected_arguments'),
    [
        ('', '{}'),
        ('  ', '{}'),
        ('{"city":', '{"city":'),
        ('["not", "an", "object"]', '["not", "an", "object"]'),
    ],
)
def test_tool_parser_preserves_degraded_arguments(raw_arguments, expected_arguments):
    parser = KimiK2ToolParser()
    payload = _tool_call('functions.weather.lookup:7', raw_arguments)

    calls = parser.parse_tool_call_complete(payload)

    assert calls is not None and len(calls) == 1
    assert calls[0].function.arguments == expected_arguments


def test_tool_parser_accepts_missing_inner_end_at_section_end(make_parser):
    arguments = '{"city":"Paris"}'
    truncated_call = f'{CALL_BEGIN}functions.weather.lookup:3{ARGUMENT_BEGIN}{arguments}'
    parser = make_parser(chat_template_kwargs={'thinking': False}, with_tools=True)

    content, tool_calls, reasoning = parser.parse_complete(
        f'before{SECTION_BEGIN}{truncated_call}{SECTION_END}')

    assert content == 'before'
    assert reasoning is None
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].id == 'functions.weather.lookup:3'
    assert tool_calls[0].function.arguments == arguments


def test_malformed_id_is_skipped_without_hiding_later_calls():
    parser = KimiK2ToolParser()
    payload = (
        _tool_call('invalid-id', '{"ignored":true}') +
        _tool_call('functions.weather.lookup:8', '{"city":"Paris"}'))

    calls = parser.parse_tool_call_complete(payload)

    assert calls is not None and len(calls) == 1
    assert calls[0].id == 'functions.weather.lookup:8'


def test_streaming_malformed_call_does_not_hide_later_call(make_parser):
    parser = make_parser(
        chat_template_kwargs={'thinking': False}, stream=True, with_tools=True)
    payload = (
        f'{SECTION_BEGIN}{CALL_BEGIN}broken' +
        _tool_call('functions.weather.lookup:8', '{"city":"Paris"}') + SECTION_END)

    deltas = parser.stream_chunk(payload, [], final=True)

    tool_deltas = [call for delta, emitted in deltas if emitted for call in delta.tool_calls or []]
    assert [call.id for call in tool_deltas if call.id] == ['functions.weather.lookup:8']
    assert ''.join(call.function.arguments or '' for call in tool_deltas) == '{"city":"Paris"}'
