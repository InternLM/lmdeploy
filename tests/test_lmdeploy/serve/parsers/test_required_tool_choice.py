# Copyright (c) OpenMMLab. All rights reserved.
import json

import pytest

from lmdeploy._guided_decoding import compile_response_format
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParser, ToolParserManager


@pytest.fixture(scope='module')
def xgrammar_compiler():
    import xgrammar as xgr

    tokenizer_info = xgr.TokenizerInfo(
        [bytes([token_id]) for token_id in range(256)],
        vocab_type=xgr.VocabType.RAW,
        vocab_size=256,
        stop_token_ids=[0],
    )
    return xgr, xgr.GrammarCompiler(tokenizer_info)


def _tools():
    return [{
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get weather for a city.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string',
                    },
                },
                'required': ['city'],
            },
        },
    }, {
        'type': 'function',
        'function': {
            'name': 'get_time',
            'parameters': {
                'type': 'object',
                'properties': {
                    'timezone': {
                        'type': 'string',
                    },
                },
                'required': ['timezone'],
            },
        },
    }]


def _request(**kwargs):
    defaults = {
        'model': 'Qwen/Qwen3-8B',
        'messages': [{
            'role': 'user',
            'content': 'weather?',
        }],
        'tools': _tools(),
        'tool_choice': 'required',
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


@pytest.fixture()
def configured_parser():
    parser_cls = ResponseParserManager.get('default')
    old_reasoning_cls = parser_cls.reasoning_parser_cls
    old_tool_cls = parser_cls.tool_parser_cls

    def _build(*, request=None, reasoning=False, reasoning_parser='default', tool_parser='qwen3', **request_kwargs):
        parser_cls.reasoning_parser_cls = ReasoningParserManager.get(reasoning_parser) if reasoning else None
        parser_cls.tool_parser_cls = ToolParserManager.get(tool_parser)
        if reasoning and request is None:
            request_kwargs.setdefault('chat_template_kwargs', {'enable_thinking': True})
        return parser_cls(request or _request(**request_kwargs))

    try:
        yield _build
    finally:
        parser_cls.reasoning_parser_cls = old_reasoning_cls
        parser_cls.tool_parser_cls = old_tool_cls


def _walk_formats(value):
    if isinstance(value, dict):
        if 'type' in value:
            yield value
        for child in value.values():
            yield from _walk_formats(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_formats(child)


@pytest.mark.parametrize(
    'parser_name',
    [
        'qwen2d5',
        'qwen3',
        'qwen3coder',
        'kimi-k2',
        'glm47',
        'deepseek-v32',
        'deepseek-v4',
        'interns2-preview',
    ],
)
@pytest.mark.parametrize('reasoning', [False, True])
def test_builtin_required_formats_compile(parser_name, reasoning, xgrammar_compiler, configured_parser):
    request = _request()
    tools = request.tools
    parser = configured_parser(
        request=request,
        reasoning=reasoning,
        tool_parser=parser_name,
    )
    response_format = parser.request.response_format

    assert response_format['type'] == 'structural_tag'
    formats = list(_walk_formats(response_format['format']))
    required_groups = [
        item for item in formats
        if item['type'] == 'tags_with_separator' and item.get('at_least_one') is True
    ]
    assert required_groups

    serialized = str(response_format)
    for tool in tools:
        assert tool.function.name in serialized
        assert str(tool.function.parameters) in serialized

    xgr, compiler = xgrammar_compiler
    compiled = compile_response_format(compiler, response_format)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_required_rejects_tool_parser_without_response_format(monkeypatch):
    class UnsupportedToolParser(ToolParser):
        pass

    parser_cls = ResponseParserManager.get('default')
    monkeypatch.setattr(parser_cls, 'tool_parser_cls', UnsupportedToolParser)

    with pytest.raises(ValueError, match='does not support `tool_choice="required"`'):
        parser_cls(_request())


@pytest.mark.parametrize('tool_parser', ['internlm', 'intern-s1', 'llama3'])
def test_required_rejects_unsupported_builtin_tool_parser(configured_parser, tool_parser):
    with pytest.raises(ValueError, match='does not support `tool_choice="required"`'):
        configured_parser(tool_parser=tool_parser)


def test_required_overrides_response_format_without_mutating_tools(configured_parser):
    request = _request(response_format={'type': 'json_object'})
    original_tools = request.tools
    original_dump = [tool.model_dump() for tool in request.tools]

    parser = configured_parser(request=request)

    assert parser.request.response_format['type'] == 'structural_tag'
    assert request.tools is original_tools
    assert [tool.model_dump() for tool in request.tools] == original_dump


def test_required_complete_parsing_accepts_multiple_calls(configured_parser):
    parser = configured_parser()
    text = (
        '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
        '<tool_call>{"name":"get_time","arguments":{"timezone":"UTC"}}</tool_call>'
    )

    content, tool_calls, reasoning = parser.parse_complete(text)

    assert content is None
    assert reasoning is None
    assert [call.function.name for call in tool_calls] == ['get_weather', 'get_time']
    assert parser.validate_complete(text) is True


def test_required_complete_validation_preserves_reasoning_tokens(configured_parser):
    parser = configured_parser(reasoning=True)
    text = (
        'Need weather</think>'
        '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
    )
    parser.reasoning_tokens = 3

    assert parser.validate_complete(text) is True
    assert parser.reasoning_tokens == 3


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('actual_call', [False, True])
def test_required_ignores_tool_examples_in_reasoning(configured_parser, stream, actual_call):
    parser = configured_parser(reasoning=True)
    call = '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
    example = '{"name":"get_weather","arguments":{"city":"Paris"}}'
    chunks = [f'Example: {example}</think>', call if actual_call else 'No tool called.']
    if stream:
        deltas = []
        for chunk in chunks:
            deltas.extend(parser.stream_chunk(chunk, []))
        assert any(delta.tool_calls for delta, _ in deltas) is actual_call
        assert parser.validate_complete() is True
    else:
        _, calls, _ = parser.parse_complete(''.join(chunks))
        assert bool(calls) is actual_call
        assert parser.validate_complete(''.join(chunks)) is True


def test_required_streaming_preserves_reasoning_and_split_tags(configured_parser):
    parser = configured_parser(reasoning=True)
    chunks = [
        'Need to ',
        'check</th',
        'ink>\n\n<tool_',
        'call>{"name":"get_weather",',
        '"arguments":{"city":"Paris"}}</tool_',
        'call>',
    ]
    reasoning_parts = []
    tool_deltas = []
    for chunk in chunks:
        for delta, _ in parser.stream_chunk(chunk, []):
            if delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
            if delta.tool_calls:
                tool_deltas.extend(delta.tool_calls)

    assert ''.join(reasoning_parts) == 'Need to check'
    assert any(call.function and call.function.name == 'get_weather' for call in tool_deltas)
    assert parser.validate_complete() is True


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('plain assistant answer', True),
        ('<tool_call>{"name":"get_weather","arguments":{"city":</tool_call>', False),
        ('<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}', False),
    ],
)
def test_required_terminal_validation_only_checks_present_call_structure(configured_parser, text, expected):
    parser = configured_parser()

    assert parser.validate_complete(text) is expected


@pytest.mark.parametrize('arguments', [{}, {'city': 42}])
def test_required_terminal_validation_only_requires_parseable_call(configured_parser, arguments):
    parser = configured_parser()
    payload = json.dumps({'name': 'get_weather', 'arguments': arguments})
    text = f'<tool_call>{payload}</tool_call>'

    assert parser.validate_complete(text) is True


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('prefix', ['', 'assistant text'])
@pytest.mark.parametrize('stop_str', ['', '<|im_end|>', '<custom-stop>'])
def test_required_validation_preserves_text_and_stop_output(configured_parser, stream, prefix, stop_str):
    parser = configured_parser(include_stop_str_in_output=bool(stop_str), return_token_ids=True)
    text = prefix + '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
    if stream:
        deltas = parser.stream_chunk(text, [])
        deltas += parser.stream_chunk(stop_str, [0], final=True)
        content = ''.join(delta.content or '' for delta, _ in deltas)
        assert parser.validate_complete()
    else:
        content, _, _ = parser.parse_complete(text + stop_str)
        assert parser.validate_complete(text + stop_str)
    assert (content or '') == prefix + stop_str


@pytest.mark.parametrize('tool_parser', ['qwen3coder', 'glm47'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('tool_choice', ['auto', 'required'])
def test_xml_direct_types(configured_parser, tool_parser, stream, tool_choice):
    values = {'string': '42', 'integer': 42, 'number': 1.5, 'boolean': True, 'null': None,
              'array': [1, 2], 'object': {'value': 42}}
    tools = _tools()[:1]
    tools[0]['function']['parameters'] = {
        'type': 'object', 'properties': {kind: {'type': kind} for kind in values}}
    parser = configured_parser(tools=tools, tool_parser=tool_parser, tool_choice=tool_choice)
    # XML string values are raw text, not JSON string literals.
    encoded = {kind: value if isinstance(value, str) else json.dumps(value) for kind, value in values.items()}
    if tool_parser == 'qwen3coder':
        payload = '<function=get_weather>' + ''.join(
            f'<parameter={kind}>{value}</parameter>' for kind, value in encoded.items()) + '</function>'
    else:
        payload = 'get_weather' + ''.join(
            f'<arg_key>{kind}</arg_key><arg_value>{value}</arg_value>' for kind, value in encoded.items())
    if stream:
        deltas = parser.stream_chunk('<tool_call>', [])
        deltas += parser.stream_chunk(payload, [])
        deltas += parser.stream_chunk('</tool_call>', [], final=True)
        arguments = ''.join(call.function.arguments or '' for delta, _ in deltas for call in delta.tool_calls or []
                            if call.function)
    else:
        _, calls, _ = parser.parse_complete(f'<tool_call>{payload}</tool_call>')
        arguments = calls[0].function.arguments
    assert json.loads(arguments) == values


@pytest.mark.parametrize('tool_parser', ['qwen3coder', 'glm47'])
@pytest.mark.parametrize('stream', [False, True])
def test_xml_required_format_round_trip(configured_parser, xgrammar_compiler, tool_parser, stream):
    tools = _tools()[:1]
    tools[0]['function']['parameters'] = {
        'type': 'object',
        'properties': {'value': {'type': 'integer'}},
        'required': ['value'],
    }
    parser = configured_parser(tools=tools, tool_parser=tool_parser)
    if tool_parser == 'qwen3coder':
        payload = '\n<function=get_weather>\n<parameter=value>\n42\n</parameter>\n</function>\n'
    else:
        payload = 'get_weather<arg_key>value</arg_key><arg_value>42</arg_value>'
    text = f'<tool_call>{payload}</tool_call>'

    xgr, compiler = xgrammar_compiler
    matcher = xgr.GrammarMatcher(compile_response_format(compiler, parser.request.response_format))
    assert matcher.accept_string(text)
    assert matcher.accept_token(0)
    if stream:
        deltas = parser.stream_chunk('<tool_call>', [])
        deltas.extend(parser.stream_chunk(payload, []))
        deltas.extend(parser.stream_chunk('</tool_call>', [], final=True))
        arguments = ''.join(call.function.arguments or '' for delta, _ in deltas for call in delta.tool_calls or []
                            if call.function)
        assert parser.validate_complete()
    else:
        _, calls, _ = parser.parse_complete(text)
        arguments = calls[0].function.arguments
        assert parser.validate_complete(text)
    assert json.loads(arguments) == {'value': 42}


@pytest.mark.parametrize(
    ('reasoning_parser', 'tool_parser', 'model', 'block_name'),
    [
        ('deepseek-v32', 'deepseek-v32', 'deepseek-ai/DeepSeek-V3.2', 'function_calls'),
        ('deepseek-v4', 'deepseek-v4', 'deepseek-ai/DeepSeek-V4', 'tool_calls'),
    ],
)
def test_deepseek_required_default_disables_reasoning_grammar(
    configured_parser,
    reasoning_parser,
    tool_parser,
    model,
    block_name,
):
    parser = configured_parser(
        request=_request(model=model),
        reasoning=True,
        reasoning_parser=reasoning_parser,
        tool_parser=tool_parser,
    )

    assert parser.request.response_format['format']['elements'][0]['type'] == 'const_string'

    completion = (
        f'\n\n<｜DSML｜{block_name}>\n'
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        f'</｜DSML｜{block_name}>'
    )
    content, tool_calls, reasoning = parser.parse_complete(completion)

    assert content is None
    assert reasoning is None
    assert tool_calls is not None
    assert tool_calls[0].function.name == 'get_weather'
    assert parser.validate_complete(completion) is True
