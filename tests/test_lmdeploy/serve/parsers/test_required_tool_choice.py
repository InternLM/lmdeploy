# Copyright (c) OpenMMLab. All rights reserved.
import json
from copy import deepcopy

import pytest

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
    'model_format',
    [
        'qwen_3',
        'qwen_3_5',
        'qwen_3_coder',
        'llama',
        'kimi',
        'glm_4_7',
        'deepseek_v3_2',
        'deepseek_v4',
    ],
)
@pytest.mark.parametrize('reasoning', [False, True])
def test_xgrammar_structural_formats_require_schema_constrained_calls(model_format, reasoning, xgrammar_compiler):
    tools = _tools()
    original = deepcopy(tools)

    if model_format == 'kimi':
        parser_cls = ToolParserManager.get('kimi-k2')
    else:
        class XGrammarToolParser(ToolParser):
            structural_tag_model = model_format

        parser_cls = XGrammarToolParser

    response_format = parser_cls.build_required_tool_response_format(
        _request(),
        tools,
        reasoning=reasoning,
    )

    assert tools == original
    assert response_format['type'] == 'structural_tag'
    formats = list(_walk_formats(response_format['format']))
    required_groups = [
        item for item in formats
        if item['type'] == 'tags_with_separator' and item.get('at_least_one') is True
    ]
    assert required_groups

    serialized = str(response_format)
    for tool in tools:
        assert tool['function']['name'] in serialized
        assert str(tool['function']['parameters']) in serialized

    xgr, compiler = xgrammar_compiler
    compiled = compiler.compile_structural_tag(response_format)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_structural_format_rejects_empty_tools():
    class XGrammarToolParser(ToolParser):
        structural_tag_model = 'qwen_3'

    with pytest.raises(ValueError, match='requires at least one tool'):
        XGrammarToolParser.build_required_tool_response_format(_request(), [], reasoning=False)


def test_internlm_required_response_format_is_parser_specific():
    parser_cls = ToolParserManager.get('internlm')

    response_format = parser_cls.build_required_tool_response_format(
        _request(),
        _tools(),
        reasoning=True,
    )

    assert response_format['type'] == 'structural_tag'
    assert response_format['format']['type'] == 'sequence'
    serialized = str(response_format)
    assert '<|action_start|><|plugin|>' in serialized
    assert '<|action_end|>' in serialized


def test_llama_required_response_format_is_parser_specific():
    parser_cls = ToolParserManager.get('llama3')

    response_format = parser_cls.build_required_tool_response_format(
        _request(model='meta-llama/Llama-3.1-8B'),
        _tools(),
        reasoning=False,
    )

    tags = [item for item in _walk_formats(response_format) if item['type'] == 'tag']
    assert tags
    assert all(tag['begin'].startswith('<|python_tag|>') for tag in tags)


@pytest.mark.parametrize(
    ('parser_name', 'model', 'reasoning', 'expected'),
    [
        ('qwen2d5', 'Qwen/Qwen2.5-7B', False, 'qwen_3'),
        ('qwen3', 'Qwen/Qwen3-8B', True, 'qwen_3'),
        ('qwen3coder', 'Qwen/Qwen3-Coder', False, 'qwen_3_coder'),
        ('qwen3coder', 'Qwen/Qwen3.5-35B', False, 'qwen_3_coder'),
        ('qwen3coder', 'Qwen/Qwen3-Coder', True, 'qwen_3_5'),
        ('llama3', 'meta-llama/Llama-3.1-8B', False, 'llama'),
        ('glm47', 'zai-org/GLM-4.7', True, 'glm_4_7'),
        ('deepseek-v32', 'deepseek-ai/DeepSeek-V3.2', True, 'deepseek_v3_2'),
        ('deepseek-v4', 'deepseek-ai/DeepSeek-V4', True, 'deepseek_v4'),
        ('internlm', 'internlm/internlm2-chat-7b', False, 'internlm'),
    ],
)
def test_builtin_parser_structural_format_mapping(parser_name, model, reasoning, expected):
    parser_cls = ToolParserManager.get(parser_name)

    assert parser_cls.get_structural_tag_model(_request(model=model), reasoning=reasoning) == expected
    assert parser_cls.supports_required_tool_choice() is True


def test_custom_parser_is_unsupported_without_structural_format():

    class CustomToolParser(ToolParser):
        pass

    assert CustomToolParser.supports_required_tool_choice() is False


def test_required_uses_tool_parser_response_format_builder():
    response_format = {
        'type': 'structural_tag',
        'format': {
            'type': 'const_string',
            'value': '<custom_required_tool_call>',
        },
    }

    class CustomRequiredToolParser(ToolParser):

        @classmethod
        def build_required_tool_response_format(cls, request, tools, *, reasoning: bool):
            assert request.tool_choice == 'required'
            assert tools
            return response_format

        @classmethod
        def get_tool_open_tag(cls):
            return '<tool_call>'

        @classmethod
        def get_tool_close_tag(cls):
            return '</tool_call>'

        @classmethod
        def get_tool_payload_format(cls):
            return 'json'

        def decode_tool_incremental(self, added_text: str, *, final: bool):
            return []

        def parse_tool_call_complete(self, payload: str):
            return None

    parser_cls = ResponseParserManager.get('default')
    old_reasoning_cls = parser_cls.reasoning_parser_cls
    old_tool_cls = parser_cls.tool_parser_cls
    try:
        parser_cls.reasoning_parser_cls = None
        parser_cls.tool_parser_cls = CustomRequiredToolParser

        assert CustomRequiredToolParser.supports_required_tool_choice() is True
        parser = parser_cls(_request())

        assert parser.request.response_format == response_format
    finally:
        parser_cls.reasoning_parser_cls = old_reasoning_cls
        parser_cls.tool_parser_cls = old_tool_cls


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
    'text',
    [
        'plain assistant answer',
        '<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
        '<tool_call>{"name":"get_weather","arguments":{"city":</tool_call>',
        'answer<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>',
    ],
)
def test_required_terminal_validation_rejects_missing_or_malformed_calls(configured_parser, text):
    parser = configured_parser()

    assert parser.validate_complete(text) is False


@pytest.mark.parametrize(
    'arguments',
    [
        {},
        {'city': 42},
    ],
)
def test_required_terminal_validation_enforces_function_schema(configured_parser, arguments):
    parser = configured_parser()
    payload = json.dumps({'name': 'get_weather', 'arguments': arguments})
    text = f'<tool_call>{payload}</tool_call>'

    assert parser.validate_complete(text) is False


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

    assert parser.reasoning_enabled is False
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
