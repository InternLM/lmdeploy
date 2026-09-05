# Copyright (c) OpenMMLab. All rights reserved.
import json

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
    compiled = compiler.compile_structural_tag(response_format)
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
        assert parser.validate_complete() is actual_call
    else:
        _, calls, _ = parser.parse_complete(''.join(chunks))
        assert bool(calls) is actual_call
        assert parser.validate_complete(''.join(chunks)) is actual_call


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
        '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}',
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


@pytest.mark.parametrize('ref', ['https://example.com/schema', 'http://127.0.0.1/schema', 'file:///etc/passwd',
                               'schema.json', '#/$defs/missing'])
@pytest.mark.parametrize('keyword', ['$ref', '$dynamicRef', '$recursiveRef'])
def test_required_rejects_unavailable_schema_references(configured_parser, monkeypatch, ref, keyword):
    def unexpected_fetch(*args, **kwargs):
        pytest.fail('Tool schema validation must not retrieve URLs')

    monkeypatch.setattr('urllib.request.urlopen', unexpected_fetch)
    tools = _tools()[:1]
    tools[0]['function']['parameters']['properties']['city'] = {keyword: ref}

    with pytest.raises(ValueError, match='[Rr]eference'):
        configured_parser(tools=tools)


def test_required_ignores_reference_keywords_in_schema_data(configured_parser, monkeypatch):
    def unexpected_fetch(*args, **kwargs):
        pytest.fail('Tool schema validation must not retrieve URLs')

    monkeypatch.setattr('urllib.request.urlopen', unexpected_fetch)
    tools = _tools()[:1]
    schema = tools[0]['function']['parameters']
    schema['default'] = {'$ref': 'https://example.com/not-a-schema-reference'}
    parser = configured_parser(tools=tools)

    assert parser.validate_complete('<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>')


@pytest.fixture()
def preflight_caches():
    from lmdeploy.serve.parsers.tool_parser.schema import _check_schema
    from lmdeploy.serve.parsers.tool_parser.tool_parser import _check_required_tool_grammar

    caches = (_check_schema, _check_required_tool_grammar)
    for cache in caches:
        cache.cache_clear()
    yield caches
    for cache in caches:
        cache.cache_clear()


def test_schema_cache_uses_content_without_sharing_validators(preflight_caches, monkeypatch):
    from lmdeploy.serve.parsers.tool_parser.schema import create_schema_validator

    def unexpected_fetch(*args, **kwargs):
        pytest.fail('Cached schema validation must not retrieve URLs')

    monkeypatch.setattr('urllib.request.urlopen', unexpected_fetch)
    schema_cache, _ = preflight_caches
    schema = {'$defs': {'value': {'type': 'integer', 'const': 42}}, '$ref': '#/$defs/value'}
    first = create_schema_validator(schema)
    copied = json.loads(json.dumps(schema))
    second = create_schema_validator(copied)
    assert schema_cache.cache_info().hits == 1
    assert first is not second
    assert first.schema is schema
    assert second.schema is copied
    assert first.is_valid(42) and second.is_valid(42)

    copied['$defs']['value']['const'] = 43
    changed = create_schema_validator(copied)
    assert changed.is_valid(43) and not changed.is_valid(42)
    assert first.is_valid(42) and not first.is_valid(43)
    assert schema_cache.cache_info().hits == 1

    copied['$ref'] = 'http://127.0.0.1/schema'
    for _ in range(2):
        with pytest.raises(ValueError, match='Only local'):
            create_schema_validator(copied)
    assert schema_cache.cache_info().currsize == 2


def test_required_preflight_cache_keeps_stream_state_per_request(configured_parser, preflight_caches):
    _, grammar_cache = preflight_caches
    first = configured_parser()
    first.stream_chunk('<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>', [], final=True)
    second = configured_parser()
    assert grammar_cache.cache_info().hits == 1
    assert first.tool_parser is not second.tool_parser
    assert first.validate_complete()
    assert not second.validate_complete()
    second.stream_chunk('<tool_call>{"name":"get_time","arguments":{"timezone":"UTC"}}</tool_call>', [], final=True)
    assert second.validate_complete()
    assert first.validate_complete()


@pytest.mark.parametrize('change', ['schema', 'tool_name', 'tool_parser', 'reasoning'])
def test_required_grammar_cache_keys_complete_format(configured_parser, preflight_caches, change):
    _, grammar_cache = preflight_caches
    first = configured_parser()
    tools = _tools()
    kwargs = {}
    if change == 'schema':
        tools[0]['function']['parameters']['properties']['city']['enum'] = ['Tokyo']
    elif change == 'tool_name':
        tools[0]['function']['name'] = 'other_weather'
    elif change == 'tool_parser':
        kwargs['tool_parser'] = 'qwen3coder'
    else:
        kwargs['reasoning'] = True
    changed = configured_parser(tools=tools, **kwargs)
    assert first.request.response_format != changed.request.response_format
    assert grammar_cache.cache_info().misses == 2
    assert grammar_cache.cache_info().currsize == 2
    repeated = configured_parser(tools=tools, **kwargs)
    assert repeated.request.response_format == changed.request.response_format
    assert grammar_cache.cache_info().hits == 1


def test_required_grammar_cache_does_not_admit_changed_invalid_schema(configured_parser, preflight_caches):
    _, grammar_cache = preflight_caches
    tools = _tools()[:1]
    configured_parser(tools=tools)
    tools[0]['function']['parameters']['properties']['city']['pattern'] = '(?=a)a'
    for _ in range(2):
        with pytest.raises(ValueError, match='Unsupported required-tool grammar'):
            configured_parser(tools=tools)
    assert grammar_cache.cache_info().currsize == 1
    assert grammar_cache.cache_info().hits == 0
    assert grammar_cache.cache_info().misses == 3
    tools[0]['function']['parameters']['properties']['city'].pop('pattern')
    assert configured_parser(tools=tools).validate_complete(
        '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>')
    assert grammar_cache.cache_info().hits == 1


def test_required_preflight_caches_evict_old_schemas(configured_parser, preflight_caches):
    tools = _tools()[:1]
    configured_parser(tools=tools)
    for index in range(max(cache.cache_info().maxsize for cache in preflight_caches)):
        tools[0]['function']['parameters']['properties']['city']['enum'] = [f'city-{index}']
        configured_parser(tools=tools)
    for cache in preflight_caches:
        assert cache.cache_info().currsize == cache.cache_info().maxsize
    misses = [cache.cache_info().misses for cache in preflight_caches]
    parser = configured_parser(tools=_tools()[:1])
    assert parser.validate_complete('<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>')
    assert [cache.cache_info().misses for cache in preflight_caches] == [count + 1 for count in misses]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('prefix', ['', 'assistant text'])
@pytest.mark.parametrize('stop_str', ['', '<|im_end|>', '<custom-stop>', '</tool_call>'])
def test_required_validation_preserves_text_and_stop_output(configured_parser, stream, prefix, stop_str):
    parser = configured_parser(include_stop_str_in_output=bool(stop_str), return_token_ids=True)
    text = prefix + '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
    if stop_str == '</tool_call>':
        text = text.removesuffix(stop_str)
    if stream:
        deltas = parser.stream_chunk(text, [])
        deltas += parser.stream_chunk(stop_str, [0], final=True)
        content = ''.join(delta.content or '' for delta, _ in deltas)
        assert parser.validate_complete()
    else:
        content, _, _ = parser.parse_complete(text + stop_str)
        assert parser.validate_complete(text + stop_str)
    assert (content or '') == prefix + (stop_str if stop_str != '</tool_call>' else '')


@pytest.mark.parametrize('tool_parser', ['qwen3coder', 'glm47'])
@pytest.mark.parametrize('stream', [False, True])
def test_xml_auto_direct_types_do_not_build_validators(configured_parser, monkeypatch, tool_parser, stream):
    from jsonschema import validators

    def unexpected_validator(*args, **kwargs):
        pytest.fail('Direct-type auto calls must not construct schema validators')

    monkeypatch.setattr(validators, 'validator_for', unexpected_validator)
    values = {'string': '42', 'integer': 42, 'number': 1.5, 'boolean': True, 'null': None,
              'array': [1, 2], 'object': {'value': 42}}
    tools = _tools()[:1]
    tools[0]['function']['parameters'] = {
        'type': 'object', 'properties': {kind: {'type': kind} for kind in values}}
    parser = configured_parser(tools=tools, tool_parser=tool_parser, tool_choice='auto')
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


def test_xml_auto_unused_complex_schema_does_not_build_validator(configured_parser, monkeypatch):
    from jsonschema import validators

    def unexpected_validator(*args, **kwargs):
        pytest.fail('Auto text responses must not construct argument validators')

    monkeypatch.setattr(validators, 'validator_for', unexpected_validator)
    tools = _tools()[:1]
    tools[0]['function']['parameters']['properties']['city'] = {'anyOf': [{'type': 'integer'}, {'type': 'string'}]}
    parser = configured_parser(tools=tools, tool_parser='qwen3coder', tool_choice='auto')
    assert parser.parse_complete('No tool needed.') == ('No tool needed.', None, None)


@pytest.mark.parametrize('tool_parser', ['qwen3coder', 'glm47'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('tool_choice', ['auto', 'required'])
@pytest.mark.parametrize('param_schema,raw,expected', [
    ({'type': ['integer', 'null'], 'enum': [None]}, 'null', None),
    ({'$ref': '#/$defs/value'}, '42', 42),
    ({'anyOf': [{'$ref': '#/$defs/value'}, {'type': 'null'}]}, '42', 42),
    ({'oneOf': [{'type': 'integer'}, {'type': 'null'}]}, '42', 42),
    ({'type': ['integer', 'number']}, '1.5', 1.5),
    ({'type': 'string'}, '42', '42'),
    ({'type': 'string'}, 'null', 'null'),
    ({'anyOf': [{'type': 'integer'}, {'type': 'string'}]}, '42', '42'),
])
def test_xml_schema_round_trip(configured_parser, xgrammar_compiler, tool_parser, stream, tool_choice,
                                        param_schema, raw, expected):
    tools = _tools()[:1]
    tools[0]['function']['parameters'] = {
        'type': 'object',
        '$id': 'https://example.com/tools/get_weather',
        '$defs': {'value': {'type': 'integer', 'const': 42}},
        'properties': {'value': param_schema},
        'required': ['value'],
    }
    parser = configured_parser(tools=tools, tool_parser=tool_parser, tool_choice=tool_choice)
    if tool_choice == 'required':
        prepared_validator = parser.tool_parser._function_validators['get_weather']
    if tool_parser == 'qwen3coder':
        text = f'<tool_call>\n<function=get_weather>\n<parameter=value>\n{raw}\n</parameter>\n</function>\n</tool_call>'
    else:
        text = f'<tool_call>get_weather<arg_key>value</arg_key><arg_value>{raw}</arg_value></tool_call>'

    xgr, compiler = xgrammar_compiler
    response_format = parser.request.response_format
    if tool_choice == 'auto':
        response_format = xgr.get_model_structural_tag(
            parser.tool_parser.structural_tag_model, tools, tool_choice='required', reasoning=False)
    matcher = xgr.GrammarMatcher(compiler.compile_structural_tag(response_format))
    assert matcher.accept_string(text)
    assert matcher.accept_token(0)
    if stream:
        deltas = []
        # Protocol tokens are emitted atomically; only fragment the payload.
        deltas.extend(parser.stream_chunk('<tool_call>', []))
        for char in text.removeprefix('<tool_call>').removesuffix('</tool_call>'):
            deltas.extend(parser.stream_chunk(char, []))
        deltas.extend(parser.stream_chunk('</tool_call>', [], final=True))
        arguments = ''.join(call.function.arguments or '' for delta, _ in deltas for call in delta.tool_calls or []
                            if call.function)
        assert parser.validate_complete()
    else:
        _, calls, _ = parser.parse_complete(text)
        arguments = calls[0].function.arguments
        assert parser.validate_complete(text)
    assert json.loads(arguments) == {'value': expected}
    if tool_choice == 'required':
        # XML coercion and complete-call validation reuse request preflight's validator.
        assert parser.tool_parser._function_validators['get_weather'] is prepared_validator


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
