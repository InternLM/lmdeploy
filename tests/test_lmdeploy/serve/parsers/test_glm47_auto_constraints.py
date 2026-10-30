# Copyright (c) OpenMMLab. All rights reserved.
"""CPU-only regressions for optional GLM tool-call constraints."""
import copy
import json
from types import SimpleNamespace

import pytest
import xgrammar as xgr
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from lmdeploy._guided_decoding import compile_response_format
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.response_parser import BaseResponseParser
from lmdeploy.serve.parsers.tool_parser.glm47_tool_constraints import build_auto_tool_format
from lmdeploy.serve.parsers.tool_parser.glm47_tool_parser import Glm47ToolParser


def _tools():
    return [{
        'type': 'function',
        'function': {
            'name': 'terminal',
            'description': 'Execute a shell command.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {'type': 'string'},
                    'summary': {'type': 'string'},
                    'timeout': {'type': 'number'},
                },
                'required': ['command'],
            },
        },
    }, {
        'type': 'function',
        'function': {
            'name': 'save_note',
            'parameters': {
                'type': 'object',
                'properties': {'text': {'type': 'string'}},
                'additionalProperties': False,
            },
        },
    }]


def _request(**kwargs):
    return ChatCompletionRequest(**{
        'model': 'test', 'messages': [], 'tools': _tools(), 'tool_choice': 'auto', **kwargs,
    })


@pytest.fixture(scope='module')
def context():
    """Use distinct control IDs and ordinary bytes, without downloading a
    model."""
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    ordinary = Tokenizer.from_str(backend.to_str())
    backend.add_tokens([
        AddedToken(tag, normalized=False)
        for tag in (*Glm47ToolParser.token_id_terminals, '<think>', '</think>')
    ])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token='<eos>', clean_up_tokenization_spaces=False)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer))

    def compile_tools(tools):
        return compile_response_format(compiler, build_auto_tool_format(tools, tokenizer))

    return SimpleNamespace(
        tokenizer=tokenizer, ordinary=ordinary, compile_tools=compile_tools,
        grammar=compile_tools(_request().tools),
    )


def _accepts(context, *, text=None, ids=None, grammar=None):
    if ids is None:
        ids = context.tokenizer.encode(text, add_special_tokens=False)
    matcher = xgr.GrammarMatcher(context.grammar if grammar is None else grammar)
    return all(matcher.accept_token(token_id) for token_id in [*ids, context.tokenizer.eos_token_id])


def _call(name='terminal', arguments=(('command', 'pwd'), ('summary', 'Check working directory'))):
    pairs = ''.join(f'<arg_key>{key}</arg_key><arg_value>{value}</arg_value>' for key, value in arguments)
    return f'<tool_call>{name}{pairs}</tool_call>'


def _parser(context, *, tool_parser='glm47', tokenizer=True, **kwargs):
    class Parser(BaseResponseParser):
        pass

    Parser.set_parsers('default', tool_parser, context.tokenizer if tokenizer else None)
    return Parser(_request(**kwargs))


@pytest.mark.parametrize('text', [
    '', 'Ordinary answer.', 'Thought.</think>Final answer.',
    _call(), _call() + _call('save_note', [('text', 'Done')]),
    'Thought.</think>' + _call() + 'Done.',
])
def test_auto_allows_text_reasoning_and_optional_calls(context, text):
    assert _accepts(context, text=text)


@pytest.mark.parametrize('key', ['description', 'cwd', 'invented_parameter', 'text'])
@pytest.mark.parametrize('additional_properties', [None, False])
def test_rejects_undeclared_argument_names(context, key, additional_properties):
    tools = _request().tools
    if additional_properties is not None:
        tools[0].function.parameters['additionalProperties'] = additional_properties
    grammar = context.compile_tools(tools)
    assert not _accepts(context, grammar=grammar, text=_call(arguments=[('command', 'pwd'), (key, 'value')]))


@pytest.mark.parametrize('key', ['description', 'path.to', '状态'])
def test_argument_names_come_from_the_request(context, key):
    tools = _request().tools
    tools[0].function.parameters['properties'][key] = {'type': 'string'}
    grammar = context.compile_tools(tools)
    assert _accepts(context, grammar=grammar, text=_call(arguments=[(key, 'value')]))


def test_argument_names_are_scoped_to_each_tool(context):
    assert not _accepts(context, text=_call('save_note', [('command', 'pwd')]))
    assert not _accepts(context, text=_call('unknown_tool', [('command', 'pwd')]))


def test_argument_order_whitespace_and_unicode(context):
    assert _accepts(context, text=(
        '<tool_call>\nterminal\n<arg_key> timeout </arg_key>\n<arg_value>90</arg_value>\n'
        '<arg_key>summary</arg_key><arg_value>读取文件 🌍</arg_value>'
        '<arg_key>command</arg_key><arg_value>pwd\nls</arg_value></tool_call>'
    ))


@pytest.mark.parametrize('literal', [
    '</arg_value>', '<tool_call>', '</think>',
    '<tool_call>terminal<arg_key>description</arg_key>',
])
def test_ordinary_token_literals_remain_argument_text(context, literal):
    tokenizer = context.tokenizer
    ordinary_ids = context.ordinary.encode(literal).ids
    assert tokenizer.decode(ordinary_ids) == literal
    assert not set(ordinary_ids).intersection(tokenizer.get_added_vocab().values())
    assert _accepts(context, ids=ordinary_ids)
    prefix = '<tool_call>terminal<arg_key>command</arg_key><arg_value>echo '
    suffix = '</arg_value></tool_call>'
    ids = (tokenizer.encode(prefix, add_special_tokens=False) + ordinary_ids
           + tokenizer.encode(suffix, add_special_tokens=False))
    assert _accepts(context, ids=ids)
    _, calls, _ = _parser(context, chat_template_kwargs={'enable_thinking': False}).parse_complete(
        prefix + literal + suffix, ids)
    assert json.loads(calls[0].function.arguments) == {'command': 'echo ' + literal}


@pytest.mark.parametrize('text', [
    '<tool_call>terminal',
    '<tool_call>terminal<arg_key>command',
    '<tool_call>terminal<arg_key>command</arg_key><arg_value>pwd',
])
def test_cannot_stop_inside_tool_call(context, text):
    matcher = xgr.GrammarMatcher(context.grammar)
    assert all(matcher.accept_token(i) for i in context.tokenizer.encode(text, add_special_tokens=False))
    assert not matcher.accept_token(context.tokenizer.eos_token_id)


@pytest.mark.parametrize('schema_change', [
    {'additionalProperties': True},
    {'additionalProperties': {'type': 'string'}},
    {'$ref': '#/$defs/args', '$defs': {'args': {'type': 'object'}}},
    {'allOf': [{'type': 'object'}]},
    {'anyOf': [{'type': 'object'}]},
    {'oneOf': [{'type': 'object'}]},
    {'patternProperties': {'^custom_': {'type': 'string'}}},
    {'properties': {}},
])
def test_open_or_composed_schemas_keep_free_argument_names(context, schema_change):
    tools = _request().tools
    tools[0].function.parameters.update(schema_change)
    assert build_auto_tool_format(tools[:1], context.tokenizer) is None
    # An open tool stays open even alongside another tool with constrained names.
    grammar = context.compile_tools(tools)
    assert _accepts(context, grammar=grammar, text=_call(arguments=[('description', 'value')]))
    assert not _accepts(context, grammar=grammar, text=_call('save_note', [('description', 'value')]))


def test_does_not_add_full_schema_validation(context):
    assert _accepts(context, text=_call(arguments=[]))  # Required fields are not enforced.
    assert _accepts(context, text=_call(arguments=[('timeout', 'not a number')]))
    assert _accepts(context, text=_call(arguments=[('command', 'pwd'), ('command', 'ls')]))


def test_missing_tokenizer_or_tools_does_not_add_constraints(context):
    assert build_auto_tool_format(_request().tools, None) is None
    assert build_auto_tool_format([], context.tokenizer) is None


def test_missing_control_ids_cannot_fall_back_to_string_matching(context, monkeypatch):
    monkeypatch.setattr(context.tokenizer, 'get_added_vocab', lambda: {})
    with pytest.raises(ValueError, match='dedicated control token IDs'):
        build_auto_tool_format(_request().tools, context.tokenizer)


def test_auto_routing_preserves_request_and_history(context):
    request = _request(messages=[
        {'role': 'user', 'content': 'Where am I?'},
        {'role': 'assistant', 'content': 'Checking.', 'reasoning_content': 'Need the working directory.'},
        {'role': 'user', 'content': 'Continue.'},
    ])
    original = copy.deepcopy(request.model_dump())

    class Parser(BaseResponseParser):
        pass

    Parser.set_parsers('default', 'glm47', context.tokenizer)
    parser = Parser(request)
    assert parser.request.response_format['type'] == 'structural_tag'
    assert parser.request.tool_choice == 'auto'
    assert parser.request.messages == original['messages']
    assert request.model_dump() == original


@pytest.mark.parametrize('kwargs', [
    {'tool_choice': 'none'},
    {'tool_choice': {'type': 'function', 'function': {'name': 'terminal'}}},
    {'tools': None},
    {'tool_parser': None},
    {'tool_parser': 'qwen3'},
    {'tokenizer': False},
])
def test_other_request_paths_are_unchanged(context, kwargs):
    assert _parser(context, **kwargs).request.response_format is None


@pytest.mark.parametrize('format_type', ['text', 'json_object'])
def test_explicit_format_routing(context, format_type):
    response_format = _parser(context, response_format={'type': format_type}).request.response_format
    # The existing normalizer treats an explicit text format as unconstrained.
    assert response_format['type'] == ('structural_tag' if format_type == 'text' else 'json_object')


def test_required_still_uses_existing_grammar(context):
    parser = _parser(context, tool_choice='required', chat_template_kwargs={'enable_thinking': False})
    assert parser.request.response_format == Glm47ToolParser.build_required_response_format(
        _request().tools, reasoning=False)
