# Copyright (c) OpenMMLab. All rights reserved.
"""Branch constraints must survive coercion through the response parser."""
import json

import pytest
from jsonschema import Draft202012Validator

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.response_parser import BaseResponseParser
from lmdeploy.serve.parsers.tool_parser import Glm47ToolParser, Qwen3CoderToolParser

PARSERS = [pytest.param(Glm47ToolParser, id='glm47'), pytest.param(Qwen3CoderToolParser, id='qwen3coder')]
BRANCH_CONST = {'anyOf': [{'type': 'string'}, {'const': 1}]}
BRANCH_ENUM = {'anyOf': [{'type': 'string'}, {'enum': [1, 3]}]}
OVERLAPPING_TYPES = [{'type': 'string'}, {'type': 'integer'}, {'type': 'number'}]
SIBLING_UNIONS = {'anyOf': [{'type': 'string'}, {'type': 'integer'}],
                 'oneOf': [{'const': '42'}, {'const': '43'}]}


def _parse(parser_cls, schema, raw_value, stream, tool_choice, defs=None):
    class Parser(BaseResponseParser):
        tool_parser_cls = parser_cls
        reasoning_parser_cls = None

    parameters = {'type': 'object', 'properties': {'value': schema}, 'required': ['value'],
                  '$defs': defs or {'One': {'const': 1}, 'Int': {'type': 'integer'}}}
    request = ChatCompletionRequest(
        model='schema-test', messages=[{'role': 'user', 'content': 'submit'}], stream=stream,
        tools=[{'type': 'function', 'function': {'name': 'submit', 'parameters': parameters}}],
        tool_choice=tool_choice)
    parser = Parser(request)
    if parser_cls is Glm47ToolParser:
        chunks = ['<tool_call>submit<arg_key>value</arg_key><arg_value>', *raw_value,
                  '</arg_value></tool_call>']
    else:
        chunks = ['<tool_call><function=submit><parameter=value>', *raw_value,
                  '</parameter></function></tool_call>']
    if stream:
        deltas = []
        for index, chunk in enumerate(chunks):
            for delta, _ in parser.stream_chunk(chunk, [], final=index == len(chunks) - 1):
                deltas.extend(delta.tool_calls or [])
        calls = parser.tool_parser.build_tool_calls(deltas)
    else:
        _, calls, _ = parser.parse_complete(''.join(chunks))
    assert len(calls) == 1
    assert calls[0].function.name == 'submit'
    return parameters, json.loads(calls[0].function.arguments)


CASES = [
    pytest.param(BRANCH_CONST, '2', '2', id='branch-const-rejects-number'),
    pytest.param(BRANCH_CONST, '1', 1, id='branch-const-accepts-number'),
    pytest.param(BRANCH_ENUM, '2', '2', id='branch-enum-rejects-number'),
    pytest.param(BRANCH_ENUM, '3', 3, id='branch-enum-accepts-number'),
    pytest.param({'anyOf': [{'type': 'string'}, {'$ref': '#/$defs/One'}]},
                 '2', '2', id='referenced-const'),
    pytest.param({'anyOf': [{'type': 'string'}, {'anyOf': [{'const': 1}, {'const': 3}]}]},
                 '2', '2', id='nested-union'),
    pytest.param({'allOf': [BRANCH_CONST, {}]}, '2', '2', id='allof-union'),
    pytest.param({'oneOf': OVERLAPPING_TYPES}, '1', '1', id='oneof-overlapping-integer'),
    pytest.param({'oneOf': OVERLAPPING_TYPES}, '1.0', '1.0', id='oneof-overlapping-integral-float'),
    pytest.param({'oneOf': OVERLAPPING_TYPES}, '1.5', 1.5, id='oneof-unambiguous-fraction'),
    pytest.param({'anyOf': OVERLAPPING_TYPES}, '1', 1, id='anyof-allows-overlap'),
    pytest.param({'oneOf': [{'type': 'string'}, {'const': 1}, {'const': 1.0}]},
                 '1', '1', id='oneof-numeric-literal-overlap'),
    pytest.param({'oneOf': [{'type': 'string'}, {'type': 'null'}, {'const': None}]},
                 'null', 'null', id='oneof-null-overlap'),
    pytest.param({'oneOf': [{'type': 'string'}, {'const': None}]},
                 'null', None, id='oneof-unambiguous-null'),
    pytest.param({'anyOf': [{'type': 'string'}, {'type': 'null'}, {'const': None}]},
                 'null', None, id='anyof-allows-null-overlap'),
    pytest.param({'oneOf': [{'type': 'string'}, {'type': 'boolean'}, {'type': 'number'}]},
                 'true', True, id='boolean-is-not-number'),
    pytest.param({'oneOf': [{'type': 'string'}, {'const': [True]}, {'const': [1]}]},
                 '[true]', [True], id='nested-boolean-is-not-number'),
    pytest.param(SIBLING_UNIONS, '42', '42', id='sibling-anyof-oneof'),
    pytest.param({'allOf': [{key: value} for key, value in SIBLING_UNIONS.items()]},
                 '42', '42', id='equivalent-allof'),
    pytest.param({'anyOf': [{'type': 'string'}, {'type': 'integer'}],
                  'oneOf': [{'const': 42}, {'const': 43}]}, '42', 42, id='sibling-unions-valid-number'),
    pytest.param({'oneOf': [{'type': 'string'}, {'$ref': '#/$defs/Int'}, {'$ref': '#/$defs/Int'}]},
                 '1', '1', id='duplicate-referenced-branches-still-count-twice'),
    pytest.param({'oneOf': [{}, {'type': 'integer'}]}, '1', '1', id='unconstrained-branch-overlap'),
    pytest.param({'oneOf': [True, {'type': 'integer'}]}, '1', '1', id='boolean-schema-overlap'),
    pytest.param({'anyOf': [False, {'type': 'number'}]}, '1', 1, id='false-schema-branch'),
    pytest.param({'anyOf': [{'type': 'string'}, True]}, 'true', True, id='true-schema-branch'),
    pytest.param({'oneOf': [True, {'type': 'string'}]}, '1', 1, id='true-schema-exclusivity'),
    pytest.param({'anyOf': [{'oneOf': OVERLAPPING_TYPES}, {'const': 2}]},
                 '1', '1', id='nested-oneof-overlap'),
    pytest.param({'oneOf': [{'const': '42'}, {'enum': ['42', '"42"']}]},
                 '"42"', '"42"', id='quoted-string-overlap'),
    pytest.param({'anyOf': [{'type': 'string'}, {'const': 1, 'description': 'one'}]},
                 'false', 'false', id='annotations-do-not-relax-constraints'),
    # Non-coercion constraints can distinguish same-type branches. Do not
    # incorrectly reject these native values based on their types alone.
    pytest.param({'oneOf': [{'type': 'object', 'required': ['a']},
                            {'type': 'object', 'required': ['b']}]},
                 '{"a":1}', {'a': 1}, id='object-constraints-disambiguate-branches'),
    pytest.param({'oneOf': [{'type': 'number', 'minimum': 0},
                            {'type': 'number', 'maximum': -1}]},
                 '1.5', 1.5, id='numeric-constraints-disambiguate-branches'),
]


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize('stream', [False, True], ids=['complete', 'stream'])
@pytest.mark.parametrize('tool_choice', ['auto', 'required'])
@pytest.mark.parametrize(('schema', 'raw_value', 'expected'), CASES)
def test_combinator_coercion_through_response_parser(parser_cls, stream, tool_choice, schema, raw_value, expected):
    parameters, actual = _parse(parser_cls, schema, raw_value, stream, tool_choice)
    Draft202012Validator.check_schema(parameters)
    validator = Draft202012Validator(parameters)
    validator.validate({'value': expected})
    validator.validate(actual)
    assert actual == {'value': expected}


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize('stream', [False, True], ids=['complete', 'stream'])
@pytest.mark.parametrize('budget_kind', ['visits', 'literals'])
def test_branch_matching_shares_resolution_budget(parser_cls, stream, budget_kind, monkeypatch):
    from lmdeploy.serve.parsers.tool_parser import xml_tool_parser

    if budget_kind == 'visits':
        monkeypatch.setattr(xml_tool_parser, '_MAX_SCHEMA_RESOLVE_VISITS', 4)
        schema, raw = BRANCH_CONST, '1'
    else:
        monkeypatch.setattr(xml_tool_parser, '_MAX_SCHEMA_LITERAL_WORK', 5)
        schema, raw = {'anyOf': [{'type': 'string'}, {'const': [1, 2]}]}, '[1,2]'
    parameters, actual = _parse(parser_cls, schema, raw, stream, 'auto')
    assert actual == {'value': raw}
    Draft202012Validator(parameters).validate(actual)


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize('stream', [False, True], ids=['complete', 'stream'])
@pytest.mark.parametrize(('raw', 'expected'), [('2', '2'), ('1', 1)])
def test_branch_matching_memoizes_shared_references(parser_cls, stream, raw, expected, monkeypatch):
    from lmdeploy.serve.parsers.tool_parser.xml_tool_parser import XmlToolParser

    levels = 12
    defs = {f'N{i}': {'anyOf': [{'$ref': f'#/$defs/N{i + 1}'} for _ in range(4)]}
            for i in range(levels)}
    defs[f'N{levels}'] = {'const': 1}
    schema = {'anyOf': [{'type': 'string'}, {'$ref': '#/$defs/N0'}]}
    calls = 0
    original = XmlToolParser._match_schema_value.__func__

    def counting(cls, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(XmlToolParser, '_match_schema_value', classmethod(counting))
    parameters, actual = _parse(parser_cls, schema, raw, stream, 'auto', defs)
    assert actual == {'value': expected}
    Draft202012Validator(parameters).validate(actual)
    # A cache hit must avoid re-expanding the four shared paths at each level.
    assert 0 < calls < 200
