# Copyright (c) OpenMMLab. All rights reserved.
"""Schema-aware coercion for hard-schema tool parameters.

Some tool suites declare parameters whose JSON Schema needs more than a
scalar ``type`` lookup: ``anyOf``/``oneOf``/``allOf`` combinations,
``$ref`` references, typeless object schemas and typeless enums. The model
emits values as bare JSON (or bare words) inside the XML value tags, and
the assembled ``arguments`` must both keep the parsed JSON value and
validate against the declared schema.
"""
import json

import pytest
from jsonschema import Draft202012Validator

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.tool_parser.glm47_tool_parser import Glm47ToolParser
from lmdeploy.serve.parsers.tool_parser.qwen3coder_tool_parser import Qwen3CoderToolParser

from .helpers import final_tool_call, stream_tool_arguments

QWEN_MODEL_ID = 'Qwen/Qwen3.5-35B-A3B'
GLM_MODEL_ID = 'zai-organization/GLM-5.3-Flash'

ANYOF_OBJECT_ARRAY = {
    'anyOf': [
        {'type': 'object', 'properties': {}},
        {'type': 'array', 'items': {'type': 'string'}},
    ],
}
ANYOF_STRING_INTEGER = {
    'anyOf': [
        {'type': 'string'},
        {'type': 'integer'},
    ],
}
ANYOF_NULLABLE_OBJECT = {
    'anyOf': [
        {'type': 'object', 'properties': {}},
        {'type': 'null'},
    ],
}
NULLABLE_INTEGER = {
    'anyOf': [
        {'type': 'integer'},
        {'type': 'null'},
    ],
}
NULLABLE_TYPE_LIST_STRING = {'type': ['string', 'null']}
NUMBER_TYPE = {'type': 'number'}
REF_OBJECT = {'$ref': '#/$defs/Shape'}
REF_STRING = {'$ref': '#/$defs/V'}
REF_EXTERNAL = {'$ref': 'https://example.com/schemas/value.json'}
REF_MISSING = {'$ref': '#/$defs/Missing'}
ENUM_STRING = {'enum': ['true', 'false']}
ENUM_INTEGER = {'enum': [1, 2]}
ENUM_NULLABLE_STRING = {'enum': ['a', None]}
TYPELESS_OBJECT = {'properties': {'kind': {'type': 'string'}}}
CONST_STRING_TRUE = {'const': 'true'}
CONST_NUMBER_STRING = {'const': '42'}
CONST_NULL = {'const': None}
REF_STRING_ESCAPED_SLASH = {'$ref': '#/$defs/A~1B'}
REF_STRING_ESCAPED_TILDE = {'$ref': '#/$defs/A~0B'}
SIBLING_TYPE_STRING = {'$ref': '#/$defs/Union', 'type': 'string'}
SIBLING_NULLABLE_STRING = {'$ref': '#/$defs/V', 'type': ['string', 'null']}
SIBLING_ENUM_STRING = {'$ref': '#/$defs/Empty', 'enum': ['true', 'false']}
ALLOF_STRING = {'allOf': [{'type': 'string'}]}
ALLOF_REF_MIN_LENGTH = {'allOf': [{'$ref': '#/$defs/Str'}, {'minLength': 1}]}
ALLOF_OBJECT = {'allOf': [{'type': 'object'}]}
ONEOF_STRING_INTEGER = {'oneOf': [{'type': 'string'}, {'type': 'integer'}]}
REF_SELFCYCLE = {'$ref': '#/$defs/Loop'}
REF_MUTUAL_CYCLE = {'$ref': '#/$defs/MutA'}
REF_INTERMEDIATE = {'$ref': '#/$defs/A'}
ANYOF_ENUM_STRING = {
    'anyOf': [{'type': 'string'}, {'type': 'boolean'}],
    'enum': ['true', 'false'],
}
ALLOF_CONST_STRING = {'allOf': [{'minLength': 1}], 'const': 'true'}
REF_NULLABLE_SIBLING_MIXED = {
    '$ref': '#/$defs/NV',
    'anyOf': [{'type': 'string'}, {'type': 'integer'}],
}
ENUM_MIXED = {'enum': ['true', 1]}

PARSERS = [
    pytest.param(Qwen3CoderToolParser, id='qwen3coder'),
    pytest.param(Glm47ToolParser, id='glm47'),
]


def _parameters(param_schema):
    return {
        'type': 'object',
        'properties': {'value': param_schema},
        'required': ['value'],
        '$defs': {
            'Shape': {
                'type': 'object',
                'properties': {'kind': {'type': 'string'}},
            },
            'V': {'type': 'string'},
            'Str': {'type': 'string'},
            'A/B': {'type': 'string'},
            'A~B': {'type': 'string'},
            'Union': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]},
            'Empty': {},
            'Loop': {'anyOf': [{'type': 'null'}, {'$ref': '#/$defs/Loop'}]},
            'MutA': {'$ref': '#/$defs/MutB'},
            'MutB': {'$ref': '#/$defs/MutA'},
            'A': {'$ref': '#/$defs/B', 'type': 'string'},
            'B': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]},
            'NV': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
        },
    }


def _request_with_parameters(parser_cls, parameters):
    model = GLM_MODEL_ID if parser_cls is Glm47ToolParser else QWEN_MODEL_ID
    return ChatCompletionRequest(
        model=model,
        messages=[{'role': 'user', 'content': 'submit the value'}],
        tools=[{
            'type': 'function',
            'function': {
                'name': 'submit_value',
                'parameters': parameters,
            },
        }],
        tool_choice='required',
    )


def _request(parser_cls, param_schema):
    return _request_with_parameters(parser_cls, _parameters(param_schema))


def _payload(parser_cls, raw_value):
    # ``feed_tool_block`` receives text AFTER the outer opening marker; the
    # qwen3coder format embeds the name marker inside its payload instead.
    if parser_cls is Glm47ToolParser:
        return f'submit_value<arg_key>value</arg_key><arg_value>{raw_value}</arg_value></tool_call>'
    return f'<function=submit_value>\n<parameter=value>{raw_value}</parameter>\n</function>'


def _chunks(parser_cls, raw_value):
    """Split the payload so the raw value itself arrives in two pieces."""
    if parser_cls is Glm47ToolParser:
        # The helper appends the outer close tag to the final chunk itself.
        return [
            'submit_value<arg_key>value</arg_key><arg_value>',
            raw_value[:1],
            raw_value[1:],
            '</arg_value>',
        ]
    return [
        '<function=submit_value>',
        '<parameter=value>',
        raw_value[:1],
        raw_value[1:],
        '</parameter>',
        '</function>',
    ]


def _parse_complete(parser_cls, param_schema, raw_value):
    parser = parser_cls()
    parser.adjust_request(_request(parser_cls, param_schema))
    tool_call = final_tool_call(parser, _payload(parser_cls, raw_value))
    assert tool_call.function.name == 'submit_value'
    return json.loads(tool_call.function.arguments)


COMPOSED_CASES = [
    pytest.param({'enum': ['true', 'false'], 'allOf': [{'enum': ['true']}]},
                 'true', 'true', id='intersect-enums'),
    pytest.param({'$ref': '#/$defs/Empty', 'enum': [None]},
                 'null', None, id='nullable-ref-empty-target'),
    pytest.param({'allOf': [{'type': ['integer', 'null']}, {'description': 'number'}]},
                 'null', None, id='nullable-allof-annotation'),
    pytest.param({'type': ['string', 'object']}, '{}', {}, id='mixed-type-list-object'),
    pytest.param({'anyOf': [{'type': 'string'}, {}]}, '[]', [], id='union-unconstrained-branch'),
    pytest.param({}, 'null', None, id='unconstrained-null'),
]


# schema, raw emitted value, expected parsed value
VALID_CASES = [
    *COMPOSED_CASES,
    # $ref -> string keeps bare numeric/boolean-looking text as strings
    pytest.param(REF_STRING, '42', '42', id='ref-string-bare-number'),
    pytest.param(REF_STRING, '"42"', '42', id='ref-string-quoted-number'),
    pytest.param(REF_STRING, 'true', 'true', id='ref-string-bare-boolean'),
    # typeless string enums keep bare boolean-looking text as strings
    pytest.param(ENUM_STRING, 'true', 'true', id='enum-string-true'),
    pytest.param(ENUM_STRING, 'false', 'false', id='enum-string-false'),
    pytest.param(ENUM_INTEGER, '1', 1, id='enum-integer-bare-number'),
    pytest.param(ENUM_NULLABLE_STRING, 'a', 'a', id='nullable-enum-string-value'),
    pytest.param(ENUM_NULLABLE_STRING, 'null', None, id='nullable-enum-string-null'),
    # nullable unions pass ``null`` through instead of stringifying it
    pytest.param(ANYOF_NULLABLE_OBJECT, 'null', None, id='nullable-object-null'),
    pytest.param(ANYOF_NULLABLE_OBJECT, '{}', {}, id='nullable-object-empty'),
    pytest.param(NULLABLE_INTEGER, 'null', None, id='nullable-integer-null'),
    pytest.param(NULLABLE_INTEGER, '3', 3, id='nullable-integer-value'),
    pytest.param(NULLABLE_TYPE_LIST_STRING, 'null', None, id='type-list-string-null'),
    pytest.param(NULLABLE_TYPE_LIST_STRING, 'hi', 'hi', id='type-list-string-value'),
    pytest.param(REF_OBJECT, '{"kind": "circle"}', {'kind': 'circle'}, id='ref-object'),
    # const-only schemas imply their constant's JSON type
    pytest.param(CONST_STRING_TRUE, 'true', 'true', id='const-string-bare-boolean'),
    pytest.param(CONST_STRING_TRUE, '"true"', 'true', id='const-string-quoted'),
    pytest.param(CONST_NUMBER_STRING, '42', '42', id='const-string-bare-number'),
    pytest.param(CONST_NULL, 'null', None, id='const-null'),
    # escaped JSON Pointer tokens resolve to the literal ``/``/``~`` keys
    pytest.param(REF_STRING_ESCAPED_SLASH, '42', '42', id='pointer-escape-slash'),
    pytest.param(REF_STRING_ESCAPED_TILDE, 'true', 'true', id='pointer-escape-tilde'),
    # constraints adjacent to a $ref apply alongside the referenced target
    pytest.param(SIBLING_TYPE_STRING, '42', '42', id='sibling-type-string'),
    pytest.param(SIBLING_TYPE_STRING, '"42"', '42', id='sibling-type-string-quoted'),
    pytest.param(SIBLING_ENUM_STRING, 'true', 'true', id='sibling-enum-string'),
    # sibling nullable + strict target: only the string form stays valid
    pytest.param(SIBLING_NULLABLE_STRING, 'null', 'null', id='sibling-nullable-target-strict'),
    pytest.param(SIBLING_NULLABLE_STRING, 'hi', 'hi', id='sibling-nullable-target-value'),
    # allOf members all constrain the value; agreeing types govern coercion
    pytest.param(ALLOF_STRING, '42', '42', id='allof-string-bare-number'),
    pytest.param(ALLOF_STRING, '"42"', '42', id='allof-string-quoted'),
    pytest.param(ALLOF_REF_MIN_LENGTH, 'hi', 'hi', id='allof-ref-minlength-string'),
    pytest.param(ALLOF_OBJECT, '{"kind": "circle"}', {'kind': 'circle'}, id='allof-object'),
    # oneOf mixed unions coerce bare JSON but keep bare null a string
    pytest.param(ONEOF_STRING_INTEGER, '5', 5, id='oneof-integer'),
    pytest.param(ONEOF_STRING_INTEGER, 'null', 'null', id='oneof-mixed-bare-null'),
    # bare null stays a string unless the schema accepts null
    pytest.param(ANYOF_STRING_INTEGER, 'null', 'null', id='mixed-union-bare-null'),
    pytest.param(TYPELESS_OBJECT, 'null', 'null', id='typeless-bare-null'),
    # quoted null decodes to the string 'null'
    pytest.param(NULLABLE_TYPE_LIST_STRING, '"null"', 'null', id='nullable-string-quoted-null'),
    pytest.param(ANYOF_STRING_INTEGER, '"null"', 'null', id='mixed-union-quoted-null'),
    # every $ref hop keeps its adjacent constraints: the intermediate layer's
    # type governs even when the chain continues past it
    pytest.param(REF_INTERMEDIATE, '42', '42', id='ref-intermediate-type-string'),
    pytest.param(REF_INTERMEDIATE, '"42"', '42', id='ref-intermediate-type-string-quoted'),
    # composition keywords merge with same-level enum/const constraints
    pytest.param(ANYOF_ENUM_STRING, 'true', 'true', id='anyof-enum-sibling-true'),
    pytest.param(ANYOF_ENUM_STRING, 'false', 'false', id='anyof-enum-sibling-false'),
    pytest.param(ALLOF_CONST_STRING, 'true', 'true', id='allof-const-sibling'),
    # a sibling union without null still forbids null beside a nullable $ref
    pytest.param(REF_NULLABLE_SIBLING_MIXED, 'null', 'null', id='ref-sibling-union-forbids-null'),
    pytest.param(REF_NULLABLE_SIBLING_MIXED, '42', '42', id='ref-sibling-union-string'),
    # mixed-type enums keep the interpretation that matches one literal
    pytest.param(ENUM_MIXED, 'true', 'true', id='mixed-enum-bare-boolean'),
    pytest.param(ENUM_MIXED, '1', 1, id='mixed-enum-bare-number'),
    pytest.param(ENUM_MIXED, '"true"', 'true', id='mixed-enum-quoted'),
]


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize(('schema', 'raw_value', 'expected'), VALID_CASES)
def test_arguments_validate_against_declared_schema(parser_cls, schema, raw_value, expected):
    """Parsed arguments must match the expectation and validate against the
    schema."""
    arguments = _parse_complete(parser_cls, schema, raw_value)
    assert arguments == {'value': expected}
    Draft202012Validator(_parameters(schema)).validate(arguments)


STREAM_CASES = [
    *COMPOSED_CASES,
    pytest.param(ANYOF_OBJECT_ARRAY, '{"kind": "circle", "tags": ["a", "b"]}',
                 {'kind': 'circle', 'tags': ['a', 'b']}, id='union-object-json'),
    pytest.param(REF_STRING, '42', '42', id='ref-string-bare-number'),
    pytest.param(REF_STRING, 'true', 'true', id='ref-string-bare-boolean'),
    pytest.param(ENUM_STRING, 'true', 'true', id='enum-string-true'),
    pytest.param(ANYOF_NULLABLE_OBJECT, 'null', None, id='nullable-object-null'),
    pytest.param(NULLABLE_INTEGER, 'null', None, id='nullable-integer-null'),
    # nullable string schemas must buffer in streaming so bare null survives
    pytest.param(NULLABLE_TYPE_LIST_STRING, 'null', None, id='type-list-string-null-stream'),
    pytest.param(ENUM_NULLABLE_STRING, 'null', None, id='nullable-enum-string-null-stream'),
    # const / escaped-pointer / $ref-sibling schemas stream like plain strings
    pytest.param(CONST_STRING_TRUE, 'true', 'true', id='const-string-bare-boolean-stream'),
    pytest.param(REF_STRING_ESCAPED_SLASH, '42', '42', id='pointer-escape-slash-stream'),
    pytest.param(SIBLING_TYPE_STRING, '42', '42', id='sibling-type-string-stream'),
    pytest.param(SIBLING_ENUM_STRING, 'true', 'true', id='sibling-enum-string-stream'),
    pytest.param(SIBLING_NULLABLE_STRING, 'null', 'null', id='sibling-nullable-target-strict-stream'),
    # a leading quote split across chunks still selects the buffered path
    pytest.param(ANYOF_STRING_INTEGER, '"null"', 'null', id='mixed-union-quoted-null-stream'),
    # allOf / oneOf coercions behave identically while streaming
    pytest.param(ALLOF_STRING, '42', '42', id='allof-string-bare-number-stream'),
    pytest.param(ONEOF_STRING_INTEGER, 'null', 'null', id='oneof-mixed-bare-null-stream'),
    pytest.param(ANYOF_STRING_INTEGER, 'null', 'null', id='mixed-union-bare-null-stream'),
    # non-finite numerics degrade to strings while streaming too
    pytest.param(ANYOF_STRING_INTEGER, 'NaN', 'NaN', id='mixed-union-nan-stream'),
    # intermediate $ref layers / enum siblings / mixed enums while streaming
    pytest.param(REF_INTERMEDIATE, '42', '42', id='ref-intermediate-type-string-stream'),
    pytest.param(ANYOF_ENUM_STRING, 'true', 'true', id='anyof-enum-sibling-stream'),
    pytest.param(REF_NULLABLE_SIBLING_MIXED, 'null', 'null', id='ref-sibling-union-forbids-null-stream'),
    pytest.param(ENUM_MIXED, 'true', 'true', id='mixed-enum-bare-boolean-stream'),
    pytest.param(ENUM_MIXED, '1', 1, id='mixed-enum-bare-number-stream'),
]


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize(('schema', 'raw_value', 'expected'), STREAM_CASES)
def test_streaming_matches_complete_parse(parser_cls, schema, raw_value, expected):
    """Chunked streaming must assemble the same schema-valid arguments."""
    parser = parser_cls()
    parser.adjust_request(_request(parser_cls, schema))
    streamed = stream_tool_arguments(parser, _chunks(parser_cls, raw_value))

    complete = _parse_complete(parser_cls, schema, raw_value)

    assert json.loads(streamed) == complete == {'value': expected}
    Draft202012Validator(_parameters(schema)).validate(json.loads(streamed))


# Unresolvable (mixed-union / typeless-object) schemas keep the bare JSON
# value as-is; emitted garbage degrades to its string form instead of erroring.
UNRESOLVED_CASES = [
    pytest.param(ANYOF_OBJECT_ARRAY, '{}', {'value': {}}, id='anyof-mixed-empty-object'),
    pytest.param(ANYOF_OBJECT_ARRAY, '{"kind": "circle", "tags": ["a"]}',
                 {'value': {'kind': 'circle', 'tags': ['a']}}, id='anyof-mixed-object'),
    pytest.param(ANYOF_OBJECT_ARRAY, '["a", "b"]', {'value': ['a', 'b']}, id='anyof-mixed-array'),
    pytest.param(ANYOF_NULLABLE_OBJECT, '{}', {'value': {}}, id='anyof-nullable-object'),
    pytest.param(TYPELESS_OBJECT, '{"kind": "circle", "radius": 2}',
                 {'value': {'kind': 'circle', 'radius': 2}}, id='typeless-object'),
    pytest.param(ANYOF_OBJECT_ARRAY, '"literal"', {'value': 'literal'}, id='quoted-string-unwraps'),
    pytest.param(ANYOF_OBJECT_ARRAY, 'plain words', {'value': 'plain words'}, id='plain-words-stay-string'),
]


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize(('schema', 'raw_value', 'expected'), UNRESOLVED_CASES)
def test_unresolved_schema_keeps_json_value(parser_cls, schema, raw_value, expected):
    """Bare JSON values must survive coercion, not become strings."""
    assert _parse_complete(parser_cls, schema, raw_value) == expected


def _strict_loads(text: str):
    """json.loads rejecting NaN/Infinity, like strict client parsers."""
    def _reject(token: str):
        raise ValueError(token)

    return json.loads(text, parse_constant=_reject)


# Non-finite numerics must degrade to their string form so the assembled
# arguments stay parseable by strict JSON clients.
STRICT_JSON_CASES = [
    pytest.param(NUMBER_TYPE, 'NaN', 'NaN', id='number-nan'),
    pytest.param(NUMBER_TYPE, 'Infinity', 'Infinity', id='number-infinity'),
    pytest.param(NUMBER_TYPE, '1e999', '1e999', id='number-overflow-exponent'),
    pytest.param(ANYOF_STRING_INTEGER, 'NaN', 'NaN', id='mixed-union-nan'),
    pytest.param(ANYOF_STRING_INTEGER, '{"a": NaN}', '{"a": NaN}', id='nested-nan-object'),
]


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize(('schema', 'raw_value', 'expected'), STRICT_JSON_CASES)
def test_arguments_remain_strict_json(parser_cls, schema, raw_value, expected):
    """Non-finite numerics degrade to strings so ``arguments`` stays strict
    JSON."""
    parser = parser_cls()
    parser.adjust_request(_request(parser_cls, schema))
    tool_call = final_tool_call(parser, _payload(parser_cls, raw_value))
    assert _strict_loads(tool_call.function.arguments) == {'value': expected}


# Cyclic or unresolvable $ref graphs must degrade deterministically to the
# string-preserving fallback instead of recursing without bound; these
# schemas are pathological so their outputs are only checked for stable
# complete/streaming equivalence, not schema validity.
DEGENERATE_REF_CASES = [
    pytest.param(REF_SELFCYCLE, 'null', None, id='anyof-self-cycle-null'),
    pytest.param(REF_SELFCYCLE, 'hi', 'hi', id='anyof-self-cycle-string'),
    pytest.param(REF_MUTUAL_CYCLE, '42', '42', id='mutual-cycle-string'),
    pytest.param(REF_MISSING, '{"a": 1}', '{"a": 1}', id='missing-target-object'),
    pytest.param(REF_EXTERNAL, '42', '42', id='external-ref-string'),
]


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize(('schema', 'raw_value', 'expected'), DEGENERATE_REF_CASES)
def test_degenerate_refs_degrade_without_recursing_forever(parser_cls, schema, raw_value, expected):
    """Cyclic/missing/external $refs never raise RecursionError and stream
    exactly like the complete parse."""
    complete = _parse_complete(parser_cls, schema, raw_value)
    assert complete == {'value': expected}

    parser = parser_cls()
    parser.adjust_request(_request(parser_cls, schema))
    streamed = stream_tool_arguments(parser, _chunks(parser_cls, raw_value))
    assert json.loads(streamed) == complete


@pytest.mark.parametrize('parser_cls', PARSERS)
def test_very_long_ref_chain_is_bounded(parser_cls):
    """A ~1500-hop non-cyclic $ref chain resolves within the depth budget
    instead of exhausting the Python stack."""
    hops = 1500
    parameters = _parameters({'$ref': '#/$defs/N0'})
    for i in range(hops):
        parameters['$defs'][f'N{i}'] = {'$ref': f'#/$defs/N{i + 1}'}
    parameters['$defs'][f'N{hops}'] = {'type': 'string'}

    parser = parser_cls()
    parser.adjust_request(_request_with_parameters(parser_cls, parameters))
    tool_call = final_tool_call(parser, _payload(parser_cls, '42'))
    assert json.loads(tool_call.function.arguments) == {'value': '42'}

    parser = parser_cls()
    parser.adjust_request(_request_with_parameters(parser_cls, parameters))
    streamed = stream_tool_arguments(parser, _chunks(parser_cls, '42'))
    assert json.loads(streamed) == {'value': '42'}


@pytest.mark.parametrize('parser_cls', PARSERS)
def test_shared_ref_union_targets_resolve_once(parser_cls, monkeypatch):
    """Unions whose branches share ``$ref`` targets cost node-count work.

    A 20-level doubling diamond has 2**20 root-to-leaf paths; without memoization every path is re-expanded inside the
    serving loop.
    """
    from lmdeploy.serve.parsers.tool_parser.xml_tool_parser import XmlToolParser

    levels = 20
    parameters = _parameters({'$ref': '#/$defs/E0'})
    for i in range(levels):
        parameters['$defs'][f'E{i}'] = {
            'anyOf': [{'$ref': f'#/$defs/E{i + 1}'}, {'$ref': f'#/$defs/E{i + 1}'}],
        }
    parameters['$defs'][f'E{levels}'] = {'type': 'string'}

    calls = []
    original = XmlToolParser._resolve_schema_impl.__func__

    def counting(cls, schema, root_schema, depth, cache, visits):
        calls.append(schema)
        return original(cls, schema, root_schema, depth, cache, visits)

    monkeypatch.setattr(XmlToolParser, '_resolve_schema_impl', classmethod(counting))

    parser = parser_cls()
    parser.adjust_request(_request_with_parameters(parser_cls, parameters))
    tool_call = final_tool_call(parser, _payload(parser_cls, '42'))
    assert json.loads(tool_call.function.arguments) == {'value': '42'}

    # The streaming decision and the argument finish each resolve once; both
    # stay in the order of magnitude of the ~60 unique schema nodes.
    assert len(calls) < 200


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize('leaf', [{'const': '42'}, {'enum': ['42', '43']}])
def test_shared_ref_enum_expansion_is_bounded(parser_cls, leaf):
    """A small reference graph must not materialize every root-to-leaf path."""
    levels = 10
    parameters = _parameters({'$ref': '#/$defs/E0'})
    for i in range(levels):
        parameters['$defs'][f'E{i}'] = {
            'anyOf': [{'$ref': f'#/$defs/E{i + 1}'} for _ in range(4)],
        }
    parameters['$defs'][f'E{levels}'] = leaf
    schema = parameters['properties']['value']
    expected = tuple(leaf.get('enum', ['42']))
    assert parser_cls._resolve_schema_coercion(schema, parameters) == ('string', False, expected)

    for stream in (False, True):
        parser = parser_cls()
        parser.adjust_request(_request_with_parameters(parser_cls, parameters))
        arguments = (stream_tool_arguments(parser, _chunks(parser_cls, '42')) if stream else
                     final_tool_call(parser, _payload(parser_cls, '42')).function.arguments)
        assert json.loads(arguments) == {'value': '42'}


@pytest.mark.parametrize('composition', ['anyOf', 'oneOf'])
def test_union_enum_dedup_preserves_json_literals(composition):
    """Deduplicate equal JSON values without losing boolean or container
    literals."""
    values = [True, 1, '1', None, [True], [1], {'a': True}, {'a': 1}, {'x': 1, 'y': 2}, 2**53, 2**53 + 1]
    schema = {composition: [{'enum': values}, {'enum': [*values, 1.0, float(2**53), {'y': 2, 'x': 1}]}]}
    _, _, literals = Glm47ToolParser._resolve_schema_coercion(schema)
    assert json.dumps(literals) == json.dumps(values)


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize('reference', [False, True])
def test_enum_work_budget_preserves_raw_value(parser_cls, reference, monkeypatch):
    """Oversized enum processing must stop before guessing a different JSON
    type."""
    from lmdeploy.serve.parsers.tool_parser import xml_tool_parser

    monkeypatch.setattr(xml_tool_parser, '_MAX_SCHEMA_LITERAL_WORK', 32, raising=False)
    large_enum = {'enum': list(range(128))}
    schema = {'$ref': '#/$defs/Large'} if reference else large_enum
    parameters = _parameters(schema)
    parameters['$defs']['Large'] = large_enum
    for stream in (False, True):
        parser = parser_cls()
        parser.adjust_request(_request_with_parameters(parser_cls, parameters))
        arguments = (stream_tool_arguments(parser, _chunks(parser_cls, '42')) if stream else
                     final_tool_call(parser, _payload(parser_cls, '42')).function.arguments)
        assert json.loads(arguments) == {'value': '42'}


def test_literal_budget_is_shared_across_union_branches(monkeypatch):
    from lmdeploy.serve.parsers.tool_parser import xml_tool_parser

    monkeypatch.setattr(xml_tool_parser, '_MAX_SCHEMA_LITERAL_WORK', 32, raising=False)
    # Each leaf fits by itself; processing all leaves and merging their values does not.
    schema = {'anyOf': [{'enum': list(range(i, i + 8))} for i in range(0, 32, 8)]}
    assert Glm47ToolParser._resolve_schema_coercion(schema) == ('string', False, None)


def test_literal_budget_covers_intersections(monkeypatch):
    from lmdeploy.serve.parsers.tool_parser import xml_tool_parser

    monkeypatch.setattr(xml_tool_parser, '_MAX_SCHEMA_LITERAL_WORK', 32)
    # Collecting these leaves fits in 32 steps; intersecting them exceeds the shared budget.
    schema = {'allOf': [{'enum': list(range(8))} for _ in range(3)]}
    assert Glm47ToolParser._resolve_schema_coercion(schema) == ('string', False, None)


@pytest.mark.parametrize('keyword', ['const', 'enum'])
@pytest.mark.parametrize('deep', [False, True])
def test_nested_literal_processing_is_bounded(keyword, deep, monkeypatch):
    from lmdeploy.serve.parsers.tool_parser import xml_tool_parser

    if deep:
        value = 42
        for _ in range(64):
            value = [value]
    else:
        monkeypatch.setattr(xml_tool_parser, '_MAX_SCHEMA_LITERAL_WORK', 32)
        value = list(range(128))
    schema = {keyword: [value] if keyword == 'enum' else value}
    assert Glm47ToolParser._resolve_schema_coercion(schema) == ('string', False, None)


def test_enum_intersection_fits_linear_work_budget(monkeypatch):
    from lmdeploy.serve.parsers.tool_parser import xml_tool_parser

    monkeypatch.setattr(xml_tool_parser, '_MAX_SCHEMA_LITERAL_WORK', 4096)
    schema = {'allOf': [{'enum': list(range(512))}, {'enum': list(range(256, 768))}]}
    assert Glm47ToolParser._resolve_schema_coercion(schema) == ('integer', False, tuple(range(256, 512)))


def test_enum_intersection_uses_json_equality():
    schema = {'allOf': [{'enum': [True, 1, [True], [1], {'a': True}, {'a': 1}]},
                        {'enum': [1.0, [1.0], {'a': 1.0}]}]}
    _, _, literals = Glm47ToolParser._resolve_schema_coercion(schema)
    assert json.dumps(literals) == json.dumps((1, [1], {'a': 1}))


@pytest.mark.parametrize('schema', [{'enum': [1, 1.0]}, {'enum': [1.0, 1]}])
def test_numeric_enum_dedup_preserves_type_inference(schema):
    assert _parse_complete(Glm47ToolParser, schema, '1.0') == {'value': 1.0}


TYPE_AND_LITERAL_CASES = [
    pytest.param({'type': ['string', 'integer']}, 'true', 'true', id='type-list-excludes-boolean'),
    pytest.param({'type': ['string', 'integer']}, '1.5', '1.5', id='type-list-excludes-fraction'),
    pytest.param({'type': ['string', 'object']}, '42', '42', id='type-list-excludes-number'),
    pytest.param({'type': ['string', 'array']}, '{}', '{}', id='type-list-excludes-object'),
    pytest.param({'anyOf': [{'type': 'string'}, {'type': 'integer'}]},
                 'true', 'true', id='anyof-excludes-boolean'),
    pytest.param({'oneOf': [{'type': 'string'}, {'type': 'integer'}]},
                 'true', 'true', id='oneof-excludes-boolean'),
    pytest.param({'$ref': '#/$defs/Union'}, 'true', 'true', id='referenced-union-excludes-boolean'),
    pytest.param({'type': ['string', 'integer'], 'enum': ['true', True]},
                 'true', 'true', id='type-list-intersects-enum'),
    pytest.param({'allOf': [{'type': ['string', 'integer']}, {'type': ['string', 'boolean']}]},
                 'true', 'true', id='allof-intersects-type-sets'),
    pytest.param({'$ref': '#/$defs/Union', 'type': ['string', 'boolean']},
                 'true', 'true', id='ref-sibling-intersects-type-sets'),
    pytest.param({'anyOf': [{'type': ['string', 'integer']}, {'type': 'object'}]},
                 'true', 'true', id='nested-union-excludes-boolean'),
    pytest.param({'type': ['string', 'integer', 'null']},
                 'true', 'true', id='nullable-mixed-excludes-boolean'),
    pytest.param({'enum': ['[true]', [1]]}, '[true]', '[true]', id='nested-array-bool-number'),
    pytest.param({'enum': ['[1]', [True]]}, '[1]', '[1]', id='nested-array-number-bool'),
    pytest.param({'enum': ['{"a":true}', {'a': 1}]},
                 '{"a":true}', '{"a":true}', id='nested-object-bool-number'),
    pytest.param({'enum': ['{"a":[true]}', {'a': [1]}]},
                 '{"a":[true]}', '{"a":[true]}', id='nested-object-array-bool-number'),
    pytest.param({'anyOf': [{'const': '[true]'}, {'const': [1]}]},
                 '[true]', '[true]', id='const-union-nested-bool-number'),
    # Allowed native values and unconstrained branches must remain usable.
    pytest.param({'type': ['string', 'integer']}, '42', 42, id='mixed-keeps-integer'),
    pytest.param({'type': ['string', 'integer']}, '1.0', 1.0, id='mixed-keeps-integral-number'),
    pytest.param({'type': ['string', 'number']}, '42', 42, id='number-includes-integer'),
    pytest.param({'type': ['string', 'number']}, '1.5', 1.5, id='mixed-keeps-number'),
    pytest.param({'type': ['string', 'boolean']}, 'true', True, id='mixed-keeps-boolean'),
    pytest.param({'type': ['string', 'object']}, '{}', {}, id='mixed-keeps-object'),
    pytest.param({'type': ['string', 'array']}, '[1]', [1], id='mixed-keeps-array'),
    pytest.param({'type': ['string', 'integer', 'null']}, 'null', None, id='mixed-keeps-null'),
    pytest.param({'anyOf': [{'type': ['string', 'integer']}, {}]},
                 'true', True, id='unconstrained-union-keeps-boolean'),
    pytest.param({'allOf': [{'type': ['string', 'integer']}, {}]},
                 '42', 42, id='unconstrained-intersection-keeps-integer'),
    pytest.param({'allOf': [{'type': ['string', 'integer']}, {'type': ['string', 'number']}]},
                 '42', 42, id='intersection-number-includes-integer'),
    pytest.param({'allOf': [{'type': ['string', 'integer']}, {'type': ['string', 'number']}]},
                 '1.5', '1.5', id='intersection-excludes-fraction'),
    pytest.param({'enum': ['[true]', [1]]}, '[1]', [1], id='nested-array-valid'),
    pytest.param({'enum': ['[true]', [1]]}, '[1.0]', [1.0], id='nested-array-numeric-equality'),
    pytest.param({'enum': ['{"a":true}', {'a': 1, 'b': [2]}]},
                 '{"b":[2.0],"a":1.0}', {'a': 1.0, 'b': [2.0]}, id='nested-object-numeric-equality'),
    pytest.param({'enum': ['[{"a":true}]', [{'a': True}]]},
                 '[{"a":true}]', [{'a': True}], id='nested-boolean-valid'),
]


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize('stream', [False, True], ids=['complete', 'stream'])
@pytest.mark.parametrize(('schema', 'raw_value', 'expected'), TYPE_AND_LITERAL_CASES)
def test_coercion_preserves_declared_types_and_json_literals(parser_cls, stream, schema, raw_value, expected):
    """A valid string interpretation must survive an invalid native JSON
    interpretation."""
    parameters = _parameters(schema)
    Draft202012Validator.check_schema(parameters)
    validator = Draft202012Validator(parameters)
    validator.validate({'value': expected})
    parser = parser_cls()
    parser.adjust_request(_request_with_parameters(parser_cls, parameters))
    arguments = (stream_tool_arguments(parser, _chunks(parser_cls, raw_value)) if stream else
                 final_tool_call(parser, _payload(parser_cls, raw_value)).function.arguments)
    actual = json.loads(arguments)
    assert actual == {'value': expected}
    validator.validate(actual)


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize('stream', [False, True], ids=['complete', 'stream'])
def test_final_enum_matching_shares_literal_budget(parser_cls, stream, monkeypatch):
    from lmdeploy.serve.parsers.tool_parser import xml_tool_parser

    # Resolving the two candidates takes four steps. Matching the value would
    # exceed the remaining budget, so preserve its raw string interpretation.
    monkeypatch.setattr(xml_tool_parser, '_MAX_SCHEMA_LITERAL_WORK', 5)
    schema = {'enum': ['[1,2]', [1, 2]]}
    parser = parser_cls()
    parser.adjust_request(_request(parser_cls, schema))
    arguments = (stream_tool_arguments(parser, _chunks(parser_cls, '[1,2]')) if stream else
                 final_tool_call(parser, _payload(parser_cls, '[1,2]')).function.arguments)
    assert json.loads(arguments) == {'value': '[1,2]'}


MULTI_PARAM_PARAMETERS = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'payload': ANYOF_OBJECT_ARRAY,
    },
    'required': ['name', 'payload'],
}


@pytest.mark.parametrize('parser_cls', PARSERS)
def test_interleaved_params_mix_streaming_and_buffered(parser_cls):
    """A streamed string parameter followed by a buffered object parameter
    assemble in order, including when fragments span chunks."""
    expected = {'name': 'Ada Lovelace', 'payload': {'kind': 'circle'}}
    if parser_cls is Glm47ToolParser:
        payload = ('submit_value<arg_key>name</arg_key><arg_value>Ada Lov'
                   'elace</arg_value><arg_key>payload</arg_key><arg_value>'
                   '{"kind": "circle"}</arg_value></tool_call>')
        chunks = [
            'submit_value<arg_key>name</arg_key><arg_value>Ada Lov',
            'elace</arg_value><arg_key>payload</arg_key><arg_value>{"kind": ',
            '"circle"}</arg_value>',
        ]
    else:
        payload = ('<function=submit_value>\n<parameter=name>Ada Lovelace</parameter>\n'
                   '<parameter=payload>{"kind": "circle"}</parameter>\n</function>')
        chunks = [
            '<function=submit_value>',
            '<parameter=name>Ada Lov',
            'elace</parameter>',
            '<parameter=payload>{"kind": ',
            '"circle"}</parameter>',
        ]

    parser = parser_cls()
    parser.adjust_request(_request_with_parameters(parser_cls, MULTI_PARAM_PARAMETERS))
    tool_call = final_tool_call(parser, payload)
    assert json.loads(tool_call.function.arguments) == expected

    parser = parser_cls()
    parser.adjust_request(_request_with_parameters(parser_cls, MULTI_PARAM_PARAMETERS))
    streamed = stream_tool_arguments(parser, chunks)
    assert json.loads(streamed) == expected
    Draft202012Validator(MULTI_PARAM_PARAMETERS).validate(json.loads(streamed))
