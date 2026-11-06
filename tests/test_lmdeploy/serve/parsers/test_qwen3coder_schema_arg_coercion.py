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


# schema, raw emitted value, expected parsed value
VALID_CASES = [
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
