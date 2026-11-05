# Copyright (c) OpenMMLab. All rights reserved.
"""Schema-aware coercion for hard-schema (KVV-style) tool parameters.

Hard-schema suites declare parameters whose JSON Schema cannot be resolved
to a single scalar type (``anyOf``/``oneOf`` unions, ``$ref`` references,
typeless object schemas). The model emits the value as bare JSON inside the
XML ``<parameter>`` tags, and the assembled ``arguments`` must keep the
parsed JSON value instead of a stringified copy (GLM-5.3-Flash 1030 report
F2 / the Sage-A1.5 + Agents-A1 overlapping failures).
"""
import json

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.tool_parser.qwen3coder_tool_parser import Qwen3CoderToolParser

from .helpers import final_tool_call, stream_tool_arguments

MODEL_ID = 'Qwen/Qwen3.5-35B-A3B'

ANYOF_OBJECT_ARRAY = {
    'anyOf': [
        {'type': 'object', 'properties': {}},
        {'type': 'array', 'items': {'type': 'string'}},
    ],
}
ANYOF_NULLABLE_OBJECT = {
    'anyOf': [
        {'type': 'object', 'properties': {}},
        {'type': 'null'},
    ],
}
REF_OBJECT = {'$ref': '#/$defs/Shape'}
TYPELESS_OBJECT = {'properties': {'kind': {'type': 'string'}}}


def _request(param_schema):
    return ChatCompletionRequest(
        model=MODEL_ID,
        messages=[{'role': 'user', 'content': 'submit the value'}],
        tools=[{
            'type': 'function',
            'function': {
                'name': 'submit_value',
                'parameters': {
                    'type': 'object',
                    'properties': {'value': param_schema},
                    'required': ['value'],
                    '$defs': {
                        'Shape': {
                            'type': 'object',
                            'properties': {'kind': {'type': 'string'}},
                        },
                    },
                },
            },
        }],
        tool_choice='required',
    )


def _payload(raw_value):
    return f'<function=submit_value>\n<parameter=value>{raw_value}</parameter>\n</function>'


@pytest.mark.parametrize(
    ('schema', 'raw_value', 'expected'),
    [
        pytest.param(ANYOF_OBJECT_ARRAY, '{}', {'value': {}}, id='anyof-mixed-empty-object'),
        pytest.param(
            ANYOF_OBJECT_ARRAY,
            '{"kind": "circle", "tags": ["a"]}',
            {'value': {'kind': 'circle', 'tags': ['a']}},
            id='anyof-mixed-object'),
        pytest.param(ANYOF_OBJECT_ARRAY, '["a", "b"]', {'value': ['a', 'b']}, id='anyof-mixed-array'),
        pytest.param(ANYOF_NULLABLE_OBJECT, '{}', {'value': {}}, id='anyof-nullable-object'),
        pytest.param(REF_OBJECT, '{"kind": "circle"}', {'value': {'kind': 'circle'}}, id='dollar-ref-object'),
        pytest.param(
            TYPELESS_OBJECT,
            '{"kind": "circle", "radius": 2}',
            {'value': {'kind': 'circle', 'radius': 2}},
            id='typeless-object'),
        pytest.param(ANYOF_OBJECT_ARRAY, '"literal"', {'value': 'literal'}, id='quoted-string-unwraps'),
        pytest.param(ANYOF_OBJECT_ARRAY, 'plain words', {'value': 'plain words'}, id='plain-words-stay-string'),
    ],
)
def test_unresolved_schema_keeps_json_value(schema, raw_value, expected):
    """Bare JSON values must survive coercion, not become strings."""
    parser = Qwen3CoderToolParser()
    parser.adjust_request(_request(schema))

    tool_call = final_tool_call(parser, _payload(raw_value))

    assert tool_call.function.name == 'submit_value'
    assert json.loads(tool_call.function.arguments) == expected


def test_streaming_matches_complete_parse_for_union_schema():
    """Chunked streaming must assemble the same arguments as complete parse."""
    parser = Qwen3CoderToolParser()
    parser.adjust_request(_request(ANYOF_OBJECT_ARRAY))
    chunks = [
        '<function=submit_value>',
        '<parameter=value>',
        '{"kind": "cir',
        'cle", "tags": ["a",',
        ' "b"]}',
        '</parameter>',
        '</function>',
    ]

    streamed = stream_tool_arguments(parser, chunks)

    complete_parser = Qwen3CoderToolParser()
    complete_parser.adjust_request(_request(ANYOF_OBJECT_ARRAY))
    complete = final_tool_call(
        complete_parser, _payload('{"kind": "circle", "tags": ["a", "b"]}'))

    assert streamed == complete.function.arguments
    assert json.loads(streamed) == {'value': {'kind': 'circle', 'tags': ['a', 'b']}}
