# Copyright (c) OpenMMLab. All rights reserved.
"""Preserve valid argument interpretations and local schema resource scope."""
import json

import pytest
from jsonschema import Draft202012Validator

from .test_xml_schema_combinators import PARSERS, _parse

MINIMUM = {'anyOf': [{'type': 'string'}, {'type': 'integer', 'minimum': 5}]}
MAXIMUM = {'anyOf': [{'type': 'string'}, {'type': 'integer', 'maximum': 10}]}
REQUIRED = {'anyOf': [{'type': 'string'}, {'type': 'object', 'required': ['id']}]}
ITEMS = {'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}]}

CASES = [
    pytest.param(MINIMUM, '2', '2', id='unknown-minimum-prefers-valid-string'),
    pytest.param(MAXIMUM, '42', '42', id='unknown-maximum-prefers-valid-string'),
    pytest.param(REQUIRED, '{}', '{}', id='unknown-required-prefers-valid-string'),
    pytest.param(ITEMS, '[1]', '[1]', id='unknown-items-prefers-valid-string'),
    # Even legal native candidates stay strings when only the string branch
    # can be proved valid by this coercion check.
    pytest.param(MINIMUM, '7', '7', id='inconclusive-valid-number-prefers-string'),
    pytest.param(REQUIRED, '{"id":1}', '{"id":1}', id='inconclusive-valid-object-prefers-string'),
    pytest.param(ITEMS, '["a"]', '["a"]', id='inconclusive-valid-array-prefers-string'),
    pytest.param({'oneOf': [{'type': 'string'}, {'type': 'integer', 'minimum': 5}]},
                 '2', '2', id='oneof-inconclusive-number'),
    pytest.param({'allOf': [MINIMUM, {}]}, '2', '2', id='nested-inconclusive-number'),
    # Keep native interpretations when they are definite, or strings cannot
    # satisfy the schema. Do not turn every inconclusive match into a string.
    pytest.param({'anyOf': [{'type': 'string'}, {'type': 'integer'}]}, '2', 2, id='definite-number'),
    pytest.param({'anyOf': [{'type': 'string'}, {'const': 2}]}, '2', 2, id='definite-constant'),
    pytest.param({'anyOf': [{'type': 'object', 'required': ['id']}, {'type': 'array'}]},
                 '{"id":1}', {'id': 1}, id='inconclusive-object-without-string'),
    pytest.param({'oneOf': [{'type': 'object', 'required': ['a']}, {'type': 'object', 'required': ['b']}]},
                 '{"a":1}', {'a': 1}, id='inconclusive-oneof-without-string'),
    pytest.param({'type': 'number', 'enum': [1, 2]}, '1.0', 1.0, id='number-enum-integral-float'),
    pytest.param({'type': 'number', 'enum': [1, 2]}, '1e0', 1.0, id='number-enum-integral-exponent'),
    pytest.param({'type': 'integer'}, '1.0', 1.0, id='integer-integral-float'),
    pytest.param({'type': 'integer'}, '1e0', 1.0, id='integer-integral-exponent'),
    pytest.param({'enum': [1, 2]}, '1.0', 1.0, id='typeless-enum-integral-float'),
    pytest.param({'type': 'number', 'const': 1}, '1e0', 1.0, id='number-const-integral-exponent'),
    pytest.param({'type': ['integer', 'string']}, '1.5', '1.5', id='fraction-is-not-integer'),
    pytest.param({'type': ['integer', 'string']}, 'true', 'true', id='boolean-is-not-integer'),
    pytest.param({'type': ['integer', 'string']}, '1e999', '1e999', id='nonfinite-is-not-integer'),
    pytest.param({'enum': ['NULL', None]}, 'NULL', 'NULL', id='uppercase-null-literal'),
    pytest.param({'enum': ['Null', None]}, 'Null', 'Null', id='mixedcase-null-literal'),
    pytest.param({'enum': ['NULL', None]}, 'null', None, id='lowercase-null-value'),
    pytest.param({'enum': ['NULL', None]}, '"NULL"', 'NULL', id='quoted-uppercase-null'),
    pytest.param({'enum': ['null', None]}, 'null', None, id='ambiguous-lowercase-null'),
    pytest.param({'enum': ['null', None]}, '"null"', 'null', id='quoted-lowercase-null'),
    pytest.param({'anyOf': [{'const': 'NULL'}, {'type': 'null'}]},
                 'NULL', 'NULL', id='uppercase-null-union'),
    pytest.param({'anyOf': [{'const': 'NULL'}, {'type': 'null'}]},
                 'null', None, id='lowercase-null-union'),
    pytest.param({'enum': ['NULL', None], 'maxLength': 4},
                 'NULL', 'NULL', id='uppercase-null-with-opaque-constraint'),
    pytest.param({'type': 'null'}, 'NULL', None, id='null-only-keeps-tolerant-coercion'),
    pytest.param({'type': ['object', 'null']}, 'NULL', None, id='nullable-object-keeps-tolerant-coercion'),
    pytest.param({'enum': ['other', None]}, 'NULL', None, id='unmatched-uppercase-null'),
]


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize('stream', [False, True], ids=['complete', 'stream'])
@pytest.mark.parametrize('tool_choice', ['auto', 'required'])
@pytest.mark.parametrize(('schema', 'raw', 'expected'), CASES)
def test_schema_candidate_selection(parser_cls, stream, tool_choice, schema, raw, expected):
    parameters, actual = _parse(parser_cls, schema, raw, stream, tool_choice)
    Draft202012Validator.check_schema(parameters)
    validator = Draft202012Validator(parameters)
    validator.validate({'value': expected})
    validator.validate(actual)
    # In particular, both NULL and null are schema-valid but distinct choices.
    assert actual == {'value': expected}
    assert type(actual['value']) is type(expected)


RESOURCE_CASES = [
    pytest.param({'$id': 'https://example.test/embedded', '$defs': {'Kind': {'type': 'string'}},
                  '$ref': '#/$defs/Kind'}, {'Kind': {'type': 'integer'}}, '42', '42', id='direct-resource'),
    pytest.param({'$id': 'embedded', '$defs': {'Kind': {'type': 'string'}},
                  '$ref': '#/$defs/Kind'}, {'Kind': {'type': 'integer'}}, '42', '42', id='relative-resource-id'),
    pytest.param({'$id': 'https://example.test/embedded', '$defs': {'Kind': {'const': 1}},
                  'anyOf': [{'type': 'string'}, {'$ref': '#/$defs/Kind'}]},
                 {'Kind': {'type': 'integer'}}, '2', '2', id='branch-matching-resource'),
    pytest.param({'$id': 'https://example.test/embedded', '$defs': {'Kind': {'const': 1}},
                  'anyOf': [{'type': 'string'}, {'$ref': '#/$defs/Kind'}]},
                 {'Kind': {'type': 'integer'}}, '1', 1, id='branch-matching-resource-control'),
    pytest.param({'$ref': '#/$defs/Embedded/$defs/Value'},
                 {'Kind': {'type': 'integer'},
                  'Embedded': {'$id': 'https://example.test/embedded',
                               '$defs': {'Kind': {'type': 'string'}, 'Value': {'$ref': '#/$defs/Kind'}}}},
                 '42', '42', id='pointer-crosses-resource-boundary'),
    pytest.param({'$ref': '#/$defs/Embedded'},
                 {'Kind': {'type': 'integer'},
                  'Embedded': {'$id': 'https://example.test/embedded', '$defs': {'Kind': {'type': 'string'}},
                               '$ref': '#/$defs/Kind'}}, '42', '42', id='reference-target-starts-resource'),
    pytest.param({'anyOf': [
        {'$id': 'https://example.test/text', '$defs': {'Kind': {'type': 'string'}}, '$ref': '#/$defs/Kind'},
        {'$id': 'https://example.test/number', '$defs': {'Kind': {'const': 1}}, '$ref': '#/$defs/Kind'},
    ]}, {'Kind': {'type': 'integer'}}, '2', '2', id='sibling-resource-scope-does-not-leak'),
    pytest.param({'$id': 'https://example.test/embedded', '$defs': {'Kind': {'type': 'string'}},
                  'allOf': [{'$ref': '#/$defs/Kind'}, {'minLength': 1}]},
                 {'Kind': {'type': 'integer'}}, '42', '42', id='allof-inherits-resource'),
    pytest.param({'$ref': '#/$defs/Kind'}, {'Kind': {'type': 'integer'}},
                 '42', 42, id='root-resource-control'),
]


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize('stream', [False, True], ids=['complete', 'stream'])
@pytest.mark.parametrize('tool_choice', ['auto', 'required'])
@pytest.mark.parametrize(('schema', 'defs', 'raw', 'expected'), RESOURCE_CASES)
def test_local_reference_resource_scope(parser_cls, stream, tool_choice, schema, defs, raw, expected):
    parameters, actual = _parse(parser_cls, schema, raw, stream, tool_choice, defs, 'https://example.test/root')
    Draft202012Validator.check_schema(parameters)
    validator = Draft202012Validator(parameters)
    validator.validate({'value': expected})
    validator.validate(actual)
    assert actual == {'value': expected}


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize('stream', [False, True], ids=['complete', 'stream'])
def test_shared_schema_cache_respects_resource(parser_cls, stream):
    shared_ref = {'$ref': '#/$defs/Kind'}
    defs = {
        'A': {'$id': 'https://example.test/a', '$defs': {'Kind': {'type': 'integer'}, 'Value': shared_ref}},
        'B': {'$id': 'https://example.test/b', '$defs': {'Kind': {'type': 'boolean'}, 'Value': shared_ref}},
    }
    schema = {'oneOf': [{'$ref': '#/$defs/A/$defs/Value'}, {'$ref': '#/$defs/B/$defs/Value'}]}
    parameters, actual = _parse(parser_cls, schema, '1', stream, 'auto', defs, 'https://example.test/root')
    Draft202012Validator(parameters).validate(actual)
    assert actual == {'value': 1}


@pytest.mark.parametrize('parser_cls', PARSERS)
@pytest.mark.parametrize('stream', [False, True], ids=['complete', 'stream'])
def test_raw_candidate_check_uses_remaining_budget(parser_cls, stream, monkeypatch):
    from lmdeploy.serve.parsers.tool_parser import xml_tool_parser

    # The native and raw candidates use separate memo tables, but share work.
    monkeypatch.setattr(xml_tool_parser, '_MAX_SCHEMA_RESOLVE_VISITS', 7)
    _, actual = _parse(parser_cls, MINIMUM, '2', stream, 'auto')
    assert actual == {'value': '2'}


@pytest.mark.parametrize('raw', ['1.5', 'true', 'NaN', '1e999'])
def test_integer_conversion_rejects_non_integral_values(raw):
    from lmdeploy.serve.parsers.tool_parser.xml_tool_parser import XmlToolParser

    value = XmlToolParser._coerce_value(raw, 'integer')
    assert value == raw
    json.dumps(value, allow_nan=False)
