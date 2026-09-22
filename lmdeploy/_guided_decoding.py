# Copyright (c) OpenMMLab. All rights reserved.
"""Backend-neutral guided-decoding grammar compilation."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any


def _json_schema_from_response_format(response_format: dict[str, Any]) -> str:
    schema: Any = response_format['json_schema']
    if isinstance(schema, dict):
        for key in ('json_schema', 'schema'):
            if key in schema:
                schema = schema[key]
                break
    if isinstance(schema, (dict, bool)):
        return json.dumps(schema, ensure_ascii=False)
    if isinstance(schema, str):
        return schema
    raise ValueError(
        f'Cannot parse schema {schema}. The schema must be either a dictionary '
        'or a string that contains the JSON Schema specification')


def _grammar_source(response_format: dict[str, Any]) -> tuple[str, str]:
    """Return the XGrammar entry point and its serialized input."""
    schema_type = response_format.get('type', 'text')
    if schema_type == 'json_schema':
        return schema_type, _json_schema_from_response_format(response_format)
    if schema_type == 'regex_schema':
        return schema_type, str(response_format.get('regex_schema', ''))
    if schema_type == 'json_object':
        schema = json.dumps({'type': 'object', 'additionalProperties': True})
        return 'json_schema', schema
    if schema_type == 'structural_tag':
        return schema_type, json.dumps(response_format, ensure_ascii=False)
    if schema_type == 'text':
        return schema_type, ''
    raise ValueError(f'unsupported format type: {schema_type}')


@lru_cache(maxsize=128)
def _check_response_format(serialized_format: str) -> None:
    """Check a response format before it reaches a shared engine loop."""
    import xgrammar as xgr

    response_format = json.loads(serialized_format)
    schema_type, source = _grammar_source(response_format)
    if schema_type == 'text':
        return
    if schema_type == 'json_schema':
        xgr.Grammar.from_json_schema(source)
    elif schema_type == 'regex_schema':
        xgr.Grammar.from_regex(source)
    elif schema_type == 'structural_tag':
        xgr.Grammar.from_structural_tag(source)


def ensure_response_format_compilable(response_format: dict[str, Any]) -> None:
    """Reject response formats that XGrammar cannot compile."""
    try:
        serialized_format = json.dumps(response_format, ensure_ascii=False, separators=(',', ':'))
        _check_response_format(serialized_format)
    except (KeyError, RuntimeError, TypeError) as err:
        raise ValueError(f'Unsupported response format: {err}') from err


def compile_response_format(compiler, response_format: dict[str, Any]):
    """Compile one public or internal response format with XGrammar.

    Structural-tag formats use XGrammar's native top-level shape.
    """
    schema_type, source = _grammar_source(response_format)
    if schema_type == 'json_schema':
        return compiler.compile_json_schema(source)
    if schema_type == 'regex_schema':
        return compiler.compile_regex(source)
    if schema_type == 'structural_tag':
        return compiler.compile_structural_tag(source)
    raise ValueError(f'unsupported format type: {schema_type}')
