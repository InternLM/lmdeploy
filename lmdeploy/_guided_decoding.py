# Copyright (c) OpenMMLab. All rights reserved.
"""Backend-neutral guided-decoding grammar compilation."""
from __future__ import annotations

import json
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


def compile_response_format(compiler, response_format: dict[str, Any]):
    """Compile one public or internal response format with XGrammar.

    Structural-tag formats use XGrammar's native top-level shape.
    """
    schema_type = response_format.get('type', 'text')
    if schema_type == 'json_schema':
        return compiler.compile_json_schema(
            _json_schema_from_response_format(response_format))
    if schema_type == 'regex_schema':
        return compiler.compile_regex(str(response_format.get('regex_schema', '')))
    if schema_type == 'json_object':
        schema = json.dumps({'type': 'object', 'additionalProperties': True})
        return compiler.compile_json_schema(schema)
    if schema_type == 'structural_tag':
        return compiler.compile_structural_tag(
            json.dumps(response_format, ensure_ascii=False))
    raise ValueError(f'unsupported format type: {schema_type}')
