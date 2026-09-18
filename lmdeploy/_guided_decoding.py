# Copyright (c) OpenMMLab. All rights reserved.
"""Backend-neutral guided-decoding grammar compilation."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

# Bounds for client-supplied grammar sources. The compile cost of a nested
# JSON schema grows exponentially with its depth and XGrammar exposes no
# depth or time limit on that path (as of 0.2.x), so unbounded schemas must be
# rejected before compilation.
MAX_GRAMMAR_SOURCE_BYTES = 16 * 1024
# Raw JSON nesting depth of a grammar source (a nested object spans two JSON
# levels per schema level through its "properties" wrapper). XGrammar's
# compile cost grows exponentially with this depth, so keep the worst
# accepted compile well under a second. Realistic structured-output schemas
# stay far below the limit.
MAX_JSON_NESTING_DEPTH = 128
# Hard budget for one grammar validation. The work itself runs in a worker
# thread, so this only bounds how long a request may wait on it.
GRAMMAR_COMPILE_TIMEOUT = 5.0


def _max_nesting_depth(value: Any) -> int:
    """Return the maximum nesting depth of a JSON-like structure.

    Iterative on purpose: a hostile deeply nested schema must not be able to
    raise RecursionError here (or inside ``json.dumps`` further down the path).
    """
    max_depth = 0
    stack = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if not isinstance(node, (dict, list)):
            # Scalars (strings, numbers, booleans, None) do not add depth.
            continue
        if depth > max_depth:
            max_depth = depth
        if isinstance(node, dict):
            stack.extend((child, depth + 1) for child in node.values())
        else:
            stack.extend((child, depth + 1) for child in node)
    return max_depth


def _check_source_size(source: str) -> None:
    # Count UTF-8 bytes, not code points: a serialized source keeps non-ASCII
    # characters (json.dumps with ensure_ascii=False), so code points can
    # understate the real size by up to 4x.
    size = len(source.encode('utf-8'))
    if size > MAX_GRAMMAR_SOURCE_BYTES:
        raise ValueError(f'grammar source exceeds the maximum size of {MAX_GRAMMAR_SOURCE_BYTES} bytes.')


def _check_schema_depth(schema: Any) -> None:
    depth = _max_nesting_depth(schema)
    if depth > MAX_JSON_NESTING_DEPTH:
        raise ValueError(f'json_schema exceeds the maximum nesting depth of {MAX_JSON_NESTING_DEPTH}.')


def _json_schema_from_response_format(response_format: dict[str, Any]) -> str:
    schema: Any = response_format['json_schema']
    if isinstance(schema, dict):
        for key in ('json_schema', 'schema'):
            if key in schema:
                schema = schema[key]
                break
    if isinstance(schema, (dict, bool)):
        _check_schema_depth(schema)
        return json.dumps(schema, ensure_ascii=False)
    if isinstance(schema, str):
        _check_source_size(schema)
        try:
            parsed = json.loads(schema)
        except RecursionError as err:
            raise ValueError(f'json_schema exceeds the maximum nesting depth of {MAX_JSON_NESTING_DEPTH}.') from err
        _check_schema_depth(parsed)
        return schema
    raise ValueError(
        f'Cannot parse schema {schema}. The schema must be either a dictionary '
        'or a string that contains the JSON Schema specification'
    )


def _grammar_source(response_format: dict[str, Any]) -> tuple[str, str]:
    """Return the XGrammar entry point and its serialized input."""
    schema_type = response_format.get('type', 'text')
    if schema_type == 'json_schema':
        source = _json_schema_from_response_format(response_format)
    elif schema_type == 'regex_schema':
        source = str(response_format.get('regex_schema', ''))
    elif schema_type == 'json_object':
        source = json.dumps({'type': 'object', 'additionalProperties': True})
        schema_type = 'json_schema'
    elif schema_type == 'structural_tag':
        # Built by the tool-call parsers and the gpt-oss response parser (the
        # latter wraps client-supplied schemas), and also accepted directly by
        # GenerationConfig. Only the size bound applies here: XGrammar guards
        # the tag's own format tree with its recursion guard, pydantic
        # serialization caps how deep the gpt-oss parser can wrap a schema,
        # and an embedded schema additionally passes the EBNF parser's fixed
        # nest limit.
        source = json.dumps(response_format, ensure_ascii=False)
    elif schema_type == 'text':
        return schema_type, ''
    else:
        raise ValueError(f'unsupported format type: {schema_type}')
    _check_source_size(source)
    return schema_type, source


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


def _ensure_response_format_compilable(response_format: dict[str, Any]) -> None:
    """Synchronous validation body; must not run on an event loop."""
    try:
        serialized_format = json.dumps(response_format, ensure_ascii=False, separators=(',', ':'))
        _check_response_format(serialized_format)
    except (KeyError, RecursionError, RuntimeError, TypeError) as err:
        raise ValueError(f'Unsupported response format: {err}') from err


VALIDATION_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix='grammar-validate')


async def ensure_response_format_compilable(response_format: dict[str, Any]) -> None:
    """Reject response formats that XGrammar cannot compile.

    The compile runs in a dedicated worker pool under a hard timeout:
    XGrammar has no compile-time bound of its own, and a synchronous compile
    on the event loop would stall every request, stream, and health check in
    the process. A compile that overruns the timeout cannot be cancelled, but
    it only ever occupies this small pool, never the interpreter's default
    executor shared by other offloaded work.
    """
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(VALIDATION_EXECUTOR, _ensure_response_format_compilable, response_format),
            timeout=GRAMMAR_COMPILE_TIMEOUT,
        )
    except asyncio.TimeoutError as err:
        raise ValueError(f'Response format validation timed out after {GRAMMAR_COMPILE_TIMEOUT}s.') from err


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
