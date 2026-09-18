# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for guided-decoding grammar validation bounds.

The response_format validation must reject unbounded grammar sources before
they reach XGrammar: the compile cost of a nested JSON schema grows
exponentially with depth and XGrammar has no depth or time limit on that
path. These tests run offline; no model, tokenizer, or GPU is required.
"""

import asyncio
import json
import threading

import pytest

from lmdeploy import _guided_decoding as gd
from lmdeploy._guided_decoding import (
    MAX_GRAMMAR_SOURCE_BYTES,
    MAX_JSON_NESTING_DEPTH,
    _grammar_source,
    compile_response_format,
    ensure_response_format_compilable,
)


def nested_object_schema(raw_depth: int, key: str = 'a') -> dict:
    """Build a response_format whose raw JSON nesting is exactly ``raw_depth``.

    The common ``properties`` shape spans two JSON levels per schema level;
    remaining levels are padded with a plain key.
    """
    schema: dict = {'type': 'object'}
    node = schema
    for _ in range((raw_depth - 1) // 2):
        child: dict = {'type': 'object'}
        node['properties'] = {key: child}
        node = child
    for _ in range((raw_depth - 1) % 2):
        child: dict = {'type': 'object'}
        node['pad'] = child
        node = child
    return {'type': 'json_schema', 'json_schema': {'name': 't', 'schema': schema}}


def json_schema_format(schema) -> dict:
    return {'type': 'json_schema', 'json_schema': {'name': 't', 'schema': schema}}


class TestGrammarSourceBounds:
    def test_schema_at_depth_cap_passes(self):
        response_format = nested_object_schema(MAX_JSON_NESTING_DEPTH)
        schema_type, source = _grammar_source(response_format)
        assert schema_type == 'json_schema'
        assert len(source) > 0

    def test_schema_beyond_depth_cap_rejected(self):
        with pytest.raises(ValueError, match='nesting depth'):
            _grammar_source(nested_object_schema(MAX_JSON_NESTING_DEPTH + 1))

    def test_deeply_nested_schema_rejected_fast(self):
        # A hostile schema must be rejected by the bounds check, never reach
        # the exponential XGrammar compile.
        with pytest.raises(ValueError, match='nesting depth'):
            _grammar_source(nested_object_schema(10000))

    def test_deep_schema_as_string_rejected(self):
        schema = json.dumps(nested_object_schema(1000)['json_schema']['schema'])
        response_format = {
            'type': 'json_schema',
            'json_schema': {
                'name': 't',
                'schema': schema,
            },
        }
        with pytest.raises(ValueError):
            _grammar_source(response_format)

    def test_oversized_regex_rejected(self):
        response_format = {'type': 'regex_schema', 'regex_schema': 'a' * (MAX_GRAMMAR_SOURCE_BYTES + 1)}
        with pytest.raises(ValueError, match='maximum size'):
            _grammar_source(response_format)

    def test_oversized_flat_schema_rejected(self):
        schema = {'type': 'object', 'properties': {f'k{i}': {'type': 'string'} for i in range(1200)}}
        with pytest.raises(ValueError, match='maximum size'):
            _grammar_source(json_schema_format(schema))

    def test_multibyte_source_counted_as_utf8_bytes(self):
        # The size limit is UTF-8 bytes: 6000 CJK characters are only 6000
        # code points but 18000 bytes, so the source must be rejected even
        # though a code-point count would pass it.
        schema = {'type': 'object', 'properties': {'名字': {'description': '描' * 6000}}}
        with pytest.raises(ValueError, match='maximum size'):
            _grammar_source(json_schema_format(schema))

    def test_regular_formats_pass(self):
        assert _grammar_source({'type': 'text'}) == ('text', '')
        assert _grammar_source({'type': 'json_object'}) == (
            'json_schema',
            json.dumps({'type': 'object', 'additionalProperties': True}),
        )
        schema_type, source = _grammar_source({'type': 'regex_schema', 'regex_schema': r'\d{3}-\d{4}'})
        assert (schema_type, source) == ('regex_schema', r'\d{3}-\d{4}')
        schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'items': {
                    'type': 'array',
                    'items': {'type': 'number'},
                },
            },
        }
        _grammar_source(json_schema_format(schema))

    def test_engine_compile_path_enforces_bounds(self):
        # compile_response_format runs in the engine process; the bounds
        # check in _grammar_source must reject before the compiler is used.
        with pytest.raises(ValueError, match='nesting depth'):
            compile_response_format(None, nested_object_schema(10000))


class TestAsyncValidation:
    def test_valid_response_format_passes(self):
        schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
            },
        }
        asyncio.run(ensure_response_format_compilable(json_schema_format(schema)))

    def test_invalid_response_format_rejected(self):
        with pytest.raises(ValueError):
            asyncio.run(ensure_response_format_compilable(nested_object_schema(10000)))

    def test_validation_runs_in_dedicated_pool(self, monkeypatch):
        seen_threads = []
        original = gd._check_response_format.__wrapped__

        def spy(serialized_format):
            seen_threads.append(threading.current_thread())
            return original(serialized_format)

        monkeypatch.setattr(gd, '_check_response_format', spy)
        schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
            },
        }
        asyncio.run(ensure_response_format_compilable(json_schema_format(schema)))
        assert seen_threads, 'validation did not reach the grammar check'
        assert all(t.name.startswith('grammar-validate') for t in seen_threads), (
            'validation must run in the dedicated grammar pool, not the default executor'
        )

    def test_validation_timeout_becomes_value_error(self, monkeypatch):
        monkeypatch.setattr(gd, 'GRAMMAR_COMPILE_TIMEOUT', 0.2)

        def slow_check(_serialized_format):
            threading.Event().wait(5)

        monkeypatch.setattr(gd, '_check_response_format', slow_check)
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
        with pytest.raises(ValueError, match='timed out'):
            asyncio.run(ensure_response_format_compilable(json_schema_format(schema)))

    def test_event_loop_stays_responsive_during_validation(self):
        """A real uncached validation must not stall the loop (GIL released by
        XGrammar, compile runs in a worker thread)."""
        schema = nested_object_schema(120, key='responsive')

        async def main():
            delays = []

            async def heartbeat():
                while True:
                    start = asyncio.get_running_loop().time()
                    await asyncio.sleep(0.02)
                    delays.append(asyncio.get_running_loop().time() - start - 0.02)

            hb = asyncio.create_task(heartbeat())
            await asyncio.sleep(0.2)
            delays.clear()
            await ensure_response_format_compilable(schema)
            await asyncio.sleep(0.1)
            hb.cancel()
            return delays

        delays = asyncio.run(main())
        assert len(delays) > 3, 'heartbeat stopped ticking during validation'
        assert max(delays) < 1.0, f'event loop stalled for {max(delays):.3f}s during validation'
