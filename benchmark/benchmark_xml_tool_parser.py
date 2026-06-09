#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Benchmark XML tool parsers: legacy (main) vs optimized implementations.

Usage:
    python benchmark/benchmark_xml_tool_parser.py
    python benchmark/benchmark_xml_tool_parser.py --iterations 5000 --value-chars 4096
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lmdeploy.serve.openai.protocol import DeltaFunctionCall, DeltaToolCall
from lmdeploy.serve.parsers.tool_parser.glm47_tool_parser import Glm47ToolParser
from lmdeploy.serve.parsers.tool_parser.qwen3coder_tool_parser import Qwen3CoderToolParser

# ---------------------------------------------------------------------------
# Legacy implementations (snapshot of main-branch logic before optimization)
# ---------------------------------------------------------------------------


class LegacyXmlToolParser:
    def __init__(self):
        self._tool_payload = ''
        self._function_param_schemas: dict[str, dict[str, dict[str, Any]]] = {}
        self._xml_has_emitted_json_start = False
        self._xml_json_closed = False
        self._xml_emitted_param_names: set[str] = set()
        self._active_tool_call_id = 'bench-tool'
        self._active_tool_index = 0
        self._name_emitted = False

    def start_tool_call(self):
        self._tool_payload = ''
        self._xml_has_emitted_json_start = False
        self._xml_json_closed = False
        self._xml_emitted_param_names.clear()
        self._name_emitted = False

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        self._tool_payload += added_text
        func_name, raw_args_dict, is_closed = self._extract_incremental_state(self._tool_payload, final=final)
        args_dict = self._coerce_args_by_schema(func_name, raw_args_dict)

        out: list[DeltaToolCall] = []
        if func_name and not self._name_emitted:
            out.append(
                DeltaToolCall(
                    id=self._active_tool_call_id,
                    index=self._active_tool_index,
                    type='function',
                    function=DeltaFunctionCall(name=func_name),
                ))
            self._name_emitted = True

        should_close = is_closed or final
        json_fragments: list[str] = []
        if not self._xml_has_emitted_json_start and (args_dict or should_close):
            json_fragments.append('{')
            self._xml_has_emitted_json_start = True

        for key, value in args_dict.items():
            if key in self._xml_emitted_param_names:
                continue
            prefix = ', ' if self._xml_emitted_param_names else ''
            json_fragments.append(f'{prefix}\"{key}\": {json.dumps(value, ensure_ascii=False)}')
            self._xml_emitted_param_names.add(key)

        if should_close and self._xml_has_emitted_json_start and not self._xml_json_closed:
            json_fragments.append('}')
            self._xml_json_closed = True

        if json_fragments:
            out.append(
                DeltaToolCall(
                    id=None,
                    index=self._active_tool_index,
                    type=None,
                    function=DeltaFunctionCall(arguments=''.join(json_fragments)),
                ))
        return out

    def _coerce_args_by_schema(self, func_name: str | None, args_dict: dict[str, Any]) -> dict[str, Any]:
        if not func_name or not args_dict:
            return args_dict
        param_schemas = self._function_param_schemas.get(func_name, {})
        if not param_schemas:
            return args_dict
        coerced: dict[str, Any] = {}
        for key, value in args_dict.items():
            if not isinstance(value, str):
                coerced[key] = value
                continue
            schema = param_schemas.get(key)
            if not isinstance(schema, dict):
                coerced[key] = value
                continue
            schema_type = schema.get('type')
            if schema_type == 'integer':
                try:
                    parsed = json.loads(value)
                    coerced[key] = parsed if isinstance(parsed, int) else value
                except json.JSONDecodeError:
                    coerced[key] = value
            else:
                coerced[key] = value
        return coerced

    def _extract_incremental_state(self, payload: str, final: bool = False):
        raise NotImplementedError


class LegacyGlm47ToolParser(LegacyXmlToolParser):
    arg_key_start_token = '<arg_key>'
    arg_key_end_token = '</arg_key>'
    arg_value_start_token = '<arg_value>'
    arg_value_end_token = '</arg_value>'

    def _extract_incremental_state(self, payload: str, final: bool = False):
        func_name, args_dict = self._parse_payload(payload, final=final)
        return func_name, args_dict, False

    def _parse_payload(self, payload: str, *, final: bool = False):
        payload = payload.strip()
        if not payload:
            return None, {}
        args_start_idx = payload.find(self.arg_key_start_token)
        if args_start_idx >= 0:
            func_name = payload[:args_start_idx].strip()
            args_text = payload[args_start_idx:]
        else:
            if not final:
                return None, {}
            func_name = payload.strip()
            args_text = ''
        if not func_name:
            return None, {}

        args_dict: dict[str, str] = {}
        search_idx = 0
        while True:
            key_start = args_text.find(self.arg_key_start_token, search_idx)
            if key_start < 0:
                break
            key_content_start = key_start + len(self.arg_key_start_token)
            key_end = args_text.find(self.arg_key_end_token, key_content_start)
            if key_end < 0:
                break
            key = args_text[key_content_start:key_end].strip()
            value_start = args_text.find(self.arg_value_start_token, key_end + len(self.arg_key_end_token))
            if value_start < 0:
                break
            value_content_start = value_start + len(self.arg_value_start_token)
            value_end = args_text.find(self.arg_value_end_token, value_content_start)
            if value_end < 0:
                break
            if key:
                args_dict[key] = args_text[value_content_start:value_end]
            search_idx = value_end + len(self.arg_value_end_token)
        return func_name, args_dict


class LegacyQwen3CoderToolParser(LegacyXmlToolParser):
    func_prefix = '<function='
    func_end_token = '</function>'
    param_prefix = '<parameter='
    param_end_token = '</parameter>'

    def _extract_incremental_state(self, payload: str, final: bool = False):
        return self._extract_params(payload)

    def _extract_params(self, content: str):
        content = content.replace('<tool_call>', '').replace('</tool_call>', '').strip()
        func_name = None
        func_start = content.find(self.func_prefix)
        if func_start != -1:
            name_start = func_start + len(self.func_prefix)
            terminators = [idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1]
            if terminators:
                func_name = content[name_start:min(terminators)].strip()

        args_dict = {}
        search_idx = 0
        while True:
            param_start = content.find(self.param_prefix, search_idx)
            if param_start == -1:
                break
            name_start = param_start + len(self.param_prefix)
            terminators = [idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1]
            if not terminators:
                break
            name_end = min(terminators)
            param_name = content[name_start:name_end].strip()
            val_start = name_end + 1
            val_end = content.find(self.param_end_token, val_start)
            if val_end == -1:
                break
            param_val_str = content[val_start:val_end].strip()
            try:
                parsed_val = json.loads(param_val_str)
                val = parsed_val if isinstance(parsed_val, str) else param_val_str
            except json.JSONDecodeError:
                val = param_val_str
            args_dict[param_name] = val
            search_idx = val_end + len(self.param_end_token)
        is_func_closed = self.func_end_token in content
        return func_name, args_dict, is_func_closed


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    name: str
    legacy_s: float
    optimized_s: float

    @property
    def speedup(self) -> float:
        return self.legacy_s / self.optimized_s if self.optimized_s > 0 else float('inf')


def _run_stream(parser_factory: Callable[[], Any], chunks: list[str], iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        parser = parser_factory()
        parser.start_tool_call()
        for chunk in chunks:
            parser.decode_tool_incremental(chunk, final=False)
        parser.decode_tool_incremental('', final=True)
    return time.perf_counter() - start


def _glm47_chunks(value_chars: int) -> list[str]:
    value = 'x' * value_chars
    return [
        'get_weather',
        '<arg_key>location</arg_key><arg_value>',
        *list(value),
        '</arg_value>',
        '<arg_key>unit</arg_key><arg_value>celsius</arg_value>',
        '</tool_call>',
    ]


def _qwen3coder_chunks(value_chars: int) -> list[str]:
    header = ['<function=search_docs>', '<parameter=query>']
    value = list('y' * value_chars)
    tail = ['</parameter>', '</function>']
    return header + value + tail


def _qwen3coder_tokenized_chunks(value_chars: int) -> list[str]:
    """Simulate token-by-token streaming like Qwen3.5 reference tests."""
    chunks = [
        '<function=get_current_temperature>',
        '\n',
        '<parameter=location>',
        '\n',
    ]
    value = 'Beijing, China'
    if value_chars > len(value):
        value = value + ('.' * (value_chars - len(value)))
    chunks.extend(list(value))
    chunks.extend(['\n', '</parameter>', '\n', '</function>'])
    return chunks


def _attach_schema(parser: Any) -> None:
    parser._function_param_schemas = {
        'get_weather': {
            'location': {
                'type': 'string'
            },
            'unit': {
                'type': 'string'
            },
        },
    }


def _bench_pair(name: str,
                legacy_factory: Callable[[], Any],
                optimized_factory: Callable[[], Any],
                chunks: list[str],
                iterations: int) -> BenchResult:
    legacy_s = _run_stream(legacy_factory, chunks, iterations)
    optimized_s = _run_stream(optimized_factory, chunks, iterations)
    return BenchResult(name=name, legacy_s=legacy_s, optimized_s=optimized_s)


def _assert_equivalent(legacy_factory: Callable[[], Any],
                       optimized_factory: Callable[[], Any],
                       chunks: list[str]) -> None:
    legacy = legacy_factory()
    optimized = optimized_factory()
    legacy.start_tool_call()
    optimized.start_tool_call()
    legacy_out: list[DeltaToolCall] = []
    optimized_out: list[DeltaToolCall] = []
    for chunk in chunks:
        legacy_out.extend(legacy.decode_tool_incremental(chunk, final=False))
        optimized_out.extend(optimized.decode_tool_incremental(chunk, final=False))
    legacy_out.extend(legacy.decode_tool_incremental('', final=True))
    optimized_out.extend(optimized.decode_tool_incremental('', final=True))

    def _normalize(calls: list[DeltaToolCall]) -> list[tuple]:
        rows = []
        for call in calls:
            fn = call.function
            rows.append((fn.name if fn else None, fn.arguments if fn else None))
        return rows

    assert _normalize(legacy_out) == _normalize(optimized_out)


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark XML tool parser optimizations')
    parser.add_argument('--iterations', type=int, default=2000, help='Repeat count per scenario')
    parser.add_argument('--value-chars', type=int, default=2048, help='Long parameter value length')
    args = parser.parse_args()

    glm_chunks = _glm47_chunks(args.value_chars)
    qwen_chunks = _qwen3coder_chunks(args.value_chars)
    qwen_token_chunks = _qwen3coder_tokenized_chunks(min(args.value_chars, 256))

    def legacy_glm():
        p = LegacyGlm47ToolParser()
        _attach_schema(p)
        return p

    def optimized_glm():
        p = Glm47ToolParser()
        _attach_schema(p)
        return p

    def legacy_qwen():
        return LegacyQwen3CoderToolParser()

    def optimized_qwen():
        return Qwen3CoderToolParser()

    _assert_equivalent(legacy_glm, optimized_glm, glm_chunks)
    _assert_equivalent(legacy_qwen, optimized_qwen, qwen_chunks)
    _assert_equivalent(legacy_qwen, optimized_qwen, qwen_token_chunks)

    results = [
        _bench_pair(
            f'GLM47 long value ({args.value_chars} chars, {len(glm_chunks)} chunks)',
            legacy_glm,
            optimized_glm,
            glm_chunks,
            args.iterations,
        ),
        _bench_pair(
            f'Qwen3Coder long value ({args.value_chars} chars, {len(qwen_chunks)} chunks)',
            legacy_qwen,
            optimized_qwen,
            qwen_chunks,
            args.iterations,
        ),
        _bench_pair(
            f'Qwen3Coder tokenized stream ({len(qwen_token_chunks)} chunks)',
            legacy_qwen,
            optimized_qwen,
            qwen_token_chunks,
            args.iterations * 4,
        ),
    ]

    print('XML Tool Parser Benchmark')
    print('=' * 72)
    print(f'Iterations per scenario: {args.iterations}')
    print()
    print(f'{"Scenario":<50} {"Legacy":>8} {"Optimized":>10} {"Speedup":>8}')
    print('-' * 72)
    for result in results:
        print(
            f'{result.name:<50} {result.legacy_s:7.3f}s {result.optimized_s:9.3f}s {result.speedup:7.2f}x')
    print('-' * 72)
    avg_speedup = sum(r.speedup for r in results) / len(results)
    print(f'Average speedup: {avg_speedup:.2f}x')


if __name__ == '__main__':
    main()
