#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Benchmark BaseResponseParser streaming and complete parsing.

Usage:
    python benchmark/benchmark_parser.py --reasoning-parser default --tool-call-parser qwen3
    python benchmark/benchmark_parser.py --reasoning-size 4096 --chunk-min 1 --chunk-max 1
"""
from __future__ import annotations

import argparse
import json
import random
import string
import time
from dataclasses import dataclass
from typing import Any

import lmdeploy.serve.parsers.reasoning_parser  # noqa: F401
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, Function, Tool
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import (
    DeepSeekV32ToolParser,
    Glm47ToolParser,
    JsonToolParser,
    KimiK2ToolParser,
    Qwen3CoderToolParser,
    ToolParser,
    ToolParserManager,
)


@dataclass(frozen=True)
class SynthConfig:
    reasoning_parser: str | None
    tool_call_parser: str | None
    reasoning_size: int
    content_size: int
    tool_call_count: int
    tool_param_count: int
    tool_payload_size: int
    enable_thinking: bool = True


def parse_size(value: str) -> tuple[int, int]:
    if ':' in value:
        lo, hi = value.split(':', 1)
        return int(lo), int(hi)
    n = int(value)
    return n, n


def random_text(rng: random.Random, size: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return ''.join(rng.choice(alphabet) for _ in range(size))


def pick_size(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi) if lo != hi else lo


def _random_arguments(param_count: int, payload_size: int, rng: random.Random) -> dict[str, str]:
    return {f'param_{j}': random_text(rng, payload_size) for j in range(param_count)}


def _json_tool_inner(function_name: str, arguments: dict[str, str], argument_field: str) -> tuple[str, list[str]]:
    return json.dumps({'name': function_name, argument_field: arguments}, ensure_ascii=False), []


def _glm47_tool_inner(function_name: str, arguments: dict[str, str]) -> tuple[str, list[str]]:
    parts = [function_name]
    # Supported GLM tokenizers emit this value terminator atomically.
    markers = ['</arg_value>']
    for name, value in arguments.items():
        arg_key = f'<arg_key>{name}</arg_key>'
        arg_value_open = '<arg_value>'
        arg_value_close = '</arg_value>'
        parts.extend((arg_key, arg_value_open, value, arg_value_close))
    return ''.join(parts), markers


def _qwen3coder_tool_inner(function_name: str, arguments: dict[str, str]) -> tuple[str, list[str]]:
    function_open = f'<function={function_name}>'
    function_close = '</function>'
    parts = [function_open]
    # Preserve the token boundaries used by _stable_arg_value_end.
    markers = ['</', 'parameter', '>']
    for name, value in arguments.items():
        parameter_open = f'<parameter={name}>'
        parameter_close = '</parameter>'
        parts.extend((parameter_open, value, parameter_close))
    parts.append(function_close)
    return ''.join(parts), markers


def _kimi_tool_inner(
    parser_cls: type[KimiK2ToolParser],
    function_name: str,
    call_index: int,
    arguments: dict[str, str],
) -> tuple[str, list[str]]:
    raw_id = f'functions.{function_name}:{call_index}'
    inner = (
        f'{parser_cls.call_begin}{raw_id}'
        f'{parser_cls.argument_begin}{json.dumps(arguments, ensure_ascii=False)}'
        f'{parser_cls.call_end}'
    )
    return inner, [parser_cls.call_begin, parser_cls.argument_begin, parser_cls.call_end]


def _deepseek_tool_inner(
    parser_cls: type[DeepSeekV32ToolParser],
    function_name: str,
    arguments: dict[str, str],
) -> tuple[str, list[str]]:
    token = parser_cls.dsml_token
    invoke_start = f'<{token}invoke'
    parameter_start = f'<{token}parameter'
    invoke_open = f'{invoke_start} name="{function_name}">'
    invoke_close = f'</{token}invoke>'
    parts = [invoke_open, '\n']
    markers = [invoke_start, parameter_start, invoke_close]
    for name, value in arguments.items():
        parameter_open = f'{parameter_start} name="{name}" string="true">'
        parameter_close = f'</{token}parameter>'
        parts.extend((parameter_open, value, parameter_close, '\n'))
        markers.append(parameter_close)
    parts.append(invoke_close)
    return ''.join(parts), markers


def _synthesize_tool_inner(
    parser_cls: type[ToolParser],
    function_name: str,
    call_index: int,
    arguments: dict[str, str],
) -> tuple[str, list[str]]:
    """Build one valid payload and markers random character chunks must not
    split."""
    if issubclass(parser_cls, JsonToolParser):
        return _json_tool_inner(function_name, arguments, parser_cls.argument_field)
    if issubclass(parser_cls, Glm47ToolParser):
        return _glm47_tool_inner(function_name, arguments)
    if issubclass(parser_cls, Qwen3CoderToolParser):
        return _qwen3coder_tool_inner(function_name, arguments)
    if issubclass(parser_cls, KimiK2ToolParser):
        return _kimi_tool_inner(parser_cls, function_name, call_index, arguments)
    if issubclass(parser_cls, DeepSeekV32ToolParser):
        return _deepseek_tool_inner(parser_cls, function_name, arguments)
    raise ValueError(f'No benchmark payload generator for tool parser {parser_cls.__name__!r}.')


def synthesize_tool_blocks(
    cfg: SynthConfig,
    rng: random.Random,
    parser_cls: type[ToolParser] | None,
) -> tuple[str, list[str], list[tuple[str, dict[str, str]]]]:
    if parser_cls is None or cfg.tool_call_count <= 0:
        return '', [], []

    tool_open = parser_cls.get_tool_open_tag() or ''
    tool_close = parser_cls.get_tool_close_tag()
    if tool_close is None and cfg.tool_call_count != 1:
        raise ValueError(f'{parser_cls.__name__} supports only one benchmark tool call block (no close tag).')

    blocks: list[str] = []
    tags: list[str] = []
    expected_calls: list[tuple[str, dict[str, str]]] = []
    if tool_open:
        tags.append(tool_open)
    if tool_close:
        tags.append(tool_close)

    for i in range(cfg.tool_call_count):
        function_name = f'bench_fn_{i}'
        arguments = _random_arguments(cfg.tool_param_count, cfg.tool_payload_size, rng)
        inner, inner_tags = _synthesize_tool_inner(parser_cls, function_name, i, arguments)
        tags.extend(inner_tags)
        expected_calls.append((function_name, arguments))

        if tool_close:
            block = f'{tool_open}{inner}{tool_close}'
        else:
            block = f'{tool_open}{inner}'
        blocks.append(block)

    return ''.join(blocks), tags, expected_calls


def synthesize_response(
    cfg: SynthConfig,
    rng: random.Random,
) -> tuple[str, list[str], list[tuple[str, dict[str, str]]]]:
    reasoning_open = reasoning_close = None
    starts_in_reasoning = False
    if cfg.reasoning_parser:
        rcls = ReasoningParserManager.get(cfg.reasoning_parser)
        reasoning_open = rcls.get_reasoning_open_tag()
        reasoning_close = rcls.get_reasoning_close_tag()
        rparser = rcls(enable_thinking=cfg.enable_thinking if cfg.enable_thinking else None)
        starts_in_reasoning = bool(rparser.starts_in_reasoning_mode())

    tool_parser_cls = None
    if cfg.tool_call_parser:
        tool_parser_cls = ToolParserManager.get(cfg.tool_call_parser)

    reasoning_body = random_text(rng, cfg.reasoning_size) if cfg.reasoning_size > 0 else ''
    content_body = random_text(rng, cfg.content_size) if cfg.content_size > 0 else ''

    reasoning_seg = ''
    if cfg.reasoning_parser and reasoning_body:
        ro = reasoning_open or ''
        rc = reasoning_close or ''
        reasoning_seg = f'{ro}{reasoning_body}{rc}'

    tool_seg, tool_tags, expected_calls = synthesize_tool_blocks(cfg, rng, tool_parser_cls)

    protected_tags: list[str] = []
    if reasoning_open:
        protected_tags.append(reasoning_open)
    if reasoning_close:
        protected_tags.append(reasoning_close)
    protected_tags.extend(tool_tags)

    if starts_in_reasoning:
        full = reasoning_seg + content_body + tool_seg
    else:
        full = content_body + reasoning_seg + tool_seg

    return full, protected_tags, expected_calls


def segment_text(text: str, tags: list[str]) -> list[tuple[str, bool]]:
    if not text:
        return []
    unique_tags = [t for t in dict.fromkeys(tags) if t]
    if not unique_tags:
        return [(text, False)]

    segments: list[tuple[str, bool]] = []
    pos = 0
    n = len(text)
    while pos < n:
        earliest_idx = -1
        earliest_tag = ''
        for tag in unique_tags:
            idx = text.find(tag, pos)
            if idx >= 0 and (earliest_idx < 0 or idx < earliest_idx):
                earliest_idx = idx
                earliest_tag = tag
        if earliest_idx < 0:
            segments.append((text[pos:], False))
            break
        if earliest_idx > pos:
            segments.append((text[pos:earliest_idx], False))
        segments.append((earliest_tag, True))
        pos = earliest_idx + len(earliest_tag)
    return segments


def chunk_segments(
    segments: list[tuple[str, bool]],
    chunk_min: int,
    chunk_max: int,
    rng: random.Random,
) -> list[str]:
    chunks: list[str] = []
    buf = ''
    target = rng.randint(chunk_min, chunk_max)

    def flush() -> None:
        nonlocal buf, target
        if buf:
            chunks.append(buf)
            buf = ''
            target = rng.randint(chunk_min, chunk_max)

    for segment, is_tag in segments:
        if is_tag:
            if buf and len(buf) + len(segment) > target:
                flush()
            buf += segment
            if len(buf) >= target or len(segment) > chunk_max:
                flush()
            continue

        offset = 0
        while offset < len(segment):
            room = target - len(buf)
            if room <= 0:
                flush()
                room = target
            take = min(len(segment) - offset, room)
            if take <= 0:
                flush()
                continue
            buf += segment[offset:offset + take]
            offset += take
            if len(buf) >= target:
                flush()

    flush()
    return chunks


def build_tools(cfg: SynthConfig) -> list[Tool] | None:
    if not cfg.tool_call_parser or cfg.tool_call_count <= 0:
        return None
    tools: list[Tool] = []
    for i in range(cfg.tool_call_count):
        params: dict[str, Any] = {
            'type': 'object',
            'properties': {
                f'param_{j}': {'type': 'string'}
                for j in range(cfg.tool_param_count)
            },
        }
        tools.append(
            Tool(function=Function(name=f'bench_fn_{i}', description='benchmark', parameters=params)))
    return tools


def build_parser(cfg: SynthConfig):
    cls = ResponseParserManager.get('default')
    cls.set_parsers(reasoning_parser_name=cfg.reasoning_parser, tool_parser_name=cfg.tool_call_parser)
    tool_choice = 'none' if not cfg.tool_call_parser else 'auto'
    request = ChatCompletionRequest(
        model='bench',
        messages=[],
        stream=True,
        tool_choice=tool_choice,
        tools=build_tools(cfg),
        chat_template_kwargs={'enable_thinking': cfg.enable_thinking} if cfg.reasoning_parser else None,
    )
    return cls(request=request)


def _normalize_output(content, tool_calls, reasoning_content):
    normalized_calls = []
    for call in tool_calls or []:
        try:
            arguments = json.loads(call.function.arguments)
        except json.JSONDecodeError as err:
            raise RuntimeError(
                f'Parser produced invalid arguments for {call.function.name!r}: {err.msg}.'
            ) from err
        normalized_calls.append((call.function.name, arguments))
    return content, normalized_calls, reasoning_content


def _parse_stream_sample(cfg: SynthConfig, chunks: list[str]):
    parser = build_parser(cfg)
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_deltas = []
    if not chunks:
        parsed_chunks = parser.stream_chunk('', [], final=True)
        chunks_with_output = (parsed_chunks, )
    else:
        chunks_with_output = (
            parser.stream_chunk(chunk, [], final=index == len(chunks) - 1)
            for index, chunk in enumerate(chunks)
        )

    for parsed_chunks in chunks_with_output:
        for message, _ in parsed_chunks:
            if message.content:
                content_parts.append(message.content)
            if message.reasoning_content:
                reasoning_parts.append(message.reasoning_content)
            if message.tool_calls:
                tool_deltas.extend(message.tool_calls)

    tool_calls = parser.tool_parser.build_tool_calls(tool_deltas) if parser.tool_parser is not None else None
    content = ''.join(content_parts) or None
    reasoning_content = ''.join(reasoning_parts) or None
    return _normalize_output(content, tool_calls, reasoning_content)


def validate_sample(
    cfg: SynthConfig,
    full_text: str,
    chunks: list[str],
    expected_calls: list[tuple[str, dict[str, str]]],
) -> None:
    """Reject malformed benchmark fixtures before recording timings."""
    streamed = _parse_stream_sample(cfg, chunks)
    complete = _normalize_output(*build_parser(cfg).parse_complete(full_text))
    if streamed != complete:
        raise RuntimeError('Streaming and complete parsing produced different normalized outputs.')
    if streamed[1] != expected_calls:
        expected_names = [name for name, _ in expected_calls]
        actual_names = [name for name, _ in streamed[1]]
        raise RuntimeError(
            'Benchmark payload did not produce the expected tool calls: '
            f'expected names {expected_names}, got {actual_names}; arguments may also differ.'
        )


def run_stream_benchmark(cfg: SynthConfig, chunks: list[str], iterations: int) -> float:
    streamed_chunks = chunks[:-1]
    final_chunk = chunks[-1] if chunks else ''
    start = time.perf_counter()
    for _ in range(iterations):
        parser = build_parser(cfg)
        for chunk in streamed_chunks:
            parser.stream_chunk(chunk, [], final=False)
        parser.stream_chunk(final_chunk, [], final=True)
    return time.perf_counter() - start


def run_complete_benchmark(cfg: SynthConfig, full_text: str, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        parser = build_parser(cfg)
        parser.parse_complete(full_text)
    return time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser(description='Benchmark BaseResponseParser')
    ap.add_argument('--reasoning-parser', default='default')
    ap.add_argument('--tool-call-parser', default='qwen3')
    ap.add_argument('--reasoning-size', default='512')
    ap.add_argument('--content-size', default='256')
    ap.add_argument('--tool-call-count', type=int, default=1)
    ap.add_argument('--tool-param-count', type=int, default=1)
    ap.add_argument('--tool-payload-size', default='2048')
    ap.add_argument('--chunk-min', type=int, default=1)
    ap.add_argument('--chunk-max', type=int, default=32)
    ap.add_argument('--iterations', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-reasoning', action='store_true', help='Disable reasoning segment')
    ap.add_argument('--no-tool', action='store_true', help='Disable tool segment')
    args = ap.parse_args()

    if args.chunk_min < 1:
        ap.error('--chunk-min must be >= 1')
    if args.chunk_max < args.chunk_min:
        ap.error('--chunk-max must be >= --chunk-min')

    rng = random.Random(args.seed)
    rs_lo, rs_hi = parse_size(args.reasoning_size)
    cs_lo, cs_hi = parse_size(args.content_size)
    ps_lo, ps_hi = parse_size(args.tool_payload_size)

    cfg = SynthConfig(
        reasoning_parser=None if args.no_reasoning else args.reasoning_parser,
        tool_call_parser=None if args.no_tool else args.tool_call_parser,
        reasoning_size=pick_size(rng, rs_lo, rs_hi),
        content_size=pick_size(rng, cs_lo, cs_hi),
        tool_call_count=args.tool_call_count,
        tool_param_count=args.tool_param_count,
        tool_payload_size=pick_size(rng, ps_lo, ps_hi),
    )

    full_text, protected_tags, expected_calls = synthesize_response(cfg, rng)
    segments = segment_text(full_text, protected_tags)
    chunks = chunk_segments(segments, args.chunk_min, args.chunk_max, rng)
    if ''.join(chunks) != full_text:
        raise RuntimeError('Chunk reassembly mismatch.')
    validate_sample(cfg, full_text, chunks, expected_calls)
    del expected_calls

    stream_s = run_stream_benchmark(cfg, chunks, args.iterations)
    complete_s = run_complete_benchmark(cfg, full_text, max(1, args.iterations // 10))
    text_len = len(full_text)

    print('BaseResponseParser Benchmark')
    print('=' * 72)
    print(f'Parser: reasoning={cfg.reasoning_parser} tool={cfg.tool_call_parser}')
    print(
        f'Text: reasoning={cfg.reasoning_size} content={cfg.content_size} '
        f'tool_calls={cfg.tool_call_count} params={cfg.tool_param_count} '
        f'payload_size={cfg.tool_payload_size}  chars={text_len}  chunks={len(chunks)}')
    print(f'Iterations: {args.iterations}')
    print()
    print(f'{"Scenario":<24} {"Total(s)":>10} {"Chars/s":>12} {"Per-iter":>12}')
    print('-' * 72)
    stream_cps = text_len * args.iterations / stream_s if stream_s > 0 else 0.0
    complete_iters = max(1, args.iterations // 10)
    complete_cps = text_len * complete_iters / complete_s if complete_s > 0 else 0.0
    print(f'{"stream_chunk":<24} {stream_s:10.3f} {stream_cps:12.0f} {len(chunks):12d}')
    print(f'{"parse_complete":<24} {complete_s:10.3f} {complete_cps:12.0f} {1:12d}')


if __name__ == '__main__':
    main()
