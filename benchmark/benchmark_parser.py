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

# Ensure tool/reasoning parser modules are registered.
import lmdeploy.serve.parsers.tool_parser  # noqa: F401
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, Function, Tool
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager


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


def _json_tool_inner(fn_index: int, param_count: int, payload_size: int, rng: random.Random) -> str:
    args = {f'param_{j}': random_text(rng, payload_size) for j in range(param_count)}
    return json.dumps({'name': f'bench_fn_{fn_index}', 'arguments': args}, ensure_ascii=False)


def _glm47_tool_inner(fn_index: int, param_count: int, payload_size: int, rng: random.Random) -> str:
    parts = [f'bench_fn_{fn_index}']
    for j in range(param_count):
        parts.append(f'<arg_key>param_{j}</arg_key>')
        parts.append(f'<arg_value>{random_text(rng, payload_size)}</arg_value>')
    return ''.join(parts)


def _qwen3coder_tool_inner(fn_index: int, param_count: int, payload_size: int, rng: random.Random) -> str:
    parts = [f'<function=bench_fn_{fn_index}>']
    for j in range(param_count):
        parts.append(f'<parameter=param_{j}>')
        parts.append(random_text(rng, payload_size))
        parts.append('</parameter>')
    parts.append('</function>')
    return ''.join(parts)


def synthesize_tool_blocks(
    cfg: SynthConfig,
    rng: random.Random,
    tool_open: str,
    tool_close: str | None,
    payload_format: str,
) -> tuple[str, list[str]]:
    if cfg.tool_call_parser is None or cfg.tool_call_count <= 0:
        return '', []

    if cfg.tool_call_parser == 'llama3' and cfg.tool_call_count != 1:
        raise ValueError('llama3 tool parser supports only one tool call block (no close tag)')

    blocks: list[str] = []
    tags: list[str] = []
    if tool_open:
        tags.append(tool_open)
    if tool_close:
        tags.append(tool_close)

    for i in range(cfg.tool_call_count):
        if payload_format == 'json':
            inner = _json_tool_inner(i, cfg.tool_param_count, cfg.tool_payload_size, rng)
        elif payload_format == 'xml' and cfg.tool_call_parser == 'glm47':
            inner = _glm47_tool_inner(i, cfg.tool_param_count, cfg.tool_payload_size, rng)
        elif payload_format == 'xml' and cfg.tool_call_parser in ('qwen3coder',):
            inner = _qwen3coder_tool_inner(i, cfg.tool_param_count, cfg.tool_payload_size, rng)
        else:
            inner = _json_tool_inner(i, cfg.tool_param_count, cfg.tool_payload_size, rng)

        if tool_close:
            block = f'{tool_open}{inner}{tool_close}'
        else:
            block = f'{tool_open}{inner}'
        blocks.append(block)

    return ''.join(blocks), tags


def synthesize_response(cfg: SynthConfig, rng: random.Random) -> tuple[str, list[str]]:
    reasoning_open = reasoning_close = None
    starts_in_reasoning = False
    if cfg.reasoning_parser:
        rcls = ReasoningParserManager.get(cfg.reasoning_parser)
        reasoning_open = rcls.get_reasoning_open_tag()
        reasoning_close = rcls.get_reasoning_close_tag()
        rparser = rcls(enable_thinking=cfg.enable_thinking if cfg.enable_thinking else None)
        starts_in_reasoning = bool(rparser.starts_in_reasoning_mode())

    tool_open = tool_close = None
    payload_format = 'json'
    if cfg.tool_call_parser:
        tcls = ToolParserManager.get(cfg.tool_call_parser)
        tool_open = tcls.get_tool_open_tag()
        tool_close = tcls.get_tool_close_tag()
        payload_format = tcls.get_tool_payload_format()

    reasoning_body = random_text(rng, cfg.reasoning_size) if cfg.reasoning_size > 0 else ''
    content_body = random_text(rng, cfg.content_size) if cfg.content_size > 0 else ''

    reasoning_seg = ''
    if cfg.reasoning_parser and reasoning_body:
        ro = reasoning_open or ''
        rc = reasoning_close or ''
        reasoning_seg = f'{ro}{reasoning_body}{rc}'

    tool_seg, tool_tags = synthesize_tool_blocks(cfg, rng, tool_open or '', tool_close, payload_format)

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

    return full, protected_tags


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


def run_stream_benchmark(cfg: SynthConfig, chunks: list[str], iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        parser = build_parser(cfg)
        for chunk in chunks:
            parser.stream_chunk(chunk, [])
    return time.perf_counter() - start


def run_complete_benchmark(cfg: SynthConfig, full_text: str, iterations: int) -> float:
    start = time.perf_counter()
    parser = build_parser(cfg)
    for _ in range(iterations):
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

    full_text, protected_tags = synthesize_response(cfg, rng)
    segments = segment_text(full_text, protected_tags)
    chunks = chunk_segments(segments, args.chunk_min, args.chunk_max, rng)
    assert ''.join(chunks) == full_text, 'chunk reassembly mismatch'

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
