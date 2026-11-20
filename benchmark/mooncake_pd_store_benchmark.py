# Copyright (c) OpenMMLab. All rights reserved.
"""Measure a fixed-route PD workload with fresh Store keys on every request.

Use the same dataset, trial and concurrency in both deployments. Each logical client keeps its session header; the
unique final message prevents a previous run's completed output from turning Store writes into existence checks.
"""
import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx


async def run(args):
    dataset = json.loads(args.dataset.read_text())
    conversations = dataset['conversations']
    bases = [conversation[0]['messages'] for conversation in conversations]
    sessions = [
        'bench-' + hashlib.sha256(json.dumps(messages[0], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        for messages in bases
    ]
    if args.session_keys:
        sessions = json.loads(args.session_keys.read_text())
        assert len(sessions) == len(bases)
    rows = []
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=1800, trust_env=False, limits=limits) as client:
        async def request(index, lane):
            base = bases[lane % len(bases)]
            messages = base + [dict(role='user', content=(
                f'Benchmark trial {args.trial}, request {index}. '
                'Review the supplied repository context. Explain the relevant implementation, '
                'possible failure cases, and a concrete test plan in detail.'))]
            body = dict(model=args.model, messages=messages, max_tokens=args.output_tokens,
                        temperature=0, ignore_eos=True, stream=True, stream_options=dict(include_usage=True))
            started = time.time()
            first = None
            usage = None
            output = hashlib.sha256()
            chunks = []
            response_id = None
            async with client.stream('POST', args.url + '/chat/completions', json=body,
                                     headers={'X-Session-ID': sessions[lane % len(sessions)]}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith('data: ') or line == 'data: [DONE]':
                        continue
                    event = json.loads(line[6:])
                    response_id = event['id']
                    if event.get('usage'):
                        usage = event['usage']
                    for choice in event['choices']:
                        delta = choice['delta']
                        text = (delta.get('reasoning_content') or '') + (delta.get('content') or '')
                        if text:
                            now = time.time()
                            if first is None:
                                first = now
                            chunks.append(now)
                            output.update(text.encode())
            completed = time.time()
            assert first is not None and usage is not None, (index, response_id, usage)
            assert usage['completion_tokens'] == args.output_tokens, (index, usage)
            row = dict(index=index, lane=lane, session=sessions[lane % len(sessions)], response_id=response_id,
                       body_hash=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                       output_hash=output.hexdigest(), started=started, first=first, completed=completed,
                       ttft_ms=(first - started) * 1000,
                       tpot_ms=(completed - first) * 1000 / (usage['completion_tokens'] - 1),
                       usage=usage, chunks=chunks)
            rows.append(row)
            print(f'{args.trial} {len(rows)}/{args.requests} lane={lane} '
                  f'TTFT={row["ttft_ms"]:.1f} TPOT={row["tpot_ms"]:.2f}', flush=True)

        async def worker(lane):
            for index in range(lane, args.requests, args.concurrency):
                await request(index, lane)

        begin = time.time()
        await asyncio.gather(*(worker(lane) for lane in range(args.concurrency)))
        end = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        trial=args.trial, concurrency=args.concurrency, requests=args.requests,
        output_tokens=args.output_tokens, begin=begin, end=end,
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        rows=sorted(rows, key=lambda row: row['index'])), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Router /v1 URL')
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--session-keys', type=Path, help='One fixed router session key per conversation')
    parser.add_argument('--trial', required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--requests', type=int, required=True)
    parser.add_argument('--output-tokens', type=int, default=2048)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
