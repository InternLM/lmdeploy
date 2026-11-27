"""Check and measure ``top_logprobs_format='flat'`` on a live ``POST
/generate``.

Needs a GPU server; nothing here runs in CI. Start a PyTorch api_server with
raw logprobs on a small model, for example::

    lmdeploy serve api_server Qwen/Qwen3-0.6B --backend pytorch \\
        --logprobs-mode raw_logprobs --server-port 23333

then run::

    python benchmark/check_generate_flat_logprobs.py --base-url http://127.0.0.1:23333

Checks (any failure raises ``AssertionError``):

1. Invariants of every flat row, from temperature-1 sampling without
   truncation: shape ``[n, k]`` matching ``output_token_logprobs``; ids
   distinct and non-negative; logprobs <= 0, descending, ``sum(exp) <= 1``;
   the sampled token's logprob equals its entry when it is in the row, and is
   not above the row's last entry otherwise.
2. Flat equals nested, and streamed chunks concatenate to the non-streamed
   array. These compare separate greedy requests up to their first differing
   output id. Prefix caching and kernel nondeterminism can move values by a
   rounding step between requests, so values are compared within ``--tol``
   (ids may differ only at the top-k boundary) and the exact-match rate is
   printed. Within one request, flat and nested are the same numbers; the CPU
   unit tests check that.
3. Mixed batches: concurrent requests with different k each get their own k.

It then prints decode throughput and response bytes for k=0/32/128 (flat) and
k=128 (nested), and whether the returned values are bf16-representable (they
are unless the server runs with ``--hf-overrides '{"fp32_lm_head": true}'`` on
a model whose LM head honours it).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time

import aiohttp
import numpy as np


def decode_flat(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``[n, k]`` (ids, logprobs) of one response or chunk."""
    shape = meta.get('output_top_logprobs_shape')
    if shape is None:
        return np.empty((0, 0), np.int32), np.empty((0, 0), np.float32)
    ids = np.frombuffer(base64.b64decode(meta['output_top_logprobs_ids_b64']), dtype='<i4').reshape(shape)
    vals = np.frombuffer(base64.b64decode(meta['output_top_logprobs_b64']), dtype='<f4').reshape(shape)
    return ids, vals


async def generate(session, url, payload):
    """POST /generate; return (output_ids, token_logprobs, metas,
    body_bytes)."""
    async with session.post(f'{url}/generate', json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f'HTTP {resp.status}: {await resp.text()}')
        if not payload.get('stream'):
            body = await resp.read()
            data = json.loads(body)
            return data['output_ids'], data['meta_info']['output_token_logprobs'] or [], [data['meta_info']], len(body)
        output_ids, token_logprobs, metas, size = [], [], [], 0
        async for line in resp.content:
            size += len(line)
            line = line.strip()
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            data = json.loads(line[len(b'data: '):])
            if 'error' in data:
                raise RuntimeError(data['error'])
            output_ids += data['output_ids']
            token_logprobs += data['meta_info']['output_token_logprobs'] or []
            metas.append(data['meta_info'])
        return output_ids, token_logprobs, metas, size


def flat_rows(token_logprobs, metas, k):
    """Concatenate per-chunk arrays, checking each chunk against its own token
    rows."""
    ids, vals = [], []
    for meta in metas:
        chunk_ids, chunk_vals = decode_flat(meta)
        n = len(meta['output_token_logprobs'] or [])
        if n == 0:
            assert meta.get('output_top_logprobs_shape') is None, meta.get('output_top_logprobs_shape')
            continue
        assert chunk_ids.shape == chunk_vals.shape == (n, k), (chunk_ids.shape, n, k)
        ids.append(chunk_ids)
        vals.append(chunk_vals)
    ids = np.concatenate(ids) if ids else np.empty((0, k), np.int32)
    vals = np.concatenate(vals) if vals else np.empty((0, k), np.float32)
    assert len(ids) == len(token_logprobs)
    return ids, vals


def check_invariants(token_logprobs, ids, vals):
    sampled_vals = np.array([v for v, _ in token_logprobs], dtype=np.float32)
    sampled_ids = np.array([t for _, t in token_logprobs])
    assert (ids >= 0).all()
    assert all(len(set(row)) == len(row) for row in ids.tolist()), 'duplicate ids in a row'
    assert (vals <= 0).all(), vals.max()
    assert (np.diff(vals, axis=1) <= 0).all(), 'row not in descending order'
    assert (np.exp(vals.astype(np.float64)).sum(1) <= 1 + 1e-2).all(), np.exp(vals).sum(1).max()
    hit = ids == sampled_ids[:, None]
    in_row = hit.any(1)
    assert (vals[hit] == sampled_vals[in_row]).all(), 'sampled logprob differs from its row entry'
    assert (sampled_vals[~in_row] <= vals[~in_row, -1]).all(), 'sampled token outside top-k ranks above it'
    return in_row.mean() if len(in_row) else float('nan')


def bf16_representable(vals: np.ndarray) -> bool:
    return bool((vals.view(np.uint32) & 0xFFFF == 0).all())


def base_payload(args, **kwargs):
    payload = dict(prompt=args.prompt, max_tokens=args.max_tokens, return_logprob=True, temperature=1.0, top_p=1.0,
                   top_k=0, min_p=0.0)
    payload.update(kwargs)
    return payload


def compare_rows(a, b, tol):
    """Compare two lists of {id: logprob} rows; return the fraction equal
    exactly."""
    exact = 0
    for i, (ra, rb) in enumerate(zip(a, b)):
        exact += ra == rb
        for tok in ra.keys() & rb.keys():
            assert abs(ra[tok] - rb[tok]) <= tol, (i, tok, ra[tok], rb[tok])
        edge = min(min(ra.values()), min(rb.values()))
        for tok in ra.keys() ^ rb.keys():
            val = ra.get(tok, rb.get(tok))
            assert val <= edge + tol, (i, tok, val, edge)
    return exact / max(len(a), 1)


def as_dicts(ids, vals):
    return [dict(zip(i, v)) for i, v in zip(ids.tolist(), vals.tolist())]


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


async def run_checks(session, args):
    k = args.k
    # 1. Invariants under sampling, streamed and not.
    for stream in (False, True):
        out, tl, metas, _ = await generate(session, args.base_url,
                                           base_payload(args, top_logprobs_num=k, top_logprobs_format='flat',
                                                        stream=stream))
        ids, vals = flat_rows(tl, metas, k)
        rate = check_invariants(tl, ids, vals)
        print(f'[ok] invariants stream={stream}: {len(ids)} rows, sampled token in top-{k}: {rate:.3f}, '
              f'bf16-representable: {bf16_representable(vals)}')

    # 2. Flat == nested, stream == non-stream (greedy).
    greedy = dict(temperature=0.0, top_logprobs_num=k)
    flat_out, flat_tl, flat_metas, _ = await generate(session, args.base_url,
                                                      base_payload(args, **greedy, top_logprobs_format='flat'))
    nest_out, nest_tl, nest_metas, _ = await generate(session, args.base_url, base_payload(args, **greedy))
    strm_out, strm_tl, strm_metas, _ = await generate(session, args.base_url,
                                                      base_payload(args, **greedy, top_logprobs_format='flat',
                                                                   stream=True))
    flat_ids, flat_vals = flat_rows(flat_tl, flat_metas, k)
    strm_ids, strm_vals = flat_rows(strm_tl, strm_metas, k)
    nested = nest_metas[0]['output_top_logprobs']
    n = common_prefix(flat_out, nest_out)
    for i in range(n):
        assert abs(flat_tl[i][0] - nest_tl[i][0]) <= args.tol, (i, flat_tl[i], nest_tl[i])
    rate = compare_rows(as_dicts(flat_ids[:n], flat_vals[:n]),
                        [{tok: val for val, tok in row} for row in nested[:n]], args.tol)
    print(f'[ok] flat ~ nested on {n}/{len(flat_out)} common greedy rows, exact: {rate:.3f}')
    n = common_prefix(flat_out, strm_out)
    rate = compare_rows(as_dicts(flat_ids[:n], flat_vals[:n]), as_dicts(strm_ids[:n], strm_vals[:n]), args.tol)
    print(f'[ok] streamed chunks ~ non-streamed on {n}/{len(flat_out)} common rows '
          f'({len(strm_metas)} chunks), exact: {rate:.3f}')

    # 3. Mixed batch.
    widths = [k, 2, 0, 8]
    results = await asyncio.gather(*[
        generate(session, args.base_url,
                 base_payload(args, top_logprobs_num=w, top_logprobs_format='flat', stream=bool(i % 2)))
        for i, w in enumerate(widths)
    ])
    for w, (_, tl, metas, _) in zip(widths, results):
        if w:
            ids, vals = flat_rows(tl, metas, w)
            check_invariants(tl, ids, vals)
        else:
            assert all(m.get('output_top_logprobs_shape') is None for m in metas)
    print(f'[ok] mixed batch widths {widths}')


async def run_bench(session, args):
    print(f'\nthroughput: {args.concurrency} concurrent requests x {args.bench_tokens} tokens, ignore_eos')
    print(f'{"case":<14}{"stream":<8}{"tok/s":>10}{"bytes/token":>14}')
    cases = [('flat', 0), ('flat', 32), ('flat', 128), ('nested', 128)]
    for stream in (False, True):
        for fmt, k in cases:
            payload = base_payload(args, max_tokens=args.bench_tokens, ignore_eos=True, top_logprobs_num=k,
                                   top_logprobs_format=fmt, stream=stream)
            start = time.perf_counter()
            results = await asyncio.gather(*[generate(session, args.base_url, payload)
                                             for _ in range(args.concurrency)])
            elapsed = time.perf_counter() - start
            tokens = sum(len(out) for out, *_ in results)
            size = sum(r[3] for r in results)
            print(f'{fmt + "/k=" + str(k):<14}{str(stream):<8}{tokens / elapsed:>10.0f}{size / tokens:>14.0f}')


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-url', default='http://127.0.0.1:23333')
    parser.add_argument('--prompt', default='Write a short story about a lighthouse keeper.')
    parser.add_argument('--k', type=int, default=128, help='top_logprobs_num for the checks.')
    parser.add_argument('--max-tokens', type=int, default=256, help='Output tokens per check request.')
    parser.add_argument('--bench-tokens', type=int, default=2048, help='Output tokens per benchmark request.')
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--tol', type=float, default=0.125, help='Value tolerance between separate requests.')
    parser.add_argument('--skip-bench', action='store_true')
    args = parser.parse_args()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        await run_checks(session, args)
        if not args.skip_bench:
            await run_bench(session, args)


if __name__ == '__main__':
    asyncio.run(main())
