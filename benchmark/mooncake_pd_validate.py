# Copyright (c) OpenMMLab. All rights reserved.
"""Compare PD replay against direct inference, including warm and partial
prefixes.

Run against an already deployed router and decode endpoint. JSONL output includes token IDs so correctness checks do not
depend on text parsing or reasoning tags. The first case does not clear local caches or previously stored objects. All
cases run before a nonzero exit reports any token mismatch.
"""
import argparse
import json
import time
from urllib.request import ProxyHandler, Request, build_opener


def generate(url, model, tokens, max_tokens):
    payload = dict(model=model, messages=[], input_ids=tokens, temperature=0, max_tokens=max_tokens,
                   stream=True, return_token_ids=True, ignore_eos=True)
    request = Request(url.rstrip('/') + '/v1/chat/completions', json.dumps(payload).encode(),
                      {'Content-Type': 'application/json'})
    start = time.perf_counter()
    first = None
    output_ids = []
    chunks = []
    finish = None
    with build_opener(ProxyHandler({})).open(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                continue
            result = json.loads(line[6:])
            if 'error' in result:
                raise RuntimeError(result['error'])
            for choice in result['choices']:
                ids = choice.get('output_ids') or []
                if ids:
                    now = time.perf_counter()
                    first = first or now
                    chunks.append(dict(time_ms=(now - start) * 1000, tokens=len(ids)))
                    output_ids.extend(ids)
                finish = choice.get('finish_reason') or finish
    elapsed = time.perf_counter() - start
    if finish not in ('stop', 'length') or not output_ids:
        raise RuntimeError(f'Incomplete generation: finish={finish}, tokens={len(output_ids)}')
    ttft = first - start
    return dict(ttft_ms=ttft * 1000, latency_ms=elapsed * 1000, output_ids=output_ids,
                tpot_ms=(elapsed - ttft) * 1000 / max(1, len(output_ids) - 1), chunks=chunks, finish_reason=finish)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pd-url', required=True)
    parser.add_argument('--direct-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[1, 2, 3, 4, 63, 64, 65, 66, 129, 8192, 32768])
    parser.add_argument('--max-tokens', type=int, default=32)
    args = parser.parse_args()
    differences = 0
    for length in args.lengths:
        tokens = ([100, 101, 102, 103] * ((length + 3) // 4))[:length]
        for case in ('first', 'repeated', 'partial'):
            if case == 'partial':
                tokens[length // 2] = 104
            pd = generate(args.pd_url, args.model, tokens, args.max_tokens)
            direct = generate(args.direct_url, args.model, tokens, args.max_tokens)
            equal = pd['output_ids'] == direct['output_ids']
            differences += not equal
            print(json.dumps(dict(length=length, case=case, equal=equal,
                                  pd=pd, direct=direct)), flush=True)
    raise SystemExit(bool(differences))


if __name__ == '__main__':
    main()
