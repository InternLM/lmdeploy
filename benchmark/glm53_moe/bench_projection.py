# Copyright (c) OpenMMLab. All rights reserved.
"""Sweep native Triton projections with cold weights and CUDA graphs.

Only cells passing a float64 comparison and a measured speed gate are selected.
"""

import argparse
import dataclasses
import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lmdeploy.pytorch.kernels.cuda.skinny_gemm import SkinnyGemmConfig, skinny_gemm

SHAPES = {
    'kpool_wk_gate': (128, 4096),
    'indexer_wq_b': (4096, 1536),
    'kda_beta': (64, 4096),
    'kda_fg_a': (256, 4096),
    'kda_qkv': (24576, 4096),
    'kda_o': (4096, 8192),
    'lm_head': (154880, 4096),
}


def configs(m, n, k):
    candidates = [
        SkinnyGemmConfig(bn, bk, warps, unroll) for bn, bk, warps, unroll in [
            (1, 256, 4, 1),
            (2, 256, 4, 2),
            (4, 256, 4, 2),
            (1, 512, 4, 2),
            (2, 512, 4, 2),
            (4, 512, 4, 2),
            (1, 1024, 4, 1),
            (2, 1024, 4, 1),
            (1, 1024, 8, 1),
            (2, 1024, 8, 1),
            (1, 4096, 4, 1),
            (1, 8192, 8, 1),
            (8, 128, 4, 2),
            (16, 128, 4, 2),
            (32, 128, 4, 2),
            (8, 256, 4, 4),
            (16, 256, 4, 4),
            (32, 256, 8, 2),
            (8, 512, 4, 2),
            (16, 512, 8, 2),
        ]
    ]
    block_m = 2**(m - 1).bit_length()
    candidates = [
        c for c in candidates if c.block_k <= 2**(k - 1).bit_length() and block_m * c.block_n * c.block_k <= 16384
    ]
    return candidates


def make_graph(fn, x, weights):
    fn(x, weights[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = []
        for weight in weights:
            out = fn(x, weight)
            # Add a small inter-GEMM kernel, as in the original PR's chain.
            out.add_(0.0)
            outputs.append(out)
    return graph, outputs


def elapsed(graph, replays):
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / replays


def validate(fn, m, n, k):
    results = []
    weight = torch.randn((n, k), device='cuda', dtype=torch.bfloat16) * k**-0.5
    for kind in ('normal', 'outlier', 'sparse'):
        # Exercise a column slice with a wider row stride too.
        storage = torch.randn((m, k + 128), device='cuda', dtype=torch.bfloat16)
        x = storage[:, :k]
        if kind == 'outlier':
            x[:, ::128] *= 16
        if kind == 'sparse':
            x[:, 1::2] = 0
        actual = fn(x, weight)
        worst = 0.0
        for offset in range(0, n, 1024):
            reference = F.linear(x.double(), weight[offset:offset + 1024].double())
            torch.testing.assert_close(actual[:, offset:offset + 1024].double(), reference, rtol=2**-8, atol=7.5e-5)
            worst = max(worst, (actual[:, offset:offset + 1024].double() - reference).abs().max().item())
        results.append(dict(kind=kind, max_abs_error=worst))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--m', type=int, nargs='+', default=[1, 2, 4, 6, 8, 12, 16])
    parser.add_argument('--shapes', nargs='+', choices=SHAPES, default=list(SHAPES))
    parser.add_argument('--cold-mib', type=int, default=512)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--replays', type=int, default=8)
    parser.add_argument('--shortlist',
                        type=Path,
                        nargs='+',
                        help='Recheck only previously selected cells after all model warmup has finished.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.manual_seed(65306)
    report = dict(started=time.time(),
                  device=torch.cuda.get_device_name(),
                  torch=torch.__version__,
                  cold_mib=args.cold_mib,
                  trials=args.trials,
                  replays=args.replays,
                  rows=[],
                  selected=[],
                  completed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shortlist = {}
    if args.shortlist:
        for path in args.shortlist:
            source = json.loads(path.read_text())
            assert source['completed']
            for row in source['selected']:
                config = {key: value for key, value in row['config'].items() if key != 'kind'}
                shortlist.setdefault((row['shape'], row['m']), []).append(SkinnyGemmConfig(**config))

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    with torch.inference_mode():
        for name in args.shapes:
            n, k = SHAPES[name]
            copies = max(1, math.ceil(args.cold_mib * 2**20 / (n * k * 2)))
            weights = torch.randn((copies, n, k), device='cuda', dtype=torch.bfloat16)
            weights *= k**-0.5
            for m in args.m:
                if args.shortlist and (name, m) not in shortlist:
                    continue
                print('START', name, m, n, k, 'weight_copies', copies, flush=True)
                x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
                base_graph, base_outputs = make_graph(F.linear, x, weights)
                candidates = []
                for config in shortlist.get((name, m), configs(m, n, k)):
                    def fn(a, b, config=config):
                        return skinny_gemm(a, b, config)
                    try:
                        graph, outputs = make_graph(fn, x, weights)
                        times = []
                        for trial in range(args.trials):
                            if trial % 2:
                                candidate = elapsed(graph, args.replays)
                                baseline = elapsed(base_graph, args.replays)
                            else:
                                baseline = elapsed(base_graph, args.replays)
                                candidate = elapsed(graph, args.replays)
                            times.append(
                                dict(baseline_us=baseline / copies,
                                     candidate_us=candidate / copies,
                                     speedup=baseline / candidate))
                        row = dict(shape=name,
                                   m=m,
                                   n=n,
                                   k=k,
                                   config=dataclasses.asdict(config),
                                   trials=times,
                                   speedup=statistics.median(t['speedup'] for t in times))
                        candidates.append(row)
                        report['rows'].append(row)
                        del graph, outputs
                    except Exception as exc:
                        report['rows'].append(
                            dict(shape=name, m=m, n=n, k=k, config=dataclasses.asdict(config), error=repr(exc)))
                        print('CONFIG_ERROR', repr(exc), flush=True)
                    save()
                if candidates:
                    best = max(candidates, key=lambda row: row['speedup'])
                    selected = best['speedup'] >= 1.05 and min(t['speedup'] for t in best['trials']) >= 1.02
                    if selected:
                        config = SkinnyGemmConfig(**best['config'])
                        best['validation'] = validate(lambda a, b: skinny_gemm(a, b, config), m, n, k)
                        report['selected'].append(best)
                    print('BEST', name, m, round(best['speedup'], 4), 'selected', selected, best['config'], flush=True)
                    save()
                del base_graph, base_outputs, x
            del weights
            gc.collect()
            torch.cuda.empty_cache()
    report['completed'] = True
    report['finished'] = time.time()
    save()
    print('COMPLETE', len(report['selected']), 'selected cells', flush=True)


if __name__ == '__main__':
    main()
