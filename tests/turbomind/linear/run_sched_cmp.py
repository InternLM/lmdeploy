#!/usr/bin/env python3
"""Row-major vs col-major tile-scheduler comparison for MoE gate_up/down (tp=1,
ep=1).

Runs bench_linear with real tuning (--iters > 0 skips the pre-tune forward, so the
GEMM Measure path actually measures) and dumps per-spec measurements via
TM_GEMM_TUNE_VERBOSE=1. Kernel names encode the scheduler raster order in the
trailing policy digits: `_00` = col-major scheduler, `_01` = row-major scheduler.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CASES = [
    # gate_up -> indexed (ibb) kernels
    'deepseek_v2_gate_up__bf16_bf16_bf16',
    'deepseek_v2_gate_up__bf16_bf16_bf16__fuse_silu',
    'qwen3_30b_a3b_gate_up__bf16_bf16_bf16',
    'mixtral_8x7b_gate_up__bf16_bf16_bf16',
    # down -> blocked (bbb) kernels
    'deepseek_v2_down__bf16_bf16_bf16',
    'qwen3_30b_a3b_down__bf16_bf16_bf16',
    'mixtral_8x7b_down__bf16_bf16_bf16',
]

BATCHES = '1,3,16,17,64,65,256,1024,4096'


def discover_cases(kind: str = 'moe') -> list[str]:
    """All tp=1, ep=1 cases; kind='moe': gate_up/down with experts;
    kind='dense': expert_num==0."""
    from tests.turbomind.linear.cases import TYPE_BF16, expand_suite

    names = set()
    for r in expand_suite('full', None, None, (TYPE_BF16.name,)):
        c = r.case
        if c.tp != 1 or c.ep != 1:
            continue
        if kind == 'moe' and c.expert_num > 0 and ('gate_up' in c.name or 'down' in c.name):
            names.add(c.name)
        elif kind == 'dense' and c.expert_num == 0:
            names.add(c.name)
    return sorted(names)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir', type=Path, default=Path('tmp/sched_cmp'))
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--cases', type=str, default=None, help='comma list; default: all MoE gate_up/down tp1ep1')
    p.add_argument('--kind', choices=['moe', 'dense'], default='moe')
    p.add_argument('--batches', type=str, default=BATCHES)
    args = p.parse_args()

    case_names = args.cases.split(',') if args.cases else discover_cases(args.kind)
    print(f"cases: {len(case_names)}", file=sys.stderr)

    args.outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['TM_GEMM_TUNE'] = 'swizzle=[0,1,2,3]'
    env['TM_GEMM_TUNE_VERBOSE'] = '1'

    for name in case_names:
        log = args.outdir / f"{name}.log"
        cmd = [
            sys.executable,
            'tests/turbomind/linear/bench_linear.py',
            '--suite', 'full',
            '--case', name,
            '--type', 'bf16_bf16_bf16',
            '--batch', args.batches,
            '--tune',
            '--iters', str(args.iters),
            '--warmup', '5',
            '--no-validate',
            '--export', str(args.outdir / 'records'),
        ]
        print(f"=== {name} ===", file=sys.stderr)
        with log.open('w') as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                 cwd=str(Path(__file__).resolve().parents[3]))
        print(f"  exit={rc} log={log}", file=sys.stderr)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
