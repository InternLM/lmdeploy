#!/usr/bin/env python3
"""Chunked full SM90 kernel-usage scan.

The full benchmark run in one process accumulates GPU memory across the large/MoE
model cases and eventually OOMs on the testbed.  This driver splits the full suite
into smaller per-name chunks, runs each chunk in a fresh Python process,
and appends the per-case Summary blocks to a single log for aggregation.

Chunks are distributed across one worker per GPU (--gpus); each worker processes
its chunks sequentially in fresh subprocesses and appends to its own
``bench.gpu<N>.log``.  The per-GPU logs are concatenated into ``bench.log`` when
all workers finish, so downstream aggregation is unchanged:

    python tests/turbomind/linear/kernel_usage_report.py tmp/sm90_bf16_scan5/bench.log

Use --type to select the TypeSpec to scan, e.g. bf16_bf16_bf16 (SM90 BF16 kernels),
bf16_e4m3b128_bf16 (FP8 weight-as-A) or e4m3k128_e4m3b128_bf16 (FP8 v3 act-as-A).
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from tests.turbomind.linear.cases import expand_suite


def chunk_names(names: list[str], size: int) -> list[list[str]]:
    return [names[i : i + size] for i in range(0, len(names), size)]


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(',') if x.strip())


def run_chunk(
    case_names: list[str],
    type_name: str,
    outdir: Path,
    log: Path,
    env: dict[str, str],
    python: str,
    tps: tuple[int, ...],
    eps: tuple[int, ...],
) -> int:
    cmd = [
        python,
        'tests/turbomind/linear/bench_linear.py',
        '--suite',
        'full',
        '--case',
        ','.join(case_names),
        '--type',
        type_name,
        '--tp',
        ','.join(map(str, tps)),
        '--ep',
        ','.join(map(str, eps)),
        '--tune',
        '--iters',
        '0',
        '--no-validate',
        '--export',
        str(outdir / 'records'),
    ]
    with log.open('ab') as f:
        # Append a small marker so we know where each chunk starts.
        f.write(f'\n# CHUNK: {case_names[0]} .. {case_names[-1]}\n'.encode())
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
        proc.wait()
        return proc.returncode


def worker(
    tag: str,
    gpu: str,
    chunks: queue.Queue[list[str]],
    type_name: str,
    outdir: Path,
    base_env: dict[str, str],
    python: str,
    tps: tuple[int, ...],
    eps: tuple[int, ...],
    failures: list[list[str]],
    lock: threading.Lock,
) -> None:
    env = base_env.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    log = outdir / f'bench.gpu{gpu}.log'
    while True:
        try:
            chunk = chunks.get_nowait()
        except queue.Empty:
            return
        with lock:
            print(f'[gpu{gpu}] start ({len(chunk)} cases): {chunk[0]} .. {chunk[-1]}', flush=True)
        rc = run_chunk(chunk, type_name, outdir, log, env, python, tps, eps)
        with lock:
            print(f'[gpu{gpu}] done rc={rc}: {chunk[0]} .. {chunk[-1]}', flush=True)
            if rc != 0:
                failures.append(chunk)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir', type=Path, default=Path('tmp/sm90_scan'))
    p.add_argument('--type', type=str, default='bf16_bf16_bf16', help='TypeSpec name to scan')
    p.add_argument('--chunk', type=int, default=6, help='case names per subprocess')
    p.add_argument('--gpus', type=str, default='0', help='comma-separated CUDA_VISIBLE_DEVICES values')
    p.add_argument('--tp', type=str, default='1,2,4,8', help='comma-separated TP sizes')
    p.add_argument('--ep', type=str, default='1,2,4,8', help='comma-separated EP sizes')
    p.add_argument('--python', type=str, default=sys.executable)
    args = p.parse_args(argv)

    tps = parse_int_list(args.tp)
    eps = parse_int_list(args.ep)
    gpus = [g.strip() for g in args.gpus.split(',') if g.strip()]

    args.outdir.mkdir(parents=True, exist_ok=True)

    runs = expand_suite('full', None, None, (args.type,), tps=tps, eps=eps)
    names = sorted({r.case.name for r in runs})
    print(f'Total {args.type} case names: {len(names)}, runs: {len(runs)}', file=sys.stderr)

    base_env = os.environ.copy()
    base_env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    base_env['TM_GEMM_TUNE'] = 'swizzle=[0,1,2,3]'

    chunks: queue.Queue[list[str]] = queue.Queue()
    for chunk in chunk_names(names, args.chunk):
        chunks.put(chunk)

    failures: list[list[str]] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker,
            args=(f'w{i}', gpu, chunks, args.type, args.outdir, base_env, args.python, tps, eps, failures, lock),
        )
        for i, gpu in enumerate(gpus)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Merge per-GPU logs into a single log for the aggregation tool.
    merged = args.outdir / 'bench.log'
    with merged.open('wb') as out:
        for gpu in gpus:
            part = args.outdir / f'bench.gpu{gpu}.log'
            if part.exists():
                out.write(part.read_bytes())

    if failures:
        print(f'\n{len(failures)} chunk(s) failed:', file=sys.stderr)
        for chunk in failures:
            print(f'  {chunk[0]} .. {chunk[-1]}', file=sys.stderr)

    print(f'\nDone. Log: {merged}', file=sys.stderr)
    print(
        f'Aggregate usage: python tests/turbomind/linear/kernel_usage_report.py {merged}',
        file=sys.stderr,
    )
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
