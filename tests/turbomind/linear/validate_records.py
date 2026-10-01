#!/usr/bin/env python3
"""Validate tuned dispatch records against the PyTorch reference, sharded
across GPUs.

For every concrete ``records.<case>__tp<TP>__ep<EP>`` file, run bench_linear
with --import (disables tuning) and --iters 0 (runs reference + compare +
check_tolerances). A case fails if the process exits non-zero or the log
contains a Traceback.

Usage: validate_records.py --records-root DIR --outdir DIR [--gpus 0,1,..]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

BATCHES = ('1,2,3,4,5,7,8,15,16,17,31,32,33,63,64,65,96,127,128,129,192,224,255,'
           '256,257,511,512,513,1023,1024,1025,2047,2048,2049,4095,4096,4097,8191,8192,8233,16384')

_RECORD_RE = re.compile(r'^records\.(?P<case>.+)__tp(?P<tp>[1-9]\d*)__ep(?P<ep>[1-9]\d*)$')


@dataclass(frozen=True)
class RecordCase:
    path: Path
    case_name: str
    tp: int
    ep: int

    @property
    def tag(self) -> str:
        return f'{self.case_name}__tp{self.tp}__ep{self.ep}'


@dataclass(frozen=True)
class ValidationResult:
    record: RecordCase
    log: Path
    returncode: int


def parse_record(path: Path) -> RecordCase | None:
    match = _RECORD_RE.fullmatch(path.name)
    if match is None:
        return None
    return RecordCase(
        path=path.resolve(),
        case_name=match.group('case'),
        tp=int(match.group('tp')),
        ep=int(match.group('ep')),
    )


def validate_shard(
    gpu: str,
    records: list[RecordCase],
    batches: str,
    outdir: Path,
    results: list[ValidationResult],
    lock: threading.Lock,
) -> None:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    repo_root = Path(__file__).resolve().parents[3]

    for record in records:
        log = outdir / f'{record.tag}.log'
        cmd = [
            sys.executable,
            '-m',
            'tests.turbomind.linear.bench_linear',
            '--suite',
            'full',
            '--case',
            record.case_name,
            '--tp',
            str(record.tp),
            '--ep',
            str(record.ep),
            '--exact-parallel',
            '--batch',
            batches,
            '--import',
            str(record.path),
            '--iters',
            '0',
        ]
        with log.open('w') as f:
            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=repo_root,
                check=False,
            )
        result = ValidationResult(record=record, log=log, returncode=proc.returncode)
        with lock:
            results.append(result)
            print(f'[gpu{gpu}] {record.tag} rc={proc.returncode}', flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--records-root', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--gpus', type=str, default='0')
    ap.add_argument('--cases', type=str, default=None)
    ap.add_argument('--batch', type=str, default=BATCHES)
    args = ap.parse_args()

    records = [
        record
        for path in sorted(args.records_root.glob('records.*'))
        if path.is_file() and (record := parse_record(path)) is not None
    ]
    if args.cases:
        want = {name.strip() for name in args.cases.split(',') if name.strip()}
        records = [record for record in records if record.case_name in want]
    if not records:
        print('no records found', file=sys.stderr)
        return 1

    gpus = [gpu.strip() for gpu in args.gpus.split(',') if gpu.strip()]
    if not gpus:
        print('no GPUs selected', file=sys.stderr)
        return 1
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Deterministic modulo split of the case list across GPUs.
    shards = [
        [record for j, record in enumerate(records) if j % len(gpus) == i]
        for i in range(len(gpus))
    ]
    results: list[ValidationResult] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=validate_shard,
            args=(gpu, shard, args.batch, args.outdir, results, lock),
        )
        for gpu, shard in zip(gpus, shards)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Report
    fails = [result for result in results if result.returncode != 0]
    tracebacks = [
        result
        for result in results
        if 'Traceback' in result.log.read_text(errors='ignore')
    ]
    print(f'validated: {len(results)}/{len(records)}')
    print(f'nonzero exits: {len(fails)}  logs with Traceback: {len(tracebacks)}')
    for result in fails:
        print(f'  FAIL rc={result.returncode}: {result.record.tag}')
    for result in tracebacks:
        print(f'  TRACEBACK: {result.record.tag}')
    return 0 if not fails and not tracebacks and len(results) == len(records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
