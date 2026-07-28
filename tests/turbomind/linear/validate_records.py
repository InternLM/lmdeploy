#!/usr/bin/env python3
"""Validate tuned dispatch records against the PyTorch reference, sharded
across GPUs.

For every case with an exported records dir, run bench_linear with --import (disables
tuning) and --iters 0 (runs reference + compare + check_tolerances). A case fails if
the process exits non-zero or the log contains a Traceback.

Usage: validate_records.py --records-root DIR --outdir DIR [--gpus 0,1,..] [--kind dense|moe]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BATCHES = ('1,2,3,4,5,7,8,15,16,17,31,32,33,63,64,65,96,127,128,129,192,224,255,'
           '256,257,511,512,513,1023,1024,1025,2047,2048,2049,4095,4096,4097,8191,8192,8233,16384')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--records-root', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--gpus', type=str, default='0')
    ap.add_argument('--cases', type=str, default=None)
    args = ap.parse_args()

    records = sorted(args.records_root.glob('records.*__tp1__ep1'))
    if args.cases:
        want = set(args.cases.split(','))
        records = [r for r in records if r.name[len('records.'):-len('__tp1__ep1')] in want]
    if not records:
        print('no records found', file=sys.stderr)
        return 1

    gpus = args.gpus.split(',')
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Deterministic modulo split of the case list across GPUs.
    for i, gpu in enumerate(gpus):
        shard = [str(r) for j, r in enumerate(records) if j % len(gpus) == i]
        (args.outdir / f'_shard{gpu}.txt').write_text('\n'.join(shard))

    running = []
    for gpu in gpus:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        script = args.outdir / f'_run{gpu}.sh'
        script.write_text(f"""#!/bin/bash
while read -r rec || [ -n "$rec" ]; do
  case_name="${{rec##*records.}}"; case_name="${{case_name%__tp1__ep1}}"
  [ -z "$case_name" ] && continue
  {sys.executable} tests/turbomind/linear/bench_linear.py --suite full \\
      --case "$case_name" --type bf16_bf16_bf16 --batch {BATCHES} \\
      --import "$rec" --iters 0 > "{args.outdir}/${{case_name}}.log" 2>&1
  echo "$case_name rc=$?" >> "{args.outdir}/_rc.txt"
done < "{args.outdir}/_shard{gpu}.txt"
""")
        script.chmod(0o755)
        running.append(subprocess.Popen(['bash', str(script)], env=env,
                                        cwd=str(Path(__file__).resolve().parents[3])))
    rcs = [p.wait() for p in running]

    # Report
    rc_file = args.outdir / '_rc.txt'
    fails = []
    done = 0
    if rc_file.exists():
        for line in rc_file.read_text().splitlines():
            name, rc = line.rsplit(' rc=', 1)
            done += 1
            if rc != '0':
                fails.append((name, rc))
    tracebacks = [p.stem for p in args.outdir.glob('*.log')
                  if 'Traceback' in p.read_text(errors='ignore')]
    print(f'validated: {done}/{len(records)}  worker_rcs={rcs}')
    print(f'nonzero exits: {len(fails)}  logs with Traceback: {len(tracebacks)}')
    for name, rc in fails:
        print(f'  FAIL rc={rc}: {name}')
    for name in tracebacks:
        print(f'  TRACEBACK: {name}')
    return 0 if not fails and not tracebacks and done == len(records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
