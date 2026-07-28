#!/usr/bin/env python3
"""Aggregate SM90 BF16 kernel usage from a bench_linear log.

`DispatchCache::Impl::Summary` (src/turbomind/kernels/gemm/dispatch_cache.cu) prints one block of
``kernel_name: count`` lines every time `Gemm::Export` is called.  This script sums those counts
across all per-case exports emitted by a full-suite run and reports used/unused SM90 BF16 kernels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_PATTERN = re.compile(r'^(sm90_bf16_bf16_bf16_.+):\s+(\d+)$')


def parse_log(path: Path, pattern: re.Pattern[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.read_text().splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        name = m.group(1)
        counts[name] = counts.get(name, 0) + int(m.group(2))
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path, help='bench_linear log file')
    p.add_argument('--pattern', default=r'^(sm90_bf16_bf16_bf16_.+):\s+(\d+)$',
                   help='regex with two groups (name, count)')
    p.add_argument('--json', type=Path, default=None, help='write aggregated counts to JSON')
    p.add_argument('--unused-only', action='store_true', help='only print unused kernels')
    args = p.parse_args(argv)

    pattern = re.compile(args.pattern)
    counts = parse_log(args.log, pattern)
    if not counts:
        print('No SM90 BF16 kernel usage lines found in log.', file=sys.stderr)
        return 1

    used = {name: c for name, c in counts.items() if c > 0}
    unused = {name: c for name, c in counts.items() if c == 0}

    if not args.unused_only:
        print(f'SM90 BF16 kernels found in log: {len(counts)}')
        print(f'  used   : {len(used)}')
        print(f'  unused : {len(unused)}')
        print(f'  total dispatch records across all case exports: {sum(counts.values())}')
        print()
        print('Used kernels (sorted by count):')
        for name, c in sorted(used.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f'  {c:6d}  {name}')
        print()

    print('Unused kernels (candidates for pruning):')
    for name in sorted(unused):
        print(f'  0  {name}')

    if args.json:
        args.json.write_text(json.dumps({
            'used': used,
            'unused': unused,
        }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
