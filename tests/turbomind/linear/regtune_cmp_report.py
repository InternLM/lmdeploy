#!/usr/bin/env python3
"""Compare two tune-verbose sweep dirs: per-problem best measured time, new vs old.

Usage: regtune_cmp_report.py OLD_DIR NEW_DIR [--threshold 0.005]
Problem key = (case, m, n, k[, fuse]) taken from the [tune] line prefix.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TUNE_RE = re.compile(r'^\[tune\] (\S+?) (\S+) swizzle=(\d+) splits=(\d+) measured=([0-9.eE+-]+)')


def best_per_problem(d: Path) -> dict:
    best = {}
    for log in sorted(d.glob('*.log')):
        case = log.stem
        for line in log.read_text(errors='ignore').splitlines():
            m = TUNE_RE.match(line)
            if not m:
                continue
            prob, kern, sw, sp, t = m.group(1), m.group(2), m.group(3), m.group(4), float(m.group(5))
            if t <= 0:
                continue
            key = (case, prob)
            cur = best.get(key)
            if cur is None or t < cur[0]:
                best[key] = (t, kern, sw, sp)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('old', type=Path)
    ap.add_argument('new', type=Path)
    ap.add_argument('--threshold', type=float, default=0.005, help='relative change to report')
    args = ap.parse_args()

    old, new = best_per_problem(args.old), best_per_problem(args.new)
    common = sorted(set(old) & set(new))
    only_new = set(new) - set(old)
    only_old = set(old) - set(new)
    if only_new or only_old:
        print(f'problems only in new: {len(only_new)}, only in old: {len(only_old)}')

    ratios = []
    improved, regressed = [], []
    for key in common:
        t_old, t_new = old[key][0], new[key][0]
        r = t_new / t_old
        ratios.append(r)
        if r < 1 - args.threshold:
            improved.append((r, key, old[key], new[key]))
        elif r > 1 + args.threshold:
            regressed.append((r, key, old[key], new[key]))

    n = len(ratios)
    mean = sum(ratios) / n
    geo = 1.0
    for r in ratios:
        geo *= r
    geo **= 1.0 / n
    print(f'problems: {n}  mean new/old={mean:.4f}  geomean={geo:.4f}')
    print(f'improved >{args.threshold:.1%}: {len(improved)} ({len(improved)/n:.1%})  '
          f'regressed: {len(regressed)} ({len(regressed)/n:.1%})')

    print('\n== top improvements ==')
    for r, key, o, nn in sorted(improved)[:15]:
        print(f'{r:.3f}  {key[0]} {key[1]}  {o[0]:.4f}->{nn[0]:.4f}ms  '
              f"old={o[1].split('tnt_')[-1]} sw{o[2]} new={nn[1].split('tnt_')[-1]} sw{nn[2]}")
    print('\n== top regressions ==')
    for r, key, o, nn in sorted(regressed, reverse=True)[:15]:
        print(f'{r:.3f}  {key[0]} {key[1]}  {o[0]:.4f}->{nn[0]:.4f}ms  '
              f"old={o[1].split('tnt_')[-1]} sw{o[2]} new={nn[1].split('tnt_')[-1]} sw{nn[2]}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
