#!/usr/bin/env python3
"""Parse [tune] logs from run_sched_cmp.py and compare row-major vs col-major
tile schedulers for the same kernel tile.

Kernel-name suffix: `_00` = col-major scheduler, `_01` = row-major scheduler.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

TUNE_RE = re.compile(
    r'^\[tune\] (?P<desc>\S+) (?P<kernel>\S+) swizzle=(?P<swizzle>\d+) '
    r'splits=(?P<splits>\d+) measured=(?P<ms>[0-9.eE+-]+)$'
)
ROW_RE = re.compile(r"\{'case':[^}]*\}")


def strip_order(kernel: str) -> tuple[str, str]:
    """Split kernel name into (name without trailing policy digits,
    scheduler)."""
    base, _, pol = kernel.rpartition('_')
    if pol in ('00', '01'):
        return base, ('row' if pol == '01' else 'col')
    return kernel, '?'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('logs', nargs='+', type=Path)
    ap.add_argument('--detail', action='store_true', help='print per-tile row/col table')
    args = ap.parse_args()

    # problem desc -> kernel base -> scheduler -> best (min over swizzle) ms
    best: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    # problem desc -> overall winner (kernel, ms) from first sorted [tune] line
    winner: dict[str, tuple[str, float]] = {}
    # benchmark rows: (case, m) -> tflops
    rows: dict[tuple[str, int], float] = {}

    for log in args.logs:
        case = log.name[: -len('.log')]
        for line in log.read_text().splitlines():
            m = TUNE_RE.match(line)
            if m:
                desc = m.group('desc')
                base, order = strip_order(m.group('kernel'))
                ms = float(m.group('ms'))
                cur = best[desc][base].get(order)
                if cur is None or ms < cur:
                    best[desc][base][order] = ms
                if desc not in winner:
                    winner[desc] = (m.group('kernel'), ms)
            elif "{'case':" in line:
                mrow = ROW_RE.search(line)
                if mrow:
                    d = eval(mrow.group(0))  # noqa: S307 - benchmark's own row dict
                    if d.get('tflops'):
                        rows[(case, d['m'])] = d['tflops']

    n_row_wins = n_col_wins = 0
    print(f"{'problem':<46} {'winner-sched':<12} {'winner-tile':<14} {'row-best':>9} {'col-best':>9} {'row/col':>8}")
    for desc in sorted(best, key=lambda d: (d.split('_')[3], [int(x) for x in re.findall(r'\d+', d.split('_')[-2])])):
        per_tile = best[desc]
        w_kernel, w_ms = winner[desc]
        _, w_order = strip_order(w_kernel)
        if w_order == 'row':
            n_row_wins += 1
        elif w_order == 'col':
            n_col_wins += 1
        # same-tile row vs col comparison, best-matched pair
        row_best = min((v['row'] for v in per_tile.values() if 'row' in v), default=None)
        col_best = min((v['col'] for v in per_tile.values() if 'col' in v), default=None)
        w_tile = re.search(r'_(\d+x\d+x\d+)_', w_kernel)
        ratio = (row_best / col_best) if (row_best and col_best) else float('nan')
        row_str = f"{row_best if row_best else float('nan'):>9.4f}"
        col_str = f"{col_best if col_best else float('nan'):>9.4f}"
        print(f"{desc:<46} {w_order:<12} {(w_tile.group(1) if w_tile else ''):<14} "
              f'{row_str} {col_str} {ratio:>8.3f}')
        if args.detail:
            for base in sorted(per_tile):
                v = per_tile[base]
                r = v.get('row')
                c = v.get('col')
                tile = re.search(r'_(\d+x\d+x\d+)_', base).group(1)
                r_s = f'{r:.4f}' if r else '-'
                c_s = f'{c:.4f}' if c else '-'
                mark = ''
                if r and c:
                    mark = f'row/col={r / c:.3f}'
                print(f'    {tile:<12} row={r_s:<9} col={c_s:<9} {mark}')

    print(f'\noverall winner scheduler: row={n_row_wins} col={n_col_wins} (of {len(best)} problems)')

    print('\nbenchmark rows (tuned winner tflops):')
    for (case, m), tf in sorted(rows.items()):
        print(f'  {case:<52} m={m:<6} {tf:8.2f} TFLOPS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
