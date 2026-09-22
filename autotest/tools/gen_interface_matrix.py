#!/usr/bin/env python3
"""Generate GitHub Actions matrix JSON from per-model ``interface`` config."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

AUTOTEST_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUTOTEST_DIR))

from utils.config_utils import get_interface_matrix  # noqa: E402


def _parse_backends(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == '':
        return None
    text = raw.strip()
    # Accept JSON list or Python-ish "['a', 'b']"
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = json.loads(text.replace("'", '"'))
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise argparse.ArgumentTypeError(f'backends must be a JSON string list, got {raw!r}')
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', default=os.environ.get('TEST_ENV', 'a100'), help='Matrix env key')
    parser.add_argument(
        '--backends',
        default=None,
        help='JSON list of backends, e.g. \'["turbomind","pytorch"]\'',
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print JSON (default: compact one line for GHA)',
    )
    args = parser.parse_args()
    backends = _parse_backends(args.backends)
    os.environ.setdefault('TEST_ENV', args.env)
    # Interface matrix ignores DEPS_PROFILE pins so MoE/VL rows with deps still appear.
    rows = get_interface_matrix(env_key=args.env, backends=backends, deps_profile='all')
    if not rows:
        print('[]' if not args.pretty else '[]', end='')
        return 0
    if args.pretty:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(rows, ensure_ascii=False, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
