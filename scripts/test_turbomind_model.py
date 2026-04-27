#!/usr/bin/env python3
"""Smoke-test one TurboMind model for agent/subagent harnesses.

Code flow: parse argv → configure HF/GPU and run one inference → print sections.

Stdout is plain text in short sections, for example:

  --- setup ---
  model: <path>
  tp: <tp>
  gpus: <gpus>
  TM_DEBUG_LEVEL: DEBUG    (only if --debug was passed)

  --- timing ---
  pipeline load: <s> s
  inference: <s> s

  --- tokens ---
  input: <n>
  generated: <n>

  --- response begin ---
  <full decoded text>
  --- response end ---

Exit code: 0 only if no uncaught exception (pipeline load + inference complete).
On failure the full traceback is printed to stderr.
Output quality is not validated.

Usage (from repo root):

  python scripts/test_turbomind_model.py \\
      [--debug] <model_path> <cache_dir> <tp> <gpus>

Optional --debug sets TM_DEBUG_LEVEL=DEBUG before loading TurboMind so asynchronous
CUDA errors surface after kernel launch (see TurboMind CUDA helpers).

Example gpus: "0" for tp=1, "0,1" for tp=2.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from typing import NamedTuple

import huggingface_hub.constants as hf_constants


class SmokeResult(NamedTuple):
    create_s: float
    infer_s: float
    text: str
    input_token_len: int
    generate_token_len: int


def _set_hf_cache(path: str) -> None:
    hf_constants.HF_HUB_CACHE = path
    hf_constants.HF_HUB_OFFLINE = 1


def parse_args(argv: list[str]) -> tuple[str, str, int, str, bool]:
    prog = os.path.basename(argv[0]) if argv else 'test_turbomind_model.py'
    rest = [a for a in argv[1:] if a != '--debug']
    debug = len(rest) != len(argv) - 1

    if len(rest) != 4:
        print(
            f'usage: {prog} [--debug] <model_path> <cache_dir> <tp> <gpus>',
            file=sys.stderr,
        )
        sys.exit(2)

    model_path, cache_dir, tp_s, gpus = rest
    try:
        tp = int(tp_s)
    except ValueError:
        print(f'invalid tp: {tp_s!r}', file=sys.stderr)
        sys.exit(2)
    return model_path, cache_dir, tp, gpus, debug


def run_smoke_infer(
    model_path: str,
    cache_dir: str,
    tp: int,
    gpus: str,
    *,
    debug: bool = False,
) -> SmokeResult:
    _set_hf_cache(cache_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    if debug:
        os.environ['TM_DEBUG_LEVEL'] = 'DEBUG'

    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

    engine_config = TurbomindEngineConfig(
        async_=1,
        max_batch_size=4,
        session_len=4096,
        cache_max_entry_count=0.5,
        max_prefill_token_num=1024,
        tp=tp,
        dp=1,
        enable_metrics=False,
        communicator='nccl',
    )
    gen_config = GenerationConfig(max_new_tokens=128, do_sample=False)
    prompt = 'Write a short paragraph about the importance of reading books.'

    t0 = time.perf_counter()
    with pipeline(model_path, backend_config=engine_config, log_level='WARNING') as pipe:
        create_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        out = pipe([prompt], gen_config=gen_config, do_preprocess=True)
        infer_s = time.perf_counter() - t1

    res = out[0]
    text = res.text if hasattr(res, 'text') else str(res)
    input_token_len = getattr(res, 'input_token_len', -1)
    generate_token_len = getattr(res, 'generate_token_len', -1)
    return SmokeResult(create_s, infer_s, text, input_token_len, generate_token_len)


def print_report(
    model_path: str,
    tp: int,
    gpus: str,
    result: SmokeResult,
    *,
    debug: bool = False,
) -> None:
    if not result.text.strip():
        print('warning: empty response text', file=sys.stderr)

    print('--- setup ---')
    print(f'model: {model_path}')
    print(f'tp: {tp}')
    print(f'gpus: {gpus}')
    if debug:
        print('TM_DEBUG_LEVEL: DEBUG')
    print()
    print('--- timing ---')
    print(f'pipeline load: {result.create_s:.2f} s')
    print(f'inference: {result.infer_s:.2f} s')
    print()
    print('--- tokens ---')
    print(f'input: {result.input_token_len}')
    print(f'generated: {result.generate_token_len}')
    print()
    print('--- response begin ---')
    print(result.text, end='')
    if result.text and not result.text.endswith('\n'):
        print()
    print('--- response end ---')


def main() -> None:
    model_path, cache_dir, tp, gpus, debug = parse_args(sys.argv)
    result = run_smoke_infer(model_path, cache_dir, tp, gpus, debug=debug)
    print_report(model_path, tp, gpus, result, debug=debug)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
