#!/usr/bin/env python3
"""Smoke-test one TurboMind model for agent/subagent harnesses.

Code flow: parse argv → resolve prompts → configure HF/GPU and run batch inference
→ print sections.

Stdout is plain text in short sections, for example:

  --- setup ---
  model: <path>
  tp: <tp>
  gpus: <gpus>
  max_new_tokens: 256
  async: 1
  session_len: 16384
  max_batch_size: 8
  enable_prefix_caching: 0
  prompt_count: 1
  prompt_source: default
  CUDA_LAUNCH_BLOCKING: 1    (only if --debug was passed)

  --- timing ---
  pipeline load: <s> s
  inference: <s> s

  --- tokens ---
  [0] source_index: 0
  [0] prompt: Write a short paragraph about the importance of reading books.
  [0] input: <n>
  [0] generated: <n>

  --- response 0 begin ---
  <full decoded text>
  --- response 0 end ---

Exit code: 0 only if no uncaught exception (pipeline load + inference complete).
On failure the full traceback is printed to stderr.
Output quality is not validated.

Usage (from repo root):

  python scripts/test_turbomind_model.py \\
      --model-id ID \\
      --cache-dir PATH \\
      --tp N \\
      --gpus DEVICES \\
      [--prompt TEXT ...] \\
      [--prompt-file PATH] \\
      [--prompt-ids N [N ...]] \\
      [--max-new-tokens N] \\
      [--async {0,1}] \\
      [--session-len N] \\
      [--max-batch-size N] \\
      [--enable-prefix-caching] \\
      [--max-prefill-token-num N] \\
      [--linear-prefix-cache-min-interval N] \\
      [--cache-prompt-boundary] \\
      [--cache-generation-boundary] \\
      [--cache-boundary-policy NAME] \\
      [--debug]

Optional prompts: repeat --prompt for multiple strings, or --prompt-file for a JSON
array of strings. --prompt and --prompt-file are mutually exclusive. When omitted, a
built-in default prompt is used. --prompt-ids selects/reorders/duplicates by 0-based
index; when omitted, all prompts run in order.

Example --prompt-ids (0-based; repeats run the same prompt again):

  # prompts.json: ["First prompt.", "Second prompt."]
  # Run second, then first twice (three responses total)
  python scripts/test_turbomind_model.py \\
      --model-id ID --cache-dir PATH --tp 1 --gpus 0 \\
      --prompt-file prompts.json \\
      --prompt-ids 1 0 0

  # Same with CLI prompts: run "B", then "A" twice
  python scripts/test_turbomind_model.py \\
      --model-id ID --cache-dir PATH --tp 1 --gpus 0 \\
      --prompt "A" --prompt "B" \\
      --prompt-ids 1 0 0

Example with prefix caching and repeated prompts:

  python scripts/test_turbomind_model.py \\
      --model-id ID --cache-dir PATH --tp 2 --gpus 0,1 \\
      --enable-prefix-caching \\
      --prompt "A" --prompt "B" \\
      --prompt-ids 0 0 1

Optional engine params (defaults shown): --max-new-tokens 256, --async 1,
--session-len 16384, --max-batch-size 8. Prefix caching: pass
--enable-prefix-caching (default off).

Optional --debug sets CUDA_LAUNCH_BLOCKING=1 before loading TurboMind so CUDA
kernels run synchronously and errors surface at the launch site.

Example gpus: "0" for tp=1, "0,1" for tp=2.

Python (repo root on PYTHONPATH):

  from scripts.test_turbomind_model import run_smoke_test

  result = run_smoke_test(
      model_id="/path/to/model",
      cache_dir="/nvme2/huggingface_hub/hub",
      tp=2,
      gpus="0,1",
      prompts=["Hello", "World"],
      prompt_ids=[0, 1, 0],
      enable_prefix_caching=True,
      max_new_tokens=512,
      emit_report=False,
  )
  assert len(result.responses) == 3
  assert result.responses[0].text.strip()

SmokeResult.responses is a list of PromptResult (index, source_index, prompt_preview,
text, input_token_len, generate_token_len). Use result.responses[0].text for the
first decoded output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import NamedTuple

import huggingface_hub.constants as hf_constants

DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_ASYNC = 1
DEFAULT_SESSION_LEN = 16384
DEFAULT_MAX_BATCH_SIZE = 8
DEFAULT_MAX_PREFILL_TOKEN_NUM = 1024
DEFAULT_LINEAR_PREFIX_CACHE_MIN_INTERVAL = 0
DEFAULT_CACHE_BOUNDARY_POLICY = ''
DEFAULT_PROMPT = 'Write a short paragraph about the importance of reading books.'
PROMPT_PREVIEW_LEN = 64


class PromptConfigError(ValueError):
    """Invalid prompt CLI/API configuration (maps to exit 2)."""


class PromptResult(NamedTuple):
    index: int
    source_index: int
    prompt_preview: str
    text: str
    input_token_len: int
    generate_token_len: int


class ResolvedPrompts(NamedTuple):
    prompts: list[str]
    source_indices: list[int]
    source: str  # 'default' | 'cli' | 'file'


class SmokeResult(NamedTuple):
    create_s: float
    infer_s: float
    responses: list[PromptResult]


def _set_hf_cache(path: str) -> None:
    hf_constants.HF_HUB_CACHE = path
    hf_constants.HF_HUB_OFFLINE = 1


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f'{value!r} must be >= 1')
    return n


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f'{value!r} must be >= 0')
    return n


def _validate_engine_params(
    *,
    max_new_tokens: int,
    async_: int,
    session_len: int,
    max_batch_size: int,
    max_prefill_token_num: int,
    linear_prefix_cache_min_interval: int,
) -> None:
    if max_new_tokens < 1:
        raise ValueError(f'max_new_tokens must be >= 1, got {max_new_tokens}')
    if async_ not in (0, 1):
        raise ValueError(f'async_ must be 0 or 1, got {async_}')
    if session_len < 1:
        raise ValueError(f'session_len must be >= 1, got {session_len}')
    if max_batch_size < 1:
        raise ValueError(f'max_batch_size must be >= 1, got {max_batch_size}')
    if max_prefill_token_num < 1:
        raise ValueError(f'max_prefill_token_num must be >= 1, got {max_prefill_token_num}')
    if linear_prefix_cache_min_interval < 0:
        raise ValueError(
            f'linear_prefix_cache_min_interval must be >= 0, got {linear_prefix_cache_min_interval}')


def _format_prompt_preview(text: str) -> str:
    escaped = text.replace('\n', '\\n')
    if len(escaped) > PROMPT_PREVIEW_LEN:
        return escaped[:PROMPT_PREVIEW_LEN] + '...'
    return escaped


def _load_prompt_file(path: str) -> list[str]:
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise PromptConfigError(f'prompt file not found: {path}')
    try:
        data = json.loads(prompt_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise PromptConfigError(f'invalid JSON in prompt file {path}: {exc}') from exc
    if not isinstance(data, list):
        raise PromptConfigError(f'prompt file must contain a JSON array of strings: {path}')
    if not data:
        raise PromptConfigError(f'prompt file is empty: {path}')
    for i, item in enumerate(data):
        if not isinstance(item, str):
            raise PromptConfigError(f'prompt file entry {i} is not a string: {path}')
        if not item:
            raise PromptConfigError(f'prompt file entry {i} is empty: {path}')
    return data


def resolve_prompts(
    *,
    prompts: list[str] | None = None,
    prompt_file: str | None = None,
    prompt_ids: list[int] | None = None,
) -> ResolvedPrompts:
    if prompts is not None and prompt_file is not None:
        raise PromptConfigError('cannot use both prompts and prompt_file')

    if prompt_file is not None:
        base = _load_prompt_file(prompt_file)
        source = 'file'
    elif prompts is not None:
        if not prompts:
            raise PromptConfigError('prompts list is empty')
        for i, prompt in enumerate(prompts):
            if not prompt:
                raise PromptConfigError(f'prompt {i} is empty')
        base = list(prompts)
        source = 'cli'
    else:
        base = [DEFAULT_PROMPT]
        source = 'default'

    if prompt_ids is None:
        ids = list(range(len(base)))
    else:
        ids = list(prompt_ids)

    for idx in ids:
        if idx < 0 or idx >= len(base):
            raise PromptConfigError(
                f'prompt_ids index {idx} out of range for {len(base)} prompts')

    resolved = [base[i] for i in ids]
    if not resolved:
        raise PromptConfigError('prompt_ids produced an empty prompt list')
    return ResolvedPrompts(resolved, ids, source)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Smoke-test one TurboMind model for agent/subagent harnesses.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
stdout sections: --- setup ---, --- timing ---, --- tokens ---, --- response N begin/end ---
Optional prompts: --prompt, --prompt-file, --prompt-ids
Optional engine params:
    --max-new-tokens
    --async
    --session-len
    --max-batch-size
    --enable-prefix-caching
    --max-prefill-token-num
    --linear-prefix-cache-min-interval
    --cache-prompt-boundary
    --cache-generation-boundary
    --cache-boundary-policy
Exit 0: load + inference complete. Exit 1: exception (traceback on stderr). Exit 2: usage error.
""",
    )
    parser.add_argument('--model-id', required=True, help='HuggingFace model id or local path')
    parser.add_argument('--cache-dir', required=True, help='HF hub cache directory (HF_HUB_CACHE)')
    parser.add_argument('--tp', required=True, type=int, help='Tensor parallel size')
    parser.add_argument('--gpus', required=True, help='CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1"')
    parser.add_argument(
        '--max-new-tokens',
        type=_positive_int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f'GenerationConfig.max_new_tokens (default: {DEFAULT_MAX_NEW_TOKENS})',
    )
    parser.add_argument(
        '--async',
        type=int,
        default=DEFAULT_ASYNC,
        choices=[0, 1],
        dest='async_',
        help='Enable async execution (default: 1). Set 0 to disable, 1 to enable.',
    )
    parser.add_argument(
        '--session-len',
        type=_positive_int,
        default=DEFAULT_SESSION_LEN,
        help=f'TurbomindEngineConfig.session_len (default: {DEFAULT_SESSION_LEN})',
    )
    parser.add_argument(
        '--max-batch-size',
        type=_positive_int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help=f'TurbomindEngineConfig.max_batch_size (default: {DEFAULT_MAX_BATCH_SIZE})',
    )
    parser.add_argument(
        '--enable-prefix-caching',
        action='store_true',
        help='Enable TurboMind prefix caching (TurbomindEngineConfig.enable_prefix_caching)',
    )
    parser.add_argument(
        '--max-prefill-token-num',
        type=_positive_int,
        default=DEFAULT_MAX_PREFILL_TOKEN_NUM,
        help=f'TurbomindEngineConfig.max_prefill_token_num (default: {DEFAULT_MAX_PREFILL_TOKEN_NUM})',
    )
    parser.add_argument(
        '--linear-prefix-cache-min-interval',
        type=_non_negative_int,
        default=DEFAULT_LINEAR_PREFIX_CACHE_MIN_INTERVAL,
        help=('TurbomindEngineConfig.linear_prefix_cache_min_interval '
              f'(default: {DEFAULT_LINEAR_PREFIX_CACHE_MIN_INTERVAL})'),
    )
    parser.add_argument(
        '--cache-prompt-boundary',
        action='store_true',
        help=('Enable prompt-boundary checkpoints '
              '(TurbomindEngineConfig.cache_prompt_boundary; recurrent/hybrid only)'),
    )
    parser.add_argument(
        '--cache-generation-boundary',
        action='store_true',
        help=('Enable generation-boundary checkpoints '
              '(TurbomindEngineConfig.cache_generation_boundary; recurrent/hybrid only)'),
    )
    parser.add_argument(
        '--cache-boundary-policy',
        default=DEFAULT_CACHE_BOUNDARY_POLICY,
        metavar='NAME',
        help=('Cache-boundary publish policy name (TurbomindEngineConfig.cache_boundary_policy; '
              'empty = default policy, "auto" = distance-gated)'),
    )
    parser.add_argument(
        '--prompt',
        action='append',
        default=None,
        metavar='TEXT',
        help='Prompt text (repeatable). Mutually exclusive with --prompt-file.',
    )
    parser.add_argument(
        '--prompt-file',
        default=None,
        metavar='PATH',
        help='JSON file containing an array of prompt strings. Mutually exclusive with --prompt.',
    )
    parser.add_argument(
        '--prompt-ids',
        nargs='+',
        type=int,
        default=None,
        metavar='N',
        help='0-based indices into the prompt list; repeats allowed. Default: all in order.',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Set CUDA_LAUNCH_BLOCKING=1 before TurboMind load',
    )
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args(argv[1:])


def run_smoke_infer(
    model_id: str,
    cache_dir: str,
    tp: int,
    gpus: str,
    resolved: ResolvedPrompts,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    async_: int = DEFAULT_ASYNC,
    session_len: int = DEFAULT_SESSION_LEN,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    enable_prefix_caching: bool = False,
    max_prefill_token_num: int = DEFAULT_MAX_PREFILL_TOKEN_NUM,
    linear_prefix_cache_min_interval: int = DEFAULT_LINEAR_PREFIX_CACHE_MIN_INTERVAL,
    cache_prompt_boundary: bool = False,
    cache_generation_boundary: bool = False,
    cache_boundary_policy: str = DEFAULT_CACHE_BOUNDARY_POLICY,
    debug: bool = False,
) -> SmokeResult:
    _validate_engine_params(
        max_new_tokens=max_new_tokens,
        async_=async_,
        session_len=session_len,
        max_batch_size=max_batch_size,
        max_prefill_token_num=max_prefill_token_num,
        linear_prefix_cache_min_interval=linear_prefix_cache_min_interval,
    )
    _set_hf_cache(cache_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    if debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

    engine_config = TurbomindEngineConfig(
        async_=async_,
        max_batch_size=max_batch_size,
        session_len=session_len,
        cache_max_entry_count=0.5,
        max_prefill_token_num=max_prefill_token_num,
        tp=tp,
        dp=1,
        enable_metrics=False,
        communicator='nccl',
        enable_prefix_caching=enable_prefix_caching,
        linear_prefix_cache_min_interval=linear_prefix_cache_min_interval,
        cache_prompt_boundary=cache_prompt_boundary,
        cache_generation_boundary=cache_generation_boundary,
        cache_boundary_policy=cache_boundary_policy,
    )
    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)

    t0 = time.perf_counter()
    with pipeline(model_id, backend_config=engine_config, log_level='WARNING',
                  trust_remote_code=True) as pipe:
        create_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        out = pipe(resolved.prompts, gen_config=gen_config, do_preprocess=True)
        infer_s = time.perf_counter() - t1

    if not isinstance(out, list):
        out = [out]
    if len(out) != len(resolved.prompts):
        raise RuntimeError(
            f'pipeline returned {len(out)} responses for {len(resolved.prompts)} prompts')

    responses: list[PromptResult] = []
    for res, source_index, prompt_text in zip(out, resolved.source_indices, resolved.prompts):
        text = res.text if hasattr(res, 'text') else str(res)
        batch_index = getattr(res, 'index', len(responses))
        responses.append(PromptResult(
            index=batch_index,
            source_index=source_index,
            prompt_preview=_format_prompt_preview(prompt_text),
            text=text,
            input_token_len=getattr(res, 'input_token_len', -1),
            generate_token_len=getattr(res, 'generate_token_len', -1),
        ))
    return SmokeResult(create_s, infer_s, responses)


def print_report(
    model_id: str,
    tp: int,
    gpus: str,
    resolved: ResolvedPrompts,
    result: SmokeResult,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    async_: int = DEFAULT_ASYNC,
    session_len: int = DEFAULT_SESSION_LEN,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    enable_prefix_caching: bool = False,
    max_prefill_token_num: int = DEFAULT_MAX_PREFILL_TOKEN_NUM,
    linear_prefix_cache_min_interval: int = DEFAULT_LINEAR_PREFIX_CACHE_MIN_INTERVAL,
    cache_prompt_boundary: bool = False,
    cache_generation_boundary: bool = False,
    cache_boundary_policy: str = DEFAULT_CACHE_BOUNDARY_POLICY,
    debug: bool = False,
) -> None:
    print('--- setup ---')
    print(f'model: {model_id}')
    print(f'tp: {tp}')
    print(f'gpus: {gpus}')
    print(f'max_new_tokens: {max_new_tokens}')
    print(f'async: {async_}')
    print(f'session_len: {session_len}')
    print(f'max_batch_size: {max_batch_size}')
    print(f'enable_prefix_caching: {1 if enable_prefix_caching else 0}')
    print(f'linear_prefix_cache_min_interval: {linear_prefix_cache_min_interval}')
    print(f'cache_prompt_boundary: {1 if cache_prompt_boundary else 0}')
    print(f'cache_generation_boundary: {1 if cache_generation_boundary else 0}')
    print(f'cache_boundary_policy: {cache_boundary_policy!r}')
    print(f'max_prefill_token_num: {max_prefill_token_num}')
    print(f'prompt_count: {len(resolved.prompts)}')
    print(f'prompt_source: {resolved.source}')
    if debug:
        print('CUDA_LAUNCH_BLOCKING: 1')
    print()
    print('--- timing ---')
    print(f'pipeline load: {result.create_s:.2f} s')
    print(f'inference: {result.infer_s:.2f} s')
    print()
    print('--- tokens ---')
    for item in result.responses:
        print(f'[{item.index}] source_index: {item.source_index}')
        print(f'[{item.index}] prompt: {item.prompt_preview}')
        print(f'[{item.index}] input: {item.input_token_len}')
        print(f'[{item.index}] generated: {item.generate_token_len}')
    print()
    for item in result.responses:
        if not item.text.strip():
            print(f'warning: empty response text at index {item.index}', file=sys.stderr)
        print(f'--- response {item.index} begin ---')
        print(item.text, end='')
        if item.text and not item.text.endswith('\n'):
            print()
        print(f'--- response {item.index} end ---')


def run_smoke_test(
    *,
    model_id: str,
    cache_dir: str,
    tp: int,
    gpus: str,
    prompts: list[str] | None = None,
    prompt_file: str | None = None,
    prompt_ids: list[int] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    async_: int = DEFAULT_ASYNC,
    session_len: int = DEFAULT_SESSION_LEN,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    enable_prefix_caching: bool = False,
    max_prefill_token_num: int = DEFAULT_MAX_PREFILL_TOKEN_NUM,
    linear_prefix_cache_min_interval: int = DEFAULT_LINEAR_PREFIX_CACHE_MIN_INTERVAL,
    cache_prompt_boundary: bool = False,
    cache_generation_boundary: bool = False,
    cache_boundary_policy: str = DEFAULT_CACHE_BOUNDARY_POLICY,
    debug: bool = False,
    emit_report: bool = True,
) -> SmokeResult:
    resolved = resolve_prompts(
        prompts=prompts,
        prompt_file=prompt_file,
        prompt_ids=prompt_ids,
    )
    result = run_smoke_infer(
        model_id,
        cache_dir,
        tp,
        gpus,
        resolved,
        max_new_tokens=max_new_tokens,
        async_=async_,
        session_len=session_len,
        max_batch_size=max_batch_size,
        enable_prefix_caching=enable_prefix_caching,
        max_prefill_token_num=max_prefill_token_num,
        linear_prefix_cache_min_interval=linear_prefix_cache_min_interval,
        cache_prompt_boundary=cache_prompt_boundary,
        cache_generation_boundary=cache_generation_boundary,
        cache_boundary_policy=cache_boundary_policy,
        debug=debug,
    )
    if emit_report:
        print_report(
            model_id,
            tp,
            gpus,
            resolved,
            result,
            max_new_tokens=max_new_tokens,
            async_=async_,
            session_len=session_len,
            max_batch_size=max_batch_size,
            enable_prefix_caching=enable_prefix_caching,
            max_prefill_token_num=max_prefill_token_num,
            linear_prefix_cache_min_interval=linear_prefix_cache_min_interval,
            cache_prompt_boundary=cache_prompt_boundary,
            cache_generation_boundary=cache_generation_boundary,
            cache_boundary_policy=cache_boundary_policy,
            debug=debug,
        )
    return result


def main() -> None:
    args = parse_args(sys.argv)
    run_smoke_test(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        tp=args.tp,
        gpus=args.gpus,
        prompts=args.prompt,
        prompt_file=args.prompt_file,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
        async_=args.async_,
        session_len=args.session_len,
        max_batch_size=args.max_batch_size,
        enable_prefix_caching=args.enable_prefix_caching,
        max_prefill_token_num=args.max_prefill_token_num,
        linear_prefix_cache_min_interval=args.linear_prefix_cache_min_interval,
        cache_prompt_boundary=args.cache_prompt_boundary,
        cache_generation_boundary=args.cache_generation_boundary,
        cache_boundary_policy=args.cache_boundary_policy,
        debug=args.debug,
        emit_report=True,
    )


if __name__ == '__main__':
    try:
        main()
    except PromptConfigError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
