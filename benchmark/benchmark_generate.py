"""Benchmark LMDeploy ``POST /generate`` with eval-style JSONL datasets.

Each JSONL row provides chat ``messages`` or ``prompt`` (a string or message list).
Rows are converted with ``apply_chat_template`` using
``--model-path`` (or the model id from ``/v1/models``). The script records streaming
latency (TTFT / ITL / TPOT), aggregates metrics, and writes summary tables plus
optional PNG plots.

Example::

    python benchmark/benchmark_generate.py \\
        --dataset-dir ./workspace/oc_data/ \\
        --datasets aime2025 \\
        --base-url http://127.0.0.1:23333 \\
        --output-tokens 100 \\
        --ignore-eos \\
        --levels 8 32 64 \\
        --output-dir build/benchmark_generate_outputs/
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from benchmark_chat_completion import (  # noqa: E402
    RequestTrace,
    _client_connector_limits,
    _extract_messages,
    _load_tokenizer,
    _read_raw_rows,
    _run_warmup,
    _split_csv,
    aggregate_traces,
    closed_loop_runner,
    fetch_model_id,
    request_rate_runner,
    write_report_artifacts,
)

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


@dataclass
class GenerateBenchmarkRequest:
    dataset: str
    id: str
    input_ids: list[int]
    image_data: Any = None


@dataclass
class GenerateStreamEvent:
    text: str = ''
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    done: bool = False

    @property
    def token_text(self) -> str:
        return self.text


SendOne = Callable[[GenerateBenchmarkRequest, str, float, int], Awaitable[RequestTrace]]


def _normalize_row(
    row: dict[str, Any],
    dataset: str,
    row_index: int,
    tokenizer,
) -> GenerateBenchmarkRequest:
    request_id = str(row.get('id', f'{dataset}-{row_index}'))
    messages = _extract_messages(row)
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return GenerateBenchmarkRequest(
        dataset=dataset,
        id=request_id,
        input_ids=tokenizer.encode(prompt_str, add_special_tokens=False),
        image_data=row.get('image_data'),
    )


def load_requests(
    dataset_dir: str | Path | None = None,
    dataset_files: Sequence[str | Path] | None = None,
    datasets: Sequence[str] | None = None,
    num_prompts: int | None = None,
    shuffle: bool = False,
    seed: int = 1,
    tokenizer=None,
) -> list[GenerateBenchmarkRequest]:
    import random

    raw_rows = _read_raw_rows(
        dataset_dir=dataset_dir,
        dataset_files=dataset_files,
        datasets=datasets,
        num_prompts=num_prompts,
        shuffle=shuffle,
    )
    if shuffle:
        random.Random(seed).shuffle(raw_rows)
    if num_prompts is not None:
        raw_rows = raw_rows[:num_prompts]
    if not raw_rows:
        raise ValueError('No benchmark requests were loaded.')

    return [_normalize_row(row, dataset, row_index, tokenizer) for row, dataset, row_index in raw_rows]


def parse_generate_sse_line(line: bytes | str) -> GenerateStreamEvent:
    if isinstance(line, bytes):
        line = line.decode('utf-8')
    line = line.strip()
    if not line:
        return GenerateStreamEvent()
    if line.startswith('data:'):
        line = line[len('data:'):].strip()
    if line == '[DONE]':
        return GenerateStreamEvent(done=True)

    data = json.loads(line)
    meta = data.get('meta_info') or {}
    finish_reason = None
    finish_obj = meta.get('finish_reason')
    if isinstance(finish_obj, dict):
        finish_reason = finish_obj.get('type')
    elif isinstance(finish_obj, str):
        finish_reason = finish_obj

    prompt_tokens = int(meta.get('prompt_tokens') or 0)
    completion_tokens = int(meta.get('completion_tokens') or 0)
    return GenerateStreamEvent(
        text=data.get('text') or '',
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def build_payload(
    request: GenerateBenchmarkRequest,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int | None,
    ignore_eos: bool,
    return_logprob: bool,
    return_routed_experts: bool,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'stream': True,
    }
    payload['input_ids'] = request.input_ids
    if request.image_data is not None:
        payload['image_data'] = request.image_data
    if max_tokens is not None:
        payload['max_tokens'] = max_tokens
    if ignore_eos:
        payload['ignore_eos'] = True
    if return_logprob:
        payload['return_logprob'] = True
    if return_routed_experts:
        payload['return_routed_experts'] = True
    if extra_body:
        payload.update(extra_body)
    return payload


def _generate_url(base_url: str) -> str:
    base_url = base_url.rstrip('/')
    if base_url.endswith('/v1'):
        base_url = base_url[:-3]
    return f'{base_url}/generate'


async def request_generate(
    session: Any,
    request: GenerateBenchmarkRequest,
    url: str,
    mode: str,
    setting: float,
    repeat: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int | None,
    ignore_eos: bool,
    return_logprob: bool,
    return_routed_experts: bool,
    extra_body: dict[str, Any] | None,
    headers: dict[str, str] | None = None,
    save_response_text: bool = False,
) -> RequestTrace:
    payload = build_payload(
        request=request,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
        return_logprob=return_logprob,
        return_routed_experts=return_routed_experts,
        extra_body=extra_body,
    )
    trace = RequestTrace(
        dataset=request.dataset,
        request_id=request.id,
        mode=mode,
        setting=setting,
        repeat=repeat,
        success=False,
        start_time=time.perf_counter(),
    )
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            trace.http_status = response.status
            if response.status != 200:
                trace.error = f'{response.status} {response.reason}: {await response.text()}'
                trace.end_time = time.perf_counter()
                return trace

            async for chunk in response.content:
                for raw_line in chunk.splitlines():
                    if not raw_line.strip():
                        continue
                    event = parse_generate_sse_line(raw_line)
                    if event.done:
                        continue
                    now = time.perf_counter()
                    if event.token_text:
                        if trace.first_token_time is None:
                            trace.first_token_time = now
                        trace.chunk_times.append(now)
                    if save_response_text and event.text:
                        trace.generated_text += event.text
                    if event.finish_reason is not None:
                        trace.finish_reason = event.finish_reason
                    if event.prompt_tokens or event.completion_tokens:
                        trace.usage_available = True
                        if event.prompt_tokens:
                            trace.prompt_tokens = event.prompt_tokens
                        if event.completion_tokens:
                            trace.completion_tokens = event.completion_tokens
            trace.end_time = time.perf_counter()
            trace.success = trace.error == ''
            return trace
    except Exception as e:  # noqa: BLE001
        trace.end_time = time.perf_counter()
        trace.error = repr(e)
        return trace


async def run_benchmark(args: argparse.Namespace) -> tuple[list[RequestTrace], list[dict[str, Any]]]:
    try:
        aiohttp = importlib.import_module('aiohttp')
    except ImportError as e:
        raise RuntimeError('aiohttp is required for live /generate benchmarking.') from e

    dataset_files = [Path(path) for path in args.dataset_files] if args.dataset_files else None
    url = _generate_url(args.base_url)
    headers = {}
    if args.api_key:
        headers['Authorization'] = f'Bearer {args.api_key}'
    extra_body = json.loads(args.extra_request_body) if args.extra_request_body else {}

    print(f'POST {url}')
    if args.max_tokens is not None:
        print(f'max_tokens={args.max_tokens}')
    if args.ignore_eos:
        print('ignore_eos=True')
    if args.return_logprob:
        print('return_logprob=True')
    if args.return_routed_experts:
        print('return_routed_experts=True')

    model_path = args.model_path
    if not model_path:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as probe_session:
            model_path = await fetch_model_id(probe_session, args.base_url, args.api_key or None)
    print(f'Using tokenizer from: {model_path}')
    tokenizer = _load_tokenizer(model_path, trust_remote_code=args.trust_remote_code)
    requests = load_requests(
        dataset_dir=args.dataset_dir,
        dataset_files=dataset_files,
        datasets=_split_csv(args.datasets),
        num_prompts=args.num_prompts,
        shuffle=args.shuffle,
        seed=args.seed,
        tokenizer=tokenizer,
    )
    pool_limit, pool_limit_per_host = _client_connector_limits(args.mode, args.levels, len(requests))
    connector = aiohttp.TCPConnector(limit=pool_limit, limit_per_host=pool_limit_per_host)
    print(f'aiohttp connection pool: limit={pool_limit}, limit_per_host={pool_limit_per_host}')

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=None),
    ) as session:

        async def send_one(
            request: GenerateBenchmarkRequest,
            mode: str,
            setting: float,
            repeat: int,
        ) -> RequestTrace:
            return await request_generate(
                session=session,
                request=request,
                url=url,
                mode=mode,
                setting=setting,
                repeat=repeat,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                ignore_eos=args.ignore_eos,
                return_logprob=args.return_logprob,
                return_routed_experts=args.return_routed_experts,
                extra_body=extra_body,
                headers=headers,
                save_response_text=args.save_response_text,
            )

        all_traces: list[RequestTrace] = []
        summaries: list[dict[str, Any]] = []
        await _run_warmup(requests, args.warmup_requests, send_one)
        for repeat in range(args.repeats):
            if args.mode == 'concurrency':
                for concurrency in args.levels:
                    print(f'benchmark {len(requests)} requests, concurrency={concurrency}...')
                    completed_count = 0
                    failed_count = 0
                    pbar = None
                    if tqdm is not None:
                        pbar = tqdm(
                            total=len(requests),
                            desc=f'repeat-{repeat} concurrency-{int(concurrency)}',
                            unit='req',
                            dynamic_ncols=True,
                        )

                    def on_done(trace: RequestTrace) -> None:
                        nonlocal completed_count, failed_count
                        if trace.success:
                            completed_count += 1
                        else:
                            failed_count += 1
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix(completed=completed_count, failed=failed_count)

                    all_traces.extend(
                        await closed_loop_runner(
                            requests,
                            concurrency=int(concurrency),
                            repeat=repeat,
                            send_one=send_one,
                            on_done=on_done,
                        )
                    )
                    if pbar is not None:
                        pbar.close()
                    print(f'write report for concurrency-{concurrency}...')
                    summaries = aggregate_traces(all_traces)
                    write_report_artifacts(
                        args.output_dir,
                        all_traces,
                        summaries,
                        mode=args.mode,
                        save_raw_requests=args.save_raw_requests,
                    )

            elif args.mode == 'request-rate':
                for request_rate in args.levels:
                    print(f'benchmark {len(requests)} requests, request_rate={request_rate}...')
                    completed_count = 0
                    failed_count = 0
                    pbar = None
                    if tqdm is not None:
                        pbar = tqdm(
                            total=len(requests),
                            desc=f'repeat-{repeat} request-rate-{request_rate}',
                            unit='req',
                            dynamic_ncols=True,
                        )

                    def on_done(trace: RequestTrace) -> None:
                        nonlocal completed_count, failed_count
                        if trace.success:
                            completed_count += 1
                        else:
                            failed_count += 1
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix(completed=completed_count, failed=failed_count)

                    all_traces.extend(
                        await request_rate_runner(
                            requests,
                            request_rate=request_rate,
                            repeat=repeat,
                            send_one=send_one,
                            seed=args.seed + repeat,
                            on_done=on_done,
                        )
                    )
                    if pbar is not None:
                        pbar.close()
                    summaries = aggregate_traces(all_traces)
                    write_report_artifacts(
                        args.output_dir,
                        all_traces,
                        summaries,
                        mode=args.mode,
                        save_raw_requests=args.save_raw_requests,
                    )

    return all_traces, summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark LMDeploy POST /generate with JSONL datasets.')
    parser.add_argument(
        '--base-url',
        default='http://127.0.0.1:23333',
        help='API server base URL. Requests go to /generate.',
    )
    parser.add_argument('--api-key', default='', help='Optional Bearer token for authenticated servers.')
    parser.add_argument(
        '--model-path',
        default='',
        help='Tokenizer/model path for apply_chat_template. Defaults to the id from /v1/models.',
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Pass trust_remote_code=True when loading the tokenizer.',
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=None,
        help='Directory containing eval JSONL files. Each file stem is used as the dataset name.',
    )
    parser.add_argument(
        '--dataset-files',
        type=Path,
        nargs='*',
        help='Explicit JSONL files to benchmark. Overrides dataset discovery from --dataset-dir.',
    )
    parser.add_argument(
        '--datasets',
        help='Comma-separated dataset names or filename-stem prefixes.',
    )
    parser.add_argument('--num-prompts', type=int, help='Maximum number of prompts sampled per dataset.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle requests before applying --num-prompts.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for shuffling and request-rate scheduling.')
    parser.add_argument(
        '--mode',
        choices=['concurrency', 'request-rate'],
        default='concurrency',
        help='Benchmark mode: closed-loop concurrency or open-loop request rate.',
    )
    parser.add_argument(
        '--levels',
        nargs='+',
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help='Space-separated sweep values (concurrency levels or request rates).',
    )
    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat each level run.')
    parser.add_argument(
        '--warmup-requests',
        type=int,
        default=1,
        help='Number of unmeasured warmup requests before measured runs.',
    )
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature.')
    parser.add_argument('--top-p', type=float, default=1.0, help='Sampling top_p.')
    parser.add_argument('--top-k', type=int, default=40, help='Sampling top_k.')
    parser.add_argument(
        '--output-tokens',
        '--max-tokens',
        type=int,
        dest='max_tokens',
        metavar='N',
        help='Set max_tokens in the /generate request. If omitted, server default (128) applies.',
    )
    parser.add_argument(
        '--ignore-eos',
        action='store_true',
        help='Set ignore_eos=true. Use with --output-tokens for fixed-length decode benchmarks.',
    )
    parser.add_argument(
        '--return-logprob',
        action='store_true',
        help='Set return_logprob=true (requires logprobs_mode in server config).',
    )
    parser.add_argument(
        '--return-routed-experts',
        action='store_true',
        help='Set return_routed_experts=true for MoE routed expert indices.',
    )
    parser.add_argument(
        '--extra-request-body',
        default='',
        help='JSON object merged into every /generate request body.',
    )
    parser.add_argument(
        '--save-raw-requests',
        action='store_true',
        help='Save per-request traces as requests.jsonl, requests.csv, and requests.json.',
    )
    parser.add_argument(
        '--save-response-text',
        action='store_true',
        help='Retain generated text in memory and raw traces.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('benchmark_outputs') / f"generate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help='Directory for summary CSV/JSON, PNG plots, HTML report, and optional raw traces.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_dir is None and not args.dataset_files:
        raise SystemExit('Provide --dataset-dir or --dataset-files.')
    traces, summaries = asyncio.run(run_benchmark(args))
    completed = sum(summary['completed'] for summary in summaries)
    failed = sum(summary['failed'] for summary in summaries)
    print(f'Recorded {len(traces)} requests: {completed} completed, {failed} failed.')


if __name__ == '__main__':
    main()
