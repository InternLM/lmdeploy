"""Benchmark OpenAI-compatible /v1/chat/completions endpoints.

This script focuses on eval-style JSONL dumps where each row contains OpenAI
chat ``messages``, or a string/list ``prompt`` (e.g. dapo-math-17k). List-type
``prompt`` values are treated as message lists. It records streaming latency traces,
aggregates TTFT/ITL/TPOT metrics, and writes table plus report artifacts for concurrency/RPS sweeps.

Generation options include ``--output-tokens`` (``max_completion_tokens``),
``--ignore-eos``, ``--return-token-ids``, ``--return-routed-experts``,
``--return-logprob``, and ``--logprobs`` / ``--top-logprobs``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import html
import importlib
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001 - tqdm is optional for CLI progress display.
    tqdm = None


@dataclass
class BenchmarkRequest:
    dataset: str
    id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    input_ids: list[int] | None = None
    image_data: Any = None


@dataclass
class SSEEvent:
    content: str = ''
    reasoning_content: str = ''
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    done: bool = False
    raw: dict[str, Any] | None = None
    routed_experts: str | None = None

    @property
    def token_text(self) -> str:
        return self.content or self.reasoning_content


@dataclass
class RequestTrace:
    dataset: str
    request_id: str
    mode: str
    setting: float
    repeat: int
    success: bool
    start_time: float = 0.0
    first_token_time: float | None = None
    end_time: float = 0.0
    chunk_times: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_available: bool = False
    generated_text: str = ''
    reasoning_text: str = ''
    finish_reason: str | None = None
    http_status: int | None = None
    error: str = ''

    @property
    def ttft_s(self) -> float:
        if self.first_token_time is None:
            return 0.0
        return max(self.first_token_time - self.start_time, 0.0)

    @property
    def e2e_latency_s(self) -> float:
        return max(self.end_time - self.start_time, 0.0)

    @property
    def itls_s(self) -> list[float]:
        return [
            max(self.chunk_times[idx] - self.chunk_times[idx - 1], 0.0)
            for idx in range(1, len(self.chunk_times))
        ]

    @property
    def tpot_s(self) -> float:
        if self.first_token_time is None or self.completion_tokens <= 0:
            return 0.0
        denominator = max(self.completion_tokens - 1, 1)
        return max(self.end_time - self.first_token_time, 0.0) / denominator


SendOne = Callable[[BenchmarkRequest, str, float, int], Awaitable[RequestTrace]]
SHARED_STORE = 'shared_store'
SHARED_STORE_NAMESPACE = 'lmdeploy'
_shared_store_actor: Any | None = None


def init_shared_store() -> Any:
    """Connect to the Ray cluster and return the LMDeploy shared_store
    actor."""
    global _shared_store_actor
    if _shared_store_actor is not None:
        return _shared_store_actor

    import ray

    ray_address = os.environ.get('RAY_ADDRESS', 'auto')
    ctx = ray.init(address=ray_address, namespace=SHARED_STORE_NAMESPACE, ignore_reinit_error=True)
    print(f'ray cluster address: {ctx.address_info["address"]}')
    _shared_store_actor = ray.get_actor(SHARED_STORE, namespace=SHARED_STORE_NAMESPACE)
    return _shared_store_actor


async def fetch_routed_experts(shared_store: Any, key: str) -> Any:
    """Fetch routed_experts from shared_store without blocking the event
    loop."""
    import ray

    ref = shared_store.get.remote(key)
    return await asyncio.to_thread(ray.get, ref)


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(',') if item.strip()]



def _client_connector_limits(
    mode: str,
    levels: Sequence[int],
    num_requests: int,
) -> tuple[int, int]:
    """Return (limit, limit_per_host) for aiohttp.TCPConnector.

    Concurrency mode sizes the pool from the sweep's max concurrency. Request-rate mode may keep all requests in flight
    to the same host.
    """
    if mode == 'concurrency':
        max_concurrency = int(max(levels)) if levels else 100
        return max(max_concurrency * 2, 100), max_concurrency * 2
    max_inflight = max(num_requests, 1)
    return max(max_inflight, 100), max_inflight


def _discover_dataset_files(dataset_dir: Path | None, dataset_files: Sequence[Path] | None) -> list[Path]:
    if dataset_files:
        return sorted(Path(path) for path in dataset_files)
    if dataset_dir is None:
        raise ValueError('Either dataset_dir or dataset_files must be provided.')
    return sorted(Path(dataset_dir).glob('*.jsonl'))


def _dataset_longest_prefix_match(dataset: str, selected: Sequence[str]) -> str | None:
    """Return the element of ``selected`` that matches ``dataset`` with longest
    prefix.

    A candidate ``item`` matches when ``dataset == item`` or ``dataset.startswith(item)``.
    Among matches, the longest ``item`` by character length wins; ties keep the first in
    ``selected`` order.
    """
    best: str | None = None
    best_len = -1
    for item in selected:
        if dataset == item or dataset.lower().startswith(item.lower()):
            n = len(item)
            if n > best_len:
                best = item
                best_len = n
    return best


def _load_tokenizer(model_path: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def _extract_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get('messages') or row.get('message')
    if messages is not None:
        return messages
    prompt = row.get('prompt')
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, str):
        return [{'role': 'user', 'content': prompt}]
    raise ValueError('row must contain messages or prompt')


def _normalize_row(
    row: dict[str, Any],
    dataset: str,
    row_index: int,
    tokenizer=None,
) -> BenchmarkRequest:
    request_id = str(row.get('id', f'{dataset}-{row_index}'))
    messages = _extract_messages(row)

    if tokenizer is not None:
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return BenchmarkRequest(
            dataset=dataset,
            id=request_id,
            input_ids=tokenizer.encode(prompt_str, add_special_tokens=False),
            image_data=row.get('image_data'),
        )

    if not messages:
        raise ValueError(f'row {row_index} in {dataset} has invalid messages')
    return BenchmarkRequest(dataset=dataset, id=request_id, messages=messages)


def _read_raw_rows(
    dataset_dir: str | Path | None = None,
    dataset_files: Sequence[str | Path] | None = None,
    datasets: Sequence[str] | None = None,
    num_prompts: int | None = None,
    shuffle: bool = False,
) -> list[tuple[dict[str, Any], str, int]]:
    """Read JSONL rows without normalization.

    When ``shuffle`` is false and ``num_prompts`` is set, stop reading after enough
    rows are collected so large files are not fully scanned.
    """
    selected = list(dict.fromkeys(datasets or []))
    files = _discover_dataset_files(
        Path(dataset_dir) if dataset_dir is not None else None,
        [Path(path) for path in dataset_files] if dataset_files is not None else None,
    )

    raw_rows: list[tuple[dict[str, Any], str, int]] = []
    dataset = 'all'
    for file_path in files:
        if selected:
            dataset = _dataset_longest_prefix_match(file_path.stem, selected)
            if dataset is None:
                continue
        with file_path.open(encoding='utf-8') as f:
            for row_index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                raw_rows.append((json.loads(line), dataset, row_index))
                if not shuffle and num_prompts is not None and len(raw_rows) >= num_prompts:
                    break
        if not shuffle and num_prompts is not None and len(raw_rows) >= num_prompts:
            break
    return raw_rows


def load_requests(
    dataset_dir: str | Path | None = None,
    dataset_files: Sequence[str | Path] | None = None,
    datasets: Sequence[str] | None = None,
    num_prompts: int | None = None,
    shuffle: bool = False,
    seed: int = 1,
    tokenizer=None,
) -> list[BenchmarkRequest]:
    """Load JSONL chat requests.

    Rows with list-type ``prompt`` (e.g. dapo-math-17k) are treated as message lists,
    matching ``benchmark_generate.py``. When ``tokenizer`` is provided (``--input-ids``),
    rows are converted to ``input_ids`` client-side.
    """
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


def parse_sse_line(line: bytes | str) -> SSEEvent:
    if isinstance(line, bytes):
        line = line.decode('utf-8')
    line = line.strip()
    if not line:
        return SSEEvent()
    if line.startswith('data:'):
        line = line[len('data:'):].strip()
    if line == '[DONE]':
        return SSEEvent(done=True)

    data = json.loads(line)
    choice = (data.get('choices') or [{}])[0]
    delta = choice.get('delta') or {}
    return SSEEvent(
        content=delta.get('content') or '',
        reasoning_content=delta.get('reasoning_content') or '',
        finish_reason=choice.get('finish_reason'),
        usage=data.get('usage'),
        raw=data,
        routed_experts=choice.get('routed_experts'),
    )


def build_payload(
    request: BenchmarkRequest,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_completion_tokens: int | None,
    ignore_eos: bool = False,
    return_token_ids: bool = False,
    return_routed_experts: bool = False,
    return_logprob: bool = False,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'model': model,
        'temperature': temperature,
        'top_p': top_p,
        'stream': True,
        'stream_options': {'include_usage': True},
    }
    if request.input_ids is not None:
        payload['messages'] = []
        payload['input_ids'] = request.input_ids
        payload['do_preprocess'] = False
        if request.image_data is not None:
            payload['image_data'] = request.image_data
    else:
        payload['messages'] = request.messages
    if top_k is not None:
        payload['top_k'] = top_k
    if max_completion_tokens is not None:
        payload['max_completion_tokens'] = max_completion_tokens
    if ignore_eos:
        payload['ignore_eos'] = True
    if return_token_ids:
        payload['return_token_ids'] = True
    if return_routed_experts:
        payload['return_routed_experts'] = True
    if return_logprob:
        payload['return_logprob'] = True
    if logprobs:
        payload['logprobs'] = True
        if top_logprobs is not None:
            payload['top_logprobs'] = top_logprobs
    if extra_body:
        payload.update(extra_body)
    return payload


def _chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip('/')
    if base_url.endswith('/v1'):
        return f'{base_url}/chat/completions'
    return f'{base_url}/v1/chat/completions'


def _models_url(base_url: str) -> str:
    base_url = base_url.rstrip('/')
    if base_url.endswith('/v1'):
        return f'{base_url}/models'
    return f'{base_url}/v1/models'


async def fetch_model_id(session: Any, base_url: str, api_key: str | None = None) -> str:
    headers = {'Authorization': f'Bearer {api_key}'} if api_key else None
    async with session.get(_models_url(base_url), headers=headers) as response:
        if response.status != 200:
            raise RuntimeError(f'Failed to fetch model from /v1/models: {response.status} {response.reason}')
        payload = await response.json()
    model_list = payload.get('data') or []
    if not model_list or not model_list[0].get('id'):
        raise RuntimeError('No model id found in /v1/models response.')
    return str(model_list[0]['id'])


async def request_chat_completion(
    session: Any,
    request: BenchmarkRequest,
    url: str,
    model: str,
    mode: str,
    setting: float,
    repeat: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_completion_tokens: int | None,
    ignore_eos: bool,
    return_token_ids: bool,
    return_routed_experts: bool,
    return_logprob: bool,
    logprobs: bool,
    top_logprobs: int | None,
    extra_body: dict[str, Any] | None,
    headers: dict[str, str] | None = None,
    save_response_text: bool = False,
    shared_store: Any | None = None,
) -> RequestTrace:
    payload = build_payload(
        request=request,
        model=model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_completion_tokens=max_completion_tokens,
        ignore_eos=ignore_eos,
        return_token_ids=return_token_ids,
        return_routed_experts=return_routed_experts,
        return_logprob=return_logprob,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
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
                    event = parse_sse_line(raw_line)
                    if event.done:
                        continue
                    now = time.perf_counter()
                    if event.token_text:
                        if trace.first_token_time is None:
                            trace.first_token_time = now
                        trace.chunk_times.append(now)
                    if save_response_text and event.content:
                        trace.generated_text += event.content
                    if save_response_text and event.reasoning_content:
                        trace.reasoning_text += event.reasoning_content
                    if event.finish_reason is not None:
                        trace.finish_reason = event.finish_reason
                    if event.usage:
                        trace.usage_available = True
                        trace.prompt_tokens = int(event.usage.get('prompt_tokens', trace.prompt_tokens) or 0)
                        trace.completion_tokens = int(
                            event.usage.get('completion_tokens', trace.completion_tokens) or 0
                        )
                    if event.routed_experts and shared_store is not None:
                        try:
                            await fetch_routed_experts(shared_store, event.routed_experts)
                        except Exception as e:  # noqa: BLE001 - record and keep consuming SSE.
                            trace.error = repr(e)

            trace.end_time = time.perf_counter()
            trace.success = trace.error == ''
            return trace
    except Exception as e:  # noqa: BLE001 - benchmark should record failures and continue.
        trace.end_time = time.perf_counter()
        trace.error = repr(e)
        return trace


async def closed_loop_runner(
    requests: Sequence[BenchmarkRequest],
    concurrency: int,
    repeat: int,
    send_one: SendOne,
    on_done: Callable[[RequestTrace], None] | None = None,
) -> list[RequestTrace]:
    queue: asyncio.Queue[BenchmarkRequest] = asyncio.Queue()
    for request in requests:
        queue.put_nowait(request)

    traces: list[RequestTrace] = []

    async def worker():
        while True:
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                trace = await send_one(request, 'concurrency', concurrency, repeat)
                traces.append(trace)
                if on_done is not None:
                    on_done(trace)
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max(concurrency, 1))]
    await asyncio.gather(*workers)
    return traces


async def request_rate_runner(
    requests: Sequence[BenchmarkRequest],
    request_rate: int,
    repeat: int,
    send_one: SendOne,
    seed: int = 1,
    on_done: Callable[[RequestTrace], None] | None = None,
) -> list[RequestTrace]:
    rng = random.Random(seed)
    tasks: list[asyncio.Task[RequestTrace]] = []
    for request in requests:
        tasks.append(asyncio.create_task(send_one(request, 'request-rate', request_rate, repeat)))
        await asyncio.sleep(rng.expovariate(request_rate))
    traces: list[RequestTrace] = []
    for task in asyncio.as_completed(tasks):
        trace = await task
        traces.append(trace)
        if on_done is not None:
            on_done(trace)
    return traces


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _latency_stats(prefix: str, values_s: Sequence[float]) -> dict[str, float]:
    values_ms = [value * 1000 for value in values_s]
    if not values_ms:
        return {
            f'ave_{prefix}_ms': 0.0,
            f'min_{prefix}_ms': 0.0,
            f'max_{prefix}_ms': 0.0,
            f'p50_{prefix}_ms': 0.0,
            f'p75_{prefix}_ms': 0.0,
            f'p95_{prefix}_ms': 0.0,
            f'p99_{prefix}_ms': 0.0,
        }
    return {
        f'ave_{prefix}_ms': _mean(values_ms),
        f'min_{prefix}_ms': min(values_ms),
        f'max_{prefix}_ms': max(values_ms),
        f'p50_{prefix}_ms': percentile(values_ms, 50),
        f'p75_{prefix}_ms': percentile(values_ms, 75),
        f'p95_{prefix}_ms': percentile(values_ms, 95),
        f'p99_{prefix}_ms': percentile(values_ms, 99),
    }


def token_distribution_stats(values: Sequence[int]) -> list[tuple[str, float]]:
    if not values:
        return []
    float_values = [float(value) for value in values]
    return [
        ('mean', _mean(float_values)),
        ('p50', percentile(float_values, 50)),
        ('p75', percentile(float_values, 75)),
        ('p90', percentile(float_values, 90)),
        ('p99', percentile(float_values, 99)),
    ]


TOKEN_STAT_COLORS = {
    'mean': 'tab:red',
    'p50': 'tab:orange',
    'p75': 'tab:green',
    'p90': 'tab:blue',
    'p99': 'tab:purple',
}


def _group_key(trace: RequestTrace) -> tuple[str, str, float, int]:
    return (trace.dataset, trace.mode, trace.setting, trace.repeat)


def aggregate_traces(traces: Sequence[RequestTrace]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, float, int], list[RequestTrace]] = defaultdict(list)
    for trace in traces:
        grouped[_group_key(trace)].append(trace)

    for (dataset, mode, setting, repeat), group in sorted(grouped.items()):
        completed = [trace for trace in group if trace.success]
        failed = len(group) - len(completed)
        start = min((trace.start_time for trace in group), default=0.0)
        end = max((trace.end_time for trace in group), default=start)
        duration = max(end - start, 0.0)
        total_input = sum(trace.prompt_tokens for trace in completed)
        total_output = sum(trace.completion_tokens for trace in completed)
        itls = [itl for trace in completed for itl in trace.itls_s]

        summary: dict[str, Any] = {
            'dataset': dataset,
            'mode': mode,
            'setting': setting,
            'repeat': repeat,
            'total': len(group),
            'completed': len(completed),
            'failed': failed,
            'success_rate': len(completed) / len(group) if group else 0.0,
            'duration_s': duration,
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'request_throughput_req_s': len(completed) / duration if duration > 0 else 0.0,
            'input_throughput_tok_s': total_input / duration if duration > 0 else 0.0,
            'output_throughput_tok_s': total_output / duration if duration > 0 else 0.0,
        }
        summary.update(_latency_stats('ttft', [trace.ttft_s for trace in completed if trace.first_token_time]))
        summary.update(_latency_stats('itl', itls))
        summary['usage_available'] = all(trace.usage_available for trace in completed) if completed else False
        summary.update(_latency_stats(
            'tpot',
            [trace.tpot_s for trace in completed if trace.first_token_time and trace.completion_tokens > 0],
        ))
        summary.update(_latency_stats('e2e_latency', [trace.e2e_latency_s for trace in completed]))
        summaries.append(summary)
    return summaries


def _trace_to_json(trace: RequestTrace) -> dict[str, Any]:
    item = asdict(trace)
    item['ttft_s'] = trace.ttft_s
    item['itls_s'] = trace.itls_s
    item['tpot_s'] = trace.tpot_s
    item['e2e_latency_s'] = trace.e2e_latency_s
    return item


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _write_requests_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        'dataset',
        'request_id',
        'mode',
        'setting',
        'repeat',
        'success',
        'http_status',
        'ttft_s',
        'tpot_s',
        'e2e_latency_s',
        'prompt_tokens',
        'completion_tokens',
        'usage_available',
        'finish_reason',
        'error',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for summary in summaries:
        for key in summary:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


_MATPLOTLIB_WARNED = False


def _matplotlib_pyplot():
    global _MATPLOTLIB_WARNED
    try:
        matplotlib = importlib.import_module('matplotlib')
        matplotlib.use('Agg')
        return importlib.import_module('matplotlib.pyplot')
    except Exception as e:
        if not _MATPLOTLIB_WARNED:
            print(
                'Warning: matplotlib is not available; skipping PNG plots. '
                f'Install it with: pip install matplotlib ({e!r})',
            )
            _MATPLOTLIB_WARNED = True
        return None


def _mode_axis_label(mode: str) -> str:
    if mode == 'concurrency':
        return 'Concurrency'
    if mode == 'request-rate':
        return 'Request rate (req/s)'
    return 'Benchmark setting'


def _plot_metric(
    output_dir: Path,
    summaries: Sequence[dict[str, Any]],
    metric: str,
    title: str,
    mode: str,
) -> Path | None:
    plt = _matplotlib_pyplot()
    if plt is None:
        return None

    by_series: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        if metric in summary:
            by_series[(str(summary['dataset']), str(summary['mode']))].append(summary)

    if not by_series:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    for (dataset, mode), rows in sorted(by_series.items()):
        rows = sorted(rows, key=lambda item: float(item['setting']))
        ax.plot(
            [float(item['setting']) for item in rows],
            [float(item.get(metric, 0.0)) for item in rows],
            marker='o',
            label=f'{dataset} ({mode})',
        )
    ax.set_title(title)
    ax.set_xlabel(_mode_axis_label(mode))
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')
    fig.tight_layout()
    path = output_dir / f'{metric}.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_latency_stats(
    output_dir: Path,
    summaries: Sequence[dict[str, Any]],
    prefix: str,
    title: str,
    mode: str,
) -> Path | None:
    plt = _matplotlib_pyplot()
    if plt is None:
        return None

    stats = [
        (f'ave_{prefix}_ms', 'ave'),
        (f'p50_{prefix}_ms', 'p50'),
        (f'p75_{prefix}_ms', 'p75'),
        (f'p95_{prefix}_ms', 'p95'),
        (f'p99_{prefix}_ms', 'p99'),
    ]
    by_series: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        if any(key in summary for key, _ in stats):
            by_series[(str(summary['dataset']), str(summary['mode']))].append(summary)

    if not by_series:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (dataset, mode), rows in sorted(by_series.items()):
        rows = sorted(rows, key=lambda item: float(item['setting']))
        x_values = [float(item['setting']) for item in rows]
        for key, stat_label in stats:
            ax.plot(
                x_values,
                [float(item.get(key, 0.0)) for item in rows],
                marker='o',
                label=f'{dataset} ({mode}) {stat_label}',
            )
    ax.set_title(title)
    ax.set_xlabel(_mode_axis_label(mode))
    ax.set_ylabel('Latency (ms)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='x-small', ncol=2)
    fig.tight_layout()
    path = output_dir / f'{prefix}_latency_stats.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_token_histogram(output_dir: Path,
                          traces: Sequence[RequestTrace],
                          token_field: str, title: str) -> Path | None:
    plt = _matplotlib_pyplot()
    if plt is None:
        return None

    values = [int(getattr(trace, token_field)) for trace in traces if trace.success and getattr(trace, token_field) > 0]
    if not values:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(max(int(math.sqrt(len(values))), 10), 80)
    ax.hist(values, bins=bins, alpha=0.85)
    for label, value in token_distribution_stats(values):
        ax.axvline(
            value,
            color=TOKEN_STAT_COLORS[label],
            linestyle='--',
            linewidth=1.4,
            label=f'{label}: {value:.1f}',
        )
    ax.set_title(title)
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Request count')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')
    fig.tight_layout()
    path = output_dir / f'{token_field}_histogram.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_html_report(path: Path, summaries: Sequence[dict[str, Any]], plot_paths: Sequence[Path], mode: str) -> None:
    headers = list(summaries[0].keys()) if summaries else []
    table_rows = []
    for summary in summaries:
        cells = ''.join(f"<td>{html.escape(str(summary.get(header, '')))}</td>" for header in headers)
        table_rows.append(f'<tr>{cells}</tr>')
    header_html = ''.join(f'<th>{html.escape(header)}</th>' for header in headers)
    plots_html = '\n'.join(
        f'<figure><img src="{html.escape(str(path.relative_to(path.parent.parent)))}" alt="{html.escape(path.stem)}">'
        f'<figcaption>{html.escape(path.stem)}</figcaption></figure>'
        for path in plot_paths
    )
    path.write_text(
        f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Chat Completions Benchmark Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 0.35rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; }}
    .chart {{ width: 100%; height: 440px; margin-bottom: 2rem; }}
  </style>
</head>
<body>
  <h1>Chat Completions Benchmark Report</h1>
  <p>Mode: {html.escape(mode)}. X-axis: {html.escape(_mode_axis_label(mode))}.</p>
  <p>PNG plots are generated from benchmark summary and request token statistics.</p>
  <h2>Plots</h2>
  {plots_html}
  <h2>Summary</h2>
  <table><thead><tr>{header_html}</tr></thead><tbody>{''.join(table_rows)}</tbody></table>
</body>
</html>
''',
        encoding='utf-8',
    )


def write_report_artifacts(
    output_dir: str | Path,
    traces: Sequence[RequestTrace],
    summaries: Sequence[dict[str, Any]],
    mode: str,
    save_raw_requests: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    if save_raw_requests:
        trace_rows = [_trace_to_json(trace) for trace in traces]
        _write_jsonl(output_dir / 'requests.jsonl', trace_rows)
        _write_requests_csv(output_dir / 'requests.csv', trace_rows)
        (output_dir / 'requests.json').write_text(
            json.dumps(trace_rows, indent=2, ensure_ascii=False), encoding='utf-8')
    _write_summary_csv(output_dir / 'summary.csv', summaries)
    (output_dir / 'summary.json').write_text(
        json.dumps(list(summaries), indent=2, ensure_ascii=False), encoding='utf-8')

    plot_paths = [
        path
        for path in [
            _plot_latency_stats(plots_dir, summaries, 'ttft', 'TTFT latency', mode),
            _plot_latency_stats(plots_dir, summaries, 'itl', 'ITL latency', mode),
            _plot_latency_stats(plots_dir, summaries, 'tpot', 'TPOT latency', mode),
            _plot_metric(plots_dir, summaries, 'input_throughput_tok_s', 'Input token throughput', mode),
            _plot_metric(plots_dir, summaries, 'output_throughput_tok_s', 'Output token throughput', mode),
            _plot_metric(plots_dir, summaries, 'success_rate', 'Success rate', mode),
            _plot_token_histogram(plots_dir, traces, 'prompt_tokens', 'Input token histogram'),
            _plot_token_histogram(plots_dir, traces, 'completion_tokens', 'Output token histogram'),
        ]
        if path is not None
    ]
    _write_html_report(output_dir / 'report.html', summaries, plot_paths, mode)


async def _run_warmup(
    requests: Sequence[BenchmarkRequest],
    warmup_requests: int,
    send_one: SendOne,
) -> None:
    if warmup_requests <= 0:
        return
    for request in list(requests)[:warmup_requests]:
        await send_one(request, 'warmup', 0.0, -1)


async def run_benchmark(args: argparse.Namespace) -> tuple[list[RequestTrace], list[dict[str, Any]]]:
    try:
        aiohttp = importlib.import_module('aiohttp')
    except ImportError as e:
        raise RuntimeError('aiohttp is required for live chat-completions benchmarking.') from e

    dataset_files = [Path(path) for path in args.dataset_files] if args.dataset_files else None
    url = _chat_completions_url(args.base_url)
    headers = {}
    if args.api_key:
        headers['Authorization'] = f'Bearer {args.api_key}'
    extra_body = json.loads(args.extra_request_body) if args.extra_request_body else {}

    shared_store = None
    if args.return_routed_experts:
        shared_store = init_shared_store()

    tokenizer = None
    if args.input_ids:
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
        model = await fetch_model_id(session, args.base_url, args.api_key)
        print(f'Using model from /v1/models: {model}')
        if args.input_ids:
            print('input_ids=True (client-side apply_chat_template, do_preprocess=False)')
        if args.max_completion_tokens is not None:
            print(f'max_completion_tokens={args.max_completion_tokens}')
        if args.ignore_eos:
            print('ignore_eos=True')
        if args.return_token_ids:
            print('return_token_ids=True')
        if args.return_routed_experts:
            print('return_routed_experts=True')
        if args.return_logprob:
            print('return_logprob=True')
        if args.logprobs:
            print(f'logprobs=True top_logprobs={args.top_logprobs!r}')

        async def send_one(request: BenchmarkRequest, mode: str, setting: float, repeat: int) -> RequestTrace:
            return await request_chat_completion(
                session=session,
                request=request,
                url=url,
                model=model,
                mode=mode,
                setting=setting,
                repeat=repeat,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_completion_tokens=args.max_completion_tokens,
                ignore_eos=args.ignore_eos,
                return_token_ids=args.return_token_ids,
                return_routed_experts=args.return_routed_experts,
                return_logprob=args.return_logprob,
                logprobs=args.logprobs,
                top_logprobs=args.top_logprobs,
                extra_body=extra_body,
                headers=headers,
                save_response_text=args.save_response_text,
                shared_store=shared_store,
            )

        all_traces: list[RequestTrace] = []
        await _run_warmup(requests, args.warmup_requests, send_one)
        for repeat in range(args.repeats):
            if args.mode == 'concurrency':
                for concurrency in args.levels:
                    print(f'benchmark with {len(requests)} for case concurrency-{concurrency}...')
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
                    print(f'write report for case concurrency-{concurrency}...')
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
    parser = argparse.ArgumentParser(description='Benchmark /v1/chat/completions with eval JSONL datasets.')
    parser.add_argument(
        '--base-url',
        default='http://127.0.0.1:23333/v1',
        help='OpenAI-compatible API base URL. Requests go to /v1/chat/completions.',
    )
    parser.add_argument('--api-key', default='', help='Bearer token used for /v1/models and chat requests.')
    parser.add_argument(
        '--input-ids',
        action='store_true',
        help='Pre-tokenize prompts client-side (apply_chat_template) and send input_ids with do_preprocess=false, '
        'matching benchmark_generate.py and POST /generate.',
    )
    parser.add_argument(
        '--model-path',
        default='',
        help='Tokenizer/model path for --input-ids. Defaults to the id from /v1/models.',
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Pass trust_remote_code=True when loading the tokenizer for --input-ids.',
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('./workspace/oc_data'),
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
        help='Comma-separated dataset names or filename-stem prefixes, e.g. "bbeh" matches bbeh*.jsonl.',
    )
    parser.add_argument('--num-prompts', type=int, help='Maximum number of sampled prompts from dataset.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle each dataset before applying --num-prompts.')
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
        default=[1, 16, 32, 64, 128, 256, 512],
        help='Space-separated sweep values. Interpreted as concurrency levels or request rates based on --mode.',
    )
    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat each dataset/level run.')
    parser.add_argument(
        '--warmup-requests',
        type=int,
        default=1,
        help='Number of unmeasured warmup requests to send before each dataset run.',
    )
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature shared by all requests.')
    parser.add_argument('--top-p', type=float, default=1.0, help='Sampling top_p shared by all requests.')
    parser.add_argument('--top-k', type=int, default=40, help='Sampling top_k shared by all requests.')
    parser.add_argument(
        '--output-tokens',
        '--max-completion-tokens',
        type=int,
        dest='max_completion_tokens',
        metavar='N',
        help='Cap generation length via max_completion_tokens (LMDeploy /v1/chat/completions). '
        'If omitted, generation stops at EOS or server default.',
    )
    parser.add_argument(
        '--ignore-eos',
        action='store_true',
        help='Set ignore_eos=true so the model does not stop at end-of-sequence tokens. '
        'Use with --output-tokens for fixed-length decode benchmarks.',
    )
    parser.add_argument(
        '--return-token-ids',
        action='store_true',
        help='Set return_token_ids=true to include output token ids in stream chunks (LMDeploy extension).',
    )
    parser.add_argument(
        '--return-routed-experts',
        action='store_true',
        help='Set return_routed_experts=true to include MoE routed expert indices (LMDeploy extension).',
    )
    parser.add_argument(
        '--return-logprob',
        action='store_true',
        help='Set return_logprob=true to include raw (logprob, token_id) pairs without OpenAI token formatting.',
    )
    parser.add_argument(
        '--logprobs',
        action='store_true',
        help='Set logprobs=true to return OpenAI-compatible per-token logprobs.',
    )
    parser.add_argument(
        '--top-logprobs',
        type=int,
        default=None,
        metavar='N',
        help='When --logprobs is set, request top_logprobs=N (default: server uses 1).',
    )
    parser.add_argument(
        '--extra-request-body',
        default='',
        help='JSON object merged into every chat request body for engine-specific options.',
    )
    parser.add_argument(
        '--save-raw-requests',
        action='store_true',
        help='Save per-request raw traces as requests.jsonl, requests.csv, and requests.json.',
    )
    parser.add_argument(
        '--save-response-text',
        action='store_true',
        help='Retain generated and reasoning text in memory and raw traces. Disabled by default to reduce memory use.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('benchmark_outputs') / f"chat_completions_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help='Directory for summary CSV/JSON, PNG plots, HTML report, and optional raw request traces.',
    )
    return parser.parse_args()


def main() -> None:
    traces, summaries = asyncio.run(run_benchmark(parse_args()))
    completed = sum(summary['completed'] for summary in summaries)
    failed = sum(summary['failed'] for summary in summaries)
    print(f'Recorded {len(traces)} requests: {completed} completed, {failed} failed.')


if __name__ == '__main__':
    main()
