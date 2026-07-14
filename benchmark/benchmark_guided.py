# Copyright (c) OpenMMLab. All rights reserved.
"""Benchmark guided decoding (response_format) overhead vs. baseline.

Philosophy of fair comparison
------------------------------
Guided decoding fundamentally changes the output length distribution: the
grammar may cause early termination (no valid continuation) or force longer
output (schema requires more fields).  Therefore, naive throughput (tok/s,
req/s) is **not** a fair comparison.

The **primary** metric is **per-token latency** (TPOT / ITL), which is
independent of output length and directly reflects the grammar bitmask
overhead.  The comparison table also reports actual output lengths and
computes a "per-token overhead %" so the cost of guided decoding is
immediately obvious.

Two run modes are supported:

* **Default** (``ignore_eos=False``): Both baseline and guided runs stop
  naturally.  This is the production-realistic mode.  TPOT is the fair
  metric; throughput numbers are shown for reference but are length-biased.
* **``--ignore-eos``**: Both runs force ``max_new_tokens`` output.  This
  isolates the pure per-step overhead of the grammar bitmask.  Note that
  guided decoding may still terminate early if the grammar has no valid
  continuation (xgrammar returns an empty token set).

Usage examples:

  # JSON schema (default schema), compare with baseline
  python3 benchmark/benchmark_guided.py \\
      ShareGPT_V3_unfiltered_cleaned_split.json \\
      Qwen/Qwen2.5-7B-Instruct \\
      --response-format json_schema \\
      --concurrency 64

  # Custom JSON schema file
  python3 benchmark/benchmark_guided.py \\
      ShareGPT_V3_unfiltered_cleaned_split.json \\
      Qwen/Qwen2.5-7B-Instruct \\
      --response-format json_schema \\
      --json-schema-path my_schema.json

  # Regex schema
  python3 benchmark/benchmark_guided.py \\
      ShareGPT_V3_unfiltered_cleaned_split.json \\
      Qwen/Qwen2.5-7B-Instruct \\
      --response-format regex_schema \\
      --regex-schema '[A-Z][a-z]+ lives in [A-Z][a-z]+\\.'

  # JSON object
  python3 benchmark/benchmark_guided.py \\
      ShareGPT_V3_unfiltered_cleaned_split.json \\
      Qwen/Qwen2.5-7B-Instruct \\
      --response-format json_object

  # Force full output length (ignore eos) to isolate pure bitmask overhead
  python3 benchmark/benchmark_guided.py \\
      ShareGPT_V3_unfiltered_cleaned_split.json \\
      Qwen/Qwen2.5-7B-Instruct \\
      --response-format json_schema \\
      --ignore-eos --concurrency 64

  # Skip baseline, only run guided
  python3 benchmark/benchmark_guided.py \\
      ShareGPT_V3_unfiltered_cleaned_split.json \\
      Qwen/Qwen2.5-7B-Instruct \\
      --response-format json_schema --no-baseline
"""
import argparse
import csv as _csv
import json
import os
import random
import time

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from lmdeploy.cli.utils import ArgumentHelper, DefaultsAndTypesHelpFormatter, get_speculative_config
from lmdeploy.messages import SpeculativeConfig
from lmdeploy.profiler import Profiler, Session
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

# A generic JSON schema that most LLMs can satisfy.
DEFAULT_JSON_SCHEMA = {
    'type': 'object',
    'properties': {
        'answer': {
            'type': 'string'
        },
        'confidence': {
            'type': 'number'
        },
    },
    'required': ['answer'],
}

DEFAULT_REGEX_SCHEMA = r'[A-Z][a-z]+ (is|are|was|were) [a-z]+\.'


def build_response_format(fmt_type: str | None,
                          json_schema: dict | None = None,
                          regex_schema: str | None = None) -> dict | None:
    if fmt_type is None:
        return None
    if fmt_type == 'json_schema':
        schema = json_schema or DEFAULT_JSON_SCHEMA
        return {'type': 'json_schema', 'json_schema': {'name': 'benchmark', 'schema': schema}}
    if fmt_type == 'json_object':
        return {'type': 'json_object'}
    if fmt_type == 'regex_schema':
        return {'type': 'regex_schema', 'regex_schema': regex_schema or DEFAULT_REGEX_SCHEMA}
    raise ValueError(f'Unsupported response_format type: {fmt_type}')


# --------------- data sampling ---------------


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: int | None = None,
) -> list[tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError('output_len too small')
    with open(dataset_path) as f:
        dataset = json.load(f)
    dataset = [data for data in dataset if len(data['conversations']) >= 2]
    dataset = [(data['conversations'][0]['value'], data['conversations'][1]['value']) for data in dataset]
    random.shuffle(dataset)

    filtered: list[tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered) == num_requests:
            break
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            continue
        if prompt_len > 1024 or (prompt_len + output_len > 2048 and fixed_output_len is None):
            continue
        filtered.append((prompt, prompt_len, output_len))

    print(f'#Input tokens: {np.sum([x[1] for x in filtered])}')
    print(f'#Output tokens (requested): {np.sum([x[2] for x in filtered])}')
    return filtered


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
) -> list[tuple[str, int, int]]:
    input_lens = np.random.randint(max(int(input_len * range_ratio), 1), input_len + 1, size=num_prompts)
    output_lens = np.random.randint(int(output_len * range_ratio), output_len + 1, size=num_prompts)

    with open(dataset_path) as f:
        dataset = json.load(f)
    dataset = [data for data in dataset if len(data['conversations']) >= 2]
    dataset = [(data['conversations'][0]['value'], data['conversations'][1]['value']) for data in dataset]
    dataset = [(q, a) for q, a in dataset if len(q) > 0]
    random.shuffle(dataset)

    requests: list[tuple[str, int, int]] = []
    for i in range(num_prompts):
        prompt = dataset[i % len(dataset)][0]
        prompt_token_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_token_ids)
        if prompt_len > input_lens[i]:
            input_ids = prompt_token_ids[:input_lens[i]]
        else:
            ratio = (input_lens[i] + prompt_len - 1) // prompt_len
            input_ids = (prompt_token_ids * ratio)[:input_lens[i]]
        prompt = tokenizer.decode(input_ids)
        requests.append((prompt, int(input_lens[i]), int(output_lens[i])))

    print(f'#Input tokens: {np.sum(input_lens)}')
    print(f'#Output tokens (requested): {np.sum(output_lens)}')
    return requests


# --------------- engine wrapper ---------------


class Engine:

    def __init__(self, model_path: str, engine_config,
                 speculative_config: SpeculativeConfig | None = None):
        self.pipe = pipeline(model_path,
                             backend_config=engine_config,
                             log_level='ERROR',
                             speculative_config=speculative_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.return_routed_experts = getattr(self.pipe.backend_config, 'enable_return_routed_experts', False)

    def process_request(self, requests, profiler: Profiler, temperature, top_p, top_k,
                        stream_output, ignore_eos, response_format=None):

        prompts = [prompt for prompt, _, _ in requests]
        gen_configs = [
            GenerationConfig(temperature=temperature,
                             top_p=top_p,
                             top_k=top_k,
                             ignore_eos=ignore_eos,
                             do_sample=False,
                             return_routed_experts=self.return_routed_experts,
                             response_format=response_format,
                             max_new_tokens=output_len) for _, _, output_len in requests
        ]

        sess: list[Session] = []
        for _, input_len, output_len in requests:
            sess.append(profiler.new_session(input_len, output_len))

        def _to_status(finish_reason):
            if finish_reason in ('length', 'stop'):
                return Session.SUCCESS
            return Session.FAIL

        profiler.start()
        for s in sess:
            s.tick(0)

        if stream_output:
            pbar = tqdm(total=len(requests))
            for output in self.pipe.stream_infer(prompts, gen_config=gen_configs, do_preprocess=False):
                idx = output.index
                n_token = output.generate_token_len
                finish_reason = output.finish_reason
                sess[idx].tick(n_token)
                if finish_reason is not None:
                    sess[idx].finish(_to_status(finish_reason))
                    pbar.update(1)
            pbar.close()
        else:
            for output in self.pipe(prompts, gen_configs, do_preprocess=False, use_tqdm=True):
                idx = output.index
                n_token = output.generate_token_len
                finish_reason = output.finish_reason
                sess[idx].tick(n_token)
                sess[idx].finish(_to_status(finish_reason))

        profiler.finish()

        # Collect actual per-request output lengths for detailed comparison
        actual_output_lens = []
        for i, s in enumerate(sess):
            actual_output_lens.append(s.ns[-1] if s.status == Session.SUCCESS else 0)
            if s.status != Session.SUCCESS:
                logger.warning(f'Request {i}: {s.ns[-1]}/{s.req_output_len} tokens, finish != length/stop')
        return actual_output_lens


# --------------- metrics extraction ---------------


def extract_metrics(profiler: Profiler, actual_output_lens: list[int], num_requests: int) -> dict:
    return {
        'elapsed': profiler.elapsed_time,
        'total_input': profiler.total_input,
        'total_output': profiler.total_output,
        'success': profiler.success,
        'avg_output_len': float(np.mean(actual_output_lens)) if actual_output_lens else 0.0,
        'median_output_len': float(np.median(actual_output_lens)) if actual_output_lens else 0.0,
        'output_throughput': profiler.output_throughput,
        'input_throughput': profiler.input_throughput,
        'rps': profiler.rps,
        'e2e_mean': profiler.e2e_mean,
        'e2e_p50': profiler.e2e_stat[0],
        'e2e_p99': profiler.e2e_stat[-1],
        'tpot_mean': profiler.tpot_mean,
        'tpot_p99': profiler.tpot_stat[-1],
        'ttft_mean': getattr(profiler, 'ttft_mean', float('inf')),
        'ttft_p99': getattr(profiler, 'ttft_stat', (float('inf'),))[-1],
        'itl_mean': getattr(profiler, 'itls_mean', float('inf')),
        'itl_p99': getattr(profiler, 'itls_stat', (float('inf'),))[-1],
    }


def print_comparison(baseline: dict | None, guided: dict, label: str,
                     stream_output: bool, ignore_eos: bool):
    """Print a side-by-side comparison with emphasis on per-token metrics."""
    col_w = 36
    num_w = 14

    def fmt(x, unit=''):
        if not stream_output and unit in ('ms',) and x == float('inf'):
            return '-'
        if x == float('inf'):
            return '-'
        if unit == 'ms':
            return f'{x * 1000:.2f}'
        if unit in ('tok/s', 'req/s'):
            return f'{x:.2f}'
        if unit in ('', 'raw'):
            if isinstance(x, float):
                return f'{x:.2f}'
            return str(x)
        return f'{x:.2f}'

    def diff_pct(b, g):
        if b is None or b == 0 or b == float('inf') or g == float('inf'):
            return '-'
        pct = (g - b) / abs(b) * 100
        sign = '+' if pct >= 0 else ''
        return f'{sign}{pct:.1f}%'

    has_base = baseline is not None
    width = col_w + num_w * (3 if has_base else 2)

    print()
    print('=' * width)
    title = f' Guided Decoding Benchmark: {label} '
    print(f'{title:=^{width}}')

    mode = 'ignore_eos=True (forced length)' if ignore_eos else 'ignore_eos=False (natural stop)'
    print(f' Mode: {mode}')
    print('-' * width)

    header = f'{"Metric":<{col_w}}'
    if has_base:
        header += f'{"Baseline":>{num_w}}{"Guided":>{num_w}}{"Diff":>{num_w}}'
    else:
        header += f'{"Guided":>{num_w}}'
    print(header)
    print('-' * width)

    # Section: output length context (critical for interpreting everything else)
    print(f'{"--- Output Length Context ---":<{width}}')
    length_rows = [
        ('Successful requests', 'success', ''),
        ('Avg output tokens / request', 'avg_output_len', 'raw'),
        ('Median output tokens / request', 'median_output_len', 'raw'),
        ('Total output tokens', 'total_output', ''),
    ]
    for name, key, unit in length_rows:
        g_val = guided[key]
        line = f'{name:<{col_w}}'
        if has_base:
            b_val = baseline[key]
            line += f'{fmt(b_val, unit):>{num_w}}{fmt(g_val, unit):>{num_w}}'
            line += f'{diff_pct(b_val, g_val):>{num_w}}'
        else:
            line += f'{fmt(g_val, unit):>{num_w}}'
        print(line)

    # Section: per-token latency (THE fair metric)
    print(f'{"--- Per-Token Latency (fair metric) ---":<{width}}')
    latency_rows = [
        ('TPOT mean (ms)', 'tpot_mean', 'ms'),
        ('TPOT P99 (ms)', 'tpot_p99', 'ms'),
    ]
    if stream_output:
        latency_rows += [
            ('ITL mean (ms)', 'itl_mean', 'ms'),
            ('ITL P99 (ms)', 'itl_p99', 'ms'),
        ]
    for name, key, unit in latency_rows:
        g_val = guided[key]
        line = f'{name:<{col_w}}'
        if has_base:
            b_val = baseline[key]
            line += f'{fmt(b_val, unit):>{num_w}}{fmt(g_val, unit):>{num_w}}'
            line += f'{diff_pct(b_val, g_val):>{num_w}}'
        else:
            line += f'{fmt(g_val, unit):>{num_w}}'
        print(line)

    # Highlight per-token overhead
    if has_base and baseline['tpot_mean'] not in (0, float('inf')) and guided['tpot_mean'] != float('inf'):
        overhead = (guided['tpot_mean'] - baseline['tpot_mean']) / baseline['tpot_mean'] * 100
        print(f'{"** Per-token overhead (TPOT)":<{col_w}}{f"{overhead:+.1f}%":>{num_w}}')

    # Section: end-to-end & TTFT (length-dependent, for reference only)
    print(f'{"--- E2E / TTFT (length-dependent, for reference) ---":<{width}}')
    e2e_rows = [
        ('E2E latency mean (ms)', 'e2e_mean', 'ms'),
        ('E2E latency P99 (ms)', 'e2e_p99', 'ms'),
    ]
    if stream_output:
        e2e_rows += [
            ('TTFT mean (ms)', 'ttft_mean', 'ms'),
            ('TTFT P99 (ms)', 'ttft_p99', 'ms'),
        ]
    for name, key, unit in e2e_rows:
        g_val = guided[key]
        line = f'{name:<{col_w}}'
        if has_base:
            b_val = baseline[key]
            line += f'{fmt(b_val, unit):>{num_w}}{fmt(g_val, unit):>{num_w}}'
            line += f'{diff_pct(b_val, g_val):>{num_w}}'
        else:
            line += f'{fmt(g_val, unit):>{num_w}}'
        print(line)

    # Section: throughput (length-biased, for reference only)
    print(f'{"--- Throughput (length-biased, for reference) ---":<{width}}')
    tp_rows = [
        ('Output throughput (tok/s)', 'output_throughput', 'tok/s'),
        ('Input throughput (tok/s)', 'input_throughput', 'tok/s'),
        ('Request throughput (req/s)', 'rps', 'req/s'),
    ]
    for name, key, unit in tp_rows:
        g_val = guided[key]
        line = f'{name:<{col_w}}'
        if has_base:
            b_val = baseline[key]
            line += f'{fmt(b_val, unit):>{num_w}}{fmt(g_val, unit):>{num_w}}'
            line += f'{diff_pct(b_val, g_val):>{num_w}}'
        else:
            line += f'{fmt(g_val, unit):>{num_w}}'
        print(line)

    print('=' * width)


# --------------- main ---------------


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark guided decoding (response_format) overhead vs. baseline',
                                     formatter_class=DefaultsAndTypesHelpFormatter)
    parser.add_argument('dataset', type=str, help='Path to the ShareGPT dataset')
    parser.add_argument('model_path',
                        type=str,
                        help='Path of the model in localhost or repo_id on huggingface.co')
    parser.add_argument('-c', '--concurrency', type=int, help='Max batch size', default=256)
    parser.add_argument('-n', '--num-prompts', type=int, help='Number of prompts', default=1000)
    parser.add_argument('--csv', type=str, help='Save results to CSV', default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stream-output', action='store_true', help='Use streaming output')
    parser.add_argument('--dataset-name', type=str, default='sharegpt', choices=['sharegpt', 'random'])
    parser.add_argument('--sharegpt-output-len', type=int, default=None)
    parser.add_argument('--random-input-len', type=int, default=None)
    parser.add_argument('--random-output-len', type=int, default=None)
    parser.add_argument('--random-range-ratio', type=float, default=0.0)

    # guided decoding
    guided = parser.add_argument_group('Guided decoding arguments')
    guided.add_argument(
        '--response-format',
        type=str,
        required=True,
        choices=['json_schema', 'json_object', 'regex_schema'],
        help='Type of response_format (required).')
    guided.add_argument(
        '--json-schema-path',
        type=str,
        default=None,
        help='Path to a JSON schema file. Uses a built-in default if omitted.')
    guided.add_argument(
        '--regex-schema',
        type=str,
        default=None,
        help='Regex pattern. Uses a built-in default if omitted.')
    guided.add_argument(
        '--ignore-eos',
        action='store_true',
        help='Force max_new_tokens output (ignore EOS). '
        'Isolates pure per-step grammar bitmask overhead. '
        'Without this flag, both runs stop naturally (production mode).')
    guided.add_argument(
        '--no-baseline',
        action='store_true',
        help='Skip the baseline run; only benchmark guided decoding.')

    # engine / sampling
    ArgumentHelper.top_p(parser)
    ArgumentHelper.temperature(parser)
    ArgumentHelper.top_k(parser)
    ArgumentHelper.log_level(parser)
    ArgumentHelper.backend(parser)

    pt_group = parser.add_argument_group('PyTorch engine arguments')
    ArgumentHelper.eager_mode(pt_group)
    ArgumentHelper.enable_return_routed_experts(pt_group)
    tp_act = ArgumentHelper.tp(pt_group)
    cache_count_act = ArgumentHelper.cache_max_entry_count(pt_group)
    session_len_act = ArgumentHelper.session_len(pt_group)
    cache_block_seq_len_act = ArgumentHelper.cache_block_seq_len(pt_group)
    prefix_caching_act = ArgumentHelper.enable_prefix_caching(pt_group)

    ArgumentHelper.add_spec_group(parser)

    tb_group = parser.add_argument_group('TurboMind engine argument')
    tb_group._group_actions.append(tp_act)
    tb_group._group_actions.append(cache_count_act)
    tb_group._group_actions.append(session_len_act)
    tb_group._group_actions.append(cache_block_seq_len_act)
    tb_group._group_actions.append(prefix_caching_act)
    ArgumentHelper.model_format(tb_group, default='hf')
    ArgumentHelper.quant_policy(tb_group, default=0)
    ArgumentHelper.num_tokens_per_iter(tb_group)
    ArgumentHelper.max_prefill_iters(tb_group)
    ArgumentHelper.communicator(tb_group)
    ArgumentHelper.async_(tb_group)

    return parser.parse_args()


def run_once(engine: Engine, requests, temperature, top_p, top_k,
             stream_output, ignore_eos, response_format=None) -> tuple[Profiler, list[int]]:
    """Run one pass.

    Returns (profiler, actual_output_lens).
    """
    profiler = Profiler(stream_output, [50, 75, 95, 99])
    actual_output_lens = engine.process_request(
        requests, profiler, temperature, top_p, top_k,
        stream_output, ignore_eos, response_format=response_format)
    profiler.compute_metrics()
    return profiler, actual_output_lens


def main():
    args = parse_args()
    random.seed(args.seed)
    os.environ['TM_LOG_LEVEL'] = args.log_level

    json_schema = None
    if args.json_schema_path:
        with open(args.json_schema_path) as f:
            json_schema = json.load(f)
    response_format = build_response_format(args.response_format,
                                            json_schema=json_schema,
                                            regex_schema=args.regex_schema)
    print(f'response_format: {json.dumps(response_format, indent=2)}')
    print(f'ignore_eos: {args.ignore_eos}')

    if args.backend == 'turbomind':
        engine_config = TurbomindEngineConfig(
            max_batch_size=args.concurrency,
            tp=args.tp,
            cache_max_entry_count=args.cache_max_entry_count,
            session_len=args.session_len,
            cache_block_seq_len=args.cache_block_seq_len,
            model_format=args.model_format,
            quant_policy=args.quant_policy,
            num_tokens_per_iter=args.num_tokens_per_iter,
            max_prefill_iters=args.max_prefill_iters,
            enable_prefix_caching=args.enable_prefix_caching,
            communicator=args.communicator,
            enable_metrics=False,
            async_=args.async_)
    elif args.backend == 'pytorch':
        engine_config = PytorchEngineConfig(
            cache_max_entry_count=args.cache_max_entry_count,
            session_len=args.session_len,
            block_size=args.cache_block_seq_len,
            max_batch_size=args.concurrency,
            tp=args.tp,
            thread_safe=False,
            eager_mode=args.eager_mode,
            enable_prefix_caching=args.enable_prefix_caching,
            enable_return_routed_experts=args.enable_return_routed_experts,
        )

    speculative_config = get_speculative_config(args)
    engine = Engine(args.model_path, engine_config, speculative_config=speculative_config)

    if args.dataset_name == 'sharegpt':
        assert args.random_input_len is None and args.random_output_len is None
        requests = sample_sharegpt_requests(args.dataset, args.num_prompts,
                                            engine.tokenizer, args.sharegpt_output_len)
    elif args.dataset_name == 'random':
        assert args.random_input_len is not None and args.random_output_len is not None
        requests = sample_random_requests(args.random_input_len, args.random_output_len,
                                          args.num_prompts, args.random_range_ratio,
                                          engine.tokenizer, args.dataset)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset_name}')

    # ---- baseline run ----
    baseline_metrics = None
    if not args.no_baseline:
        print('\n' + '=' * 60)
        print(' Running BASELINE (no response_format) ...')
        print('=' * 60)
        t0 = time.perf_counter()
        baseline_profiler, baseline_out_lens = run_once(
            engine, requests, args.temperature, args.top_p, args.top_k,
            args.stream_output, args.ignore_eos, response_format=None)
        baseline_metrics = extract_metrics(baseline_profiler, baseline_out_lens, len(requests))
        print(f'Baseline done in {time.perf_counter() - t0:.1f}s')
        baseline_profiler.summarize(title='Baseline (no response_format)',
                                    hyperparams=[('Concurrency', args.concurrency)])

    # ---- guided run ----
    print('\n' + '=' * 60)
    print(f' Running GUIDED ({args.response_format}) ...')
    print('=' * 60)
    t0 = time.perf_counter()
    guided_profiler, guided_out_lens = run_once(
        engine, requests, args.temperature, args.top_p, args.top_k,
        args.stream_output, args.ignore_eos, response_format=response_format)
    guided_metrics = extract_metrics(guided_profiler, guided_out_lens, len(requests))
    print(f'Guided done in {time.perf_counter() - t0:.1f}s')
    guided_profiler.summarize(title=f'Guided ({args.response_format})',
                              hyperparams=[('Concurrency', args.concurrency)])

    # ---- comparison ----
    print_comparison(baseline_metrics, guided_metrics,
                     label=f'{args.response_format} @ bs={args.concurrency}',
                     stream_output=args.stream_output,
                     ignore_eos=args.ignore_eos)

    # ---- CSV ----
    if args.csv:
        for tag, metrics in [('baseline', baseline_metrics), ('guided', guided_metrics)]:
            if metrics is None:
                continue
            row = {
                'run': tag,
                'response_format': args.response_format,
                'ignore_eos': args.ignore_eos,
                'backend': args.backend,
                'bs': args.concurrency,
                'num_prompts': args.num_prompts,
                'dataset_name': args.dataset_name,
                **metrics,
            }
            file_exists = os.path.isfile(args.csv)
            with open(args.csv, 'a') as f:
                writer = _csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        print(f'Results appended to {args.csv}')

    engine.pipe.close()


if __name__ == '__main__':
    main()
