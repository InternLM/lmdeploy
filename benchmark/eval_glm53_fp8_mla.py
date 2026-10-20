# Copyright (c) OpenMMLab. All rights reserved.
"""Run vLLM's 5-shot GSM8K prompts/scorer against an LMDeploy completion API.

Run in a vLLM client environment. Start each cache policy with TP8 and MTP5, keeping the other server settings
identical. The model field is included because LMDeploy requires it on completion requests.
"""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import aiohttp


async def evaluate(args, gsm):
    prompts, labels = gsm._build_gsm8k_prompts(args.num_questions, 5)
    states, output_tokens = [''] * len(prompts), [0] * len(prompts)
    semaphore = asyncio.Semaphore(args.concurrency)
    failures = []
    prompt_path = args.output_dir / 'prompts.jsonl'
    with prompt_path.open('w') as stream:
        for index, (prompt, label) in enumerate(zip(prompts, labels)):
            stream.write(json.dumps(dict(index=index, prompt=prompt, label=label)) + '\n')
    started = time.perf_counter()
    with (args.output_dir / 'responses.jsonl').open('w') as stream:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            async def one(index):
                payload = dict(model=args.model, prompt=prompts[index], temperature=0, max_tokens=256,
                               stop=['Question', 'Assistant:', '<|separator|>'], seed=42)
                async with semaphore:
                    try:
                        async with session.post(f'{args.base_url.rstrip("/")}/v1/completions',
                                                json=payload) as response:
                            response.raise_for_status()
                            raw = await response.json()
                        states[index] = raw['choices'][0]['text']
                        output_tokens[index] = raw['usage']['completion_tokens']
                        record = dict(index=index, response=raw, label=labels[index],
                                      predicted=gsm.get_answer_value(states[index]))
                    except Exception as error:
                        failures.append(index)
                        record = dict(index=index, error=repr(error))
                    stream.write(json.dumps(record) + '\n')
                    stream.flush()
                    if (index + 1) % 100 == 0:
                        print(f'completed through {index + 1}/{len(prompts)}', flush=True)
            await asyncio.gather(*(one(i) for i in range(len(prompts))))
    result = gsm._score_gsm8k(states, output_tokens, labels, 5, 256, time.perf_counter() - started)
    result['failed_requests'] = sorted(failures)
    result['scorer_sha256'] = hashlib.sha256(args.scorer.read_bytes()).hexdigest()
    result['prompts_sha256'] = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} requests failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:23333')
    parser.add_argument('--model', required=True)
    parser.add_argument('--vllm-checkout', type=Path, required=True,
                        help='Checkout containing tests/evals/gsm8k/gsm8k_eval.py')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num-questions', type=int, default=1319)
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--data-dir', type=Path,
                        help='Optional directory containing gsm8k_train.jsonl and gsm8k_test.jsonl')
    args = parser.parse_args()
    if not 1 <= args.num_questions <= 1319 or args.concurrency < 1:
        parser.error('num-questions must be in [1, 1319] and concurrency must be positive')
    args.scorer = args.vllm_checkout / 'tests/evals/gsm8k/gsm8k_eval.py'
    spec = importlib.util.spec_from_file_location('vllm_gsm8k_eval', args.scorer)
    gsm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gsm)
    if args.data_dir:
        data = {
            split: [json.loads(line) for line in (args.data_dir / f'gsm8k_{split}.jsonl').read_text().splitlines()]
            for split in ('train', 'test')
        }
        gsm.load_gsm8k_data = lambda: (data['train'], data['test'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    asyncio.run(evaluate(args, gsm))


if __name__ == '__main__':
    main()
