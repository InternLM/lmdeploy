# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import csv
import json
import os
import random
import time
from collections import OrderedDict
from typing import List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

from lmdeploy import (GenerationConfig, PytorchEngineConfig,
                      TurbomindEngineConfig, pipeline)
from lmdeploy.cli.utils import ArgumentHelper, DefaultsAndTypesHelpFormatter


def sample_requests(dataset_path: str, num_requests: int,
                    tokenizer) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data['conversations']) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data['conversations'][0]['value'],
                data['conversations'][1]['value']) for data in dataset]

    # pre-sample to avoid go through all the dataset
    dataset = random.sample(dataset, max(int(num_requests * 1.2), 1000))

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


class Engine:

    def __init__(self, model_path: str, engine_config, csv: str):
        self.pipe = pipeline(model_path, backend_config=engine_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True)

        self.csv = csv

    def process_request(self, requests, concurrency, temperature, top_p, top_k,
                        stream_output):

        stats = OrderedDict(
            (session_id, None) for session_id in range(len(requests)))
        prompts = [prompt for prompt, _, _ in requests]
        gen_configs = [
            GenerationConfig(temperature=temperature,
                             top_p=top_p,
                             top_k=top_k,
                             ignore_eos=True,
                             max_new_tokens=output_len)
            for _, _, output_len in requests
        ]

        start = time.perf_counter()
        if stream_output:
            pbar = tqdm(total=len(requests))
            for output in self.pipe.stream_infer(prompts,
                                                 gen_configs,
                                                 do_preprocess=False):
                session_id = output.session_id
                n_token = output.generate_token_len
                finish_reason = output.finish_reason
                stats[session_id] = (n_token, finish_reason)
                if finish_reason is not None:
                    pbar.update(1)
        else:
            for output in self.pipe(prompts,
                                    gen_configs,
                                    do_preprocess=False,
                                    use_tqdm=True):
                session_id = output.session_id
                n_token = output.generate_token_len
                finish_reason = output.finish_reason
                stats[session_id] = (n_token, finish_reason)

        elapsed_time = time.perf_counter() - start

        completion_tokens = 0
        for session_id, (n_token, finish_reason) in stats.items():
            assert finish_reason == 'length', \
                f'unexpected finish_reason of session_id={session_id}, ' \
                f'prompt={requests[session_id][0]}'
            assert n_token - 1 <= requests[session_id][-1] <= n_token, \
                f'request to generate {requests[session_id][-1]} tokens, ' \
                f'but got {n_token} tokens'
            completion_tokens += n_token

        prompt_tokens = 0
        for _, input_len, _ in requests:
            prompt_tokens += input_len

        completion_token_throughput = completion_tokens / elapsed_time
        total_token_throughput = (prompt_tokens +
                                  completion_tokens) / elapsed_time
        rps = len(requests) / elapsed_time
        rpm = rps * 60

        print(f'\n{"-" * 50}\nconcurrency: {concurrency}\n'
              f'elapsed_time: {elapsed_time:.3f}s\n')

        print(
            f'number of prompts: {len(requests)}\n'
            f'number of prompt tokens: {prompt_tokens:.0f}\n'
            f'number of completion tokens: {completion_tokens:.0f}\n'
            f'token throughput (completion token): {completion_token_throughput:.3f} token/s\n'  # noqa
            f'token throughput (prompt + completion token): {total_token_throughput:.3f} token/s\n'  # noqa
            f'RPS (request per second): {rps:.3f} req/s\n'
            f'RPM (request per minute): {rpm:.3f} req/min\n'
            f'{"-" * 50}\n')

        if self.csv:
            with open(self.csv, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'batch', 'num_promts', 'RPS', 'RPM',
                    'throughput(out tok/s)', 'throughput(total tok/s)'
                ])
                writer.writerow([
                    concurrency,
                    len(requests), f'{rps:.3f}', f'{rpm:.3f}',
                    f'{completion_token_throughput:.3f}',
                    f'{total_token_throughput:.3f}'
                ])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark the request throughput of lmdeploy '
        'in localhost',
        formatter_class=DefaultsAndTypesHelpFormatter)
    parser.add_argument('dataset', type=str, help='the path dataset')
    parser.add_argument('model_path',
                        type=str,
                        help='the path of the model in localhost or '
                        'the repo_id of the model in huggingface.co')
    parser.add_argument(
        '-c',
        '--concurrency',
        type=int,
        help='Number of working threads to process the sampled prompts',
        default=256)
    parser.add_argument('-n',
                        '--num-prompts',
                        type=int,
                        help='Number of prompts to process',
                        default=5000)
    parser.add_argument('--csv',
                        type=str,
                        help='Where to save the result.',
                        default='./profile_pipeline_api.csv')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Seed used in sampling prompts from dataset')
    parser.add_argument('--stream-output',
                        action='store_true',
                        help='Trust remote code for loading hf models')
    # other args
    ArgumentHelper.top_p(parser)
    ArgumentHelper.temperature(parser)
    ArgumentHelper.top_k(parser)
    ArgumentHelper.log_level(parser)
    ArgumentHelper.backend(parser)

    # pytorch engine args
    pt_group = parser.add_argument_group('PyTorch engine arguments')
    tp_act = ArgumentHelper.tp(pt_group)
    session_len_act = ArgumentHelper.session_len(pt_group, default=4096)
    cache_count_act = ArgumentHelper.cache_max_entry_count(pt_group)
    cache_block_seq_len_act = ArgumentHelper.cache_block_seq_len(pt_group)
    prefix_caching_act = ArgumentHelper.enable_prefix_caching(pt_group)

    # turbomind engine args
    tb_group = parser.add_argument_group('TurboMind engine argument')
    tb_group._group_actions.append(tp_act)
    tb_group._group_actions.append(session_len_act)
    tb_group._group_actions.append(cache_count_act)
    tb_group._group_actions.append(cache_block_seq_len_act)
    tb_group._group_actions.append(prefix_caching_act)
    ArgumentHelper.model_format(tb_group, default='hf')
    ArgumentHelper.quant_policy(tb_group, default=0)
    ArgumentHelper.num_tokens_per_iter(tb_group)
    ArgumentHelper.max_prefill_iters(tb_group)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    os.environ['TM_LOG_LEVEL'] = args.log_level
    if args.backend == 'turbomind':
        engine_config = TurbomindEngineConfig(
            session_len=args.session_len,
            max_batch_size=args.concurrency,
            tp=args.tp,
            cache_max_entry_count=args.cache_max_entry_count,
            cache_block_seq_len=args.cache_block_seq_len,
            model_format=args.model_format,
            quant_policy=args.quant_policy,
            num_tokens_per_iter=args.num_tokens_per_iter,
            max_prefill_iters=args.max_prefill_iters,
            enable_prefix_caching=args.enable_prefix_caching,
        )
    elif args.backend == 'pytorch':
        engine_config = PytorchEngineConfig(
            session_len=args.session_len,
            cache_max_entry_count=args.cache_max_entry_count,
            block_size=args.cache_block_seq_len,
            max_batch_size=args.concurrency,
            tp=args.tp,
            thread_safe=False,
            enable_prefix_caching=args.enable_prefix_caching,
        )

    engine = Engine(args.model_path, engine_config, csv=args.csv)

    requests = sample_requests(args.dataset, args.num_prompts,
                               engine.tokenizer)

    engine.process_request(requests,
                           temperature=args.temperature,
                           top_p=args.top_p,
                           top_k=args.top_k,
                           concurrency=args.concurrency,
                           stream_output=args.stream_output)


if __name__ == '__main__':
    main()
