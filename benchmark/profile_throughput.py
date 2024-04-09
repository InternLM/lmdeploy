# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import csv
import json
import os
import random
import time
from queue import Queue
from threading import Thread
from typing import List, Tuple, Union

import numpy as np
from tqdm import tqdm

from lmdeploy.cli.utils import ArgumentHelper, DefaultsAndTypesHelpFormatter
from lmdeploy.messages import (EngineGenerationConfig, PytorchEngineConfig,
                               TurbomindEngineConfig)
from lmdeploy.pytorch.engine.engine import EngineInstance
from lmdeploy.tokenizer import DetokenizeState, Tokenizer


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: Tokenizer,
) -> List[Tuple[str, int, int]]:
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

    def __init__(self, model_path: str,
                 engine_config: Union[PytorchEngineConfig,
                                      TurbomindEngineConfig], csv: str):
        if isinstance(engine_config, TurbomindEngineConfig):
            from lmdeploy.turbomind import TurboMind
            tm_model = TurboMind.from_pretrained(model_path,
                                                 engine_config=engine_config)
        elif isinstance(engine_config, PytorchEngineConfig):
            from lmdeploy.pytorch.engine import Engine as PytorchEngine
            tm_model = PytorchEngine(model_path, engine_config=engine_config)

        self.tm_model = tm_model
        self.tokenizer = tm_model.tokenizer

        self.csv = csv
        self.pbar = None

    def _inference(self, req_queue: Queue, res_queue: Queue, session_id: int,
                   temperature: float, top_p: float, top_k: int,
                   stream_output: bool):
        model_inst = self.tm_model.create_instance()
        stats = []
        # get each generated token's latency
        per_token_latency_stats = []
        for prompt, input_seqlen, output_seqlen in iter(
                req_queue.get, [None, None, None]):
            _per_token_latency_stats = [0] * (output_seqlen + 1)
            state = DetokenizeState()
            prev = time.perf_counter()
            n_prev_token = 0

            input_ids = self.tokenizer(prompt).input_ids

            for outputs in model_inst.stream_infer(
                    session_id,
                    input_ids=input_ids,
                    gen_config=EngineGenerationConfig(
                        max_new_tokens=output_seqlen,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        ignore_eos=True),
                    sequence_start=True,
                    sequence_end=True,
                    stream_output=stream_output):
                _, res, n_token = outputs
                _, state = self.tokenizer.detokenize_incrementally(res, state)
                now = time.perf_counter()
                if n_prev_token != n_token:
                    _per_token_latency_stats[n_prev_token] = np.round(
                        now - prev, 3)
                    n_prev_token = n_token
                prev = now
            # for pytorch engine to restart a session
            if isinstance(model_inst, EngineInstance):
                model_inst.end(session_id)
            assert output_seqlen <= n_token <= output_seqlen + 1, \
                f'Error. session_id({session_id}) request {output_seqlen} ' \
                f'tokens, but generate {n_token} tokens.\n' \
                f'prompt: {prompt}'

            first_token_latency = _per_token_latency_stats[0]
            completion_tokens = n_token
            total_tokens = n_token + input_seqlen
            stats.append([
                first_token_latency, completion_tokens, output_seqlen,
                total_tokens
            ])
            # skip the first token latency
            per_token_latency_stats.append(_per_token_latency_stats[1:])
            self.pbar.update(1)
        res_queue.put((session_id, stats, per_token_latency_stats))

    def process_request(self, requests, concurrency, temperature, top_p, top_k,
                        stream_output):
        res_queue = Queue()
        req_queue = Queue()
        threads = []

        self.pbar = tqdm(total=len(requests))

        # feed request to q
        for req in requests:
            req_queue.put(req)
        for i in range(concurrency):
            req_queue.put([None, None, None])

        start = time.time()

        # start threads
        for i in range(concurrency):
            t = Thread(target=self._inference,
                       args=(req_queue, res_queue, i, temperature, top_p,
                             top_k, stream_output),
                       daemon=True)
            t.start()
            threads.append(t)

        # wait for finish
        for t in threads:
            t.join()

        elapsed_time = time.time() - start

        stats = []
        per_token_latency_stats = []
        while not res_queue.empty():
            session_id, _stats, _per_token_latency_stats = res_queue.get()
            stats.append(np.array(_stats))
            per_token_latency_stats += [
                item for sublist in _per_token_latency_stats
                for item in sublist
            ]
        stats = np.concatenate(stats).reshape(-1, 4)

        first_token_latency_min = np.min(stats[:, 0], axis=0)
        first_token_latency_max = np.max(stats[:, 0], axis=0)
        first_token_latency_ave = np.mean(stats[:, 0], axis=0)
        completion_tokens = np.sum(stats[:, 1], axis=0)
        total_tokens = np.sum(stats[:, 3], axis=0)
        prompt_tokens = total_tokens - completion_tokens
        completion_token_throughput = completion_tokens / elapsed_time
        total_token_throughput = total_tokens / elapsed_time
        rps = len(requests) / elapsed_time
        rpm = rps * 60

        per_token_latency_stats.sort()
        percentiles = [
            np.round(
                per_token_latency_stats[int(percent *
                                            len(per_token_latency_stats))], 3)
            for percent in [0.5, 0.75, 0.95, 0.99]
        ]

        print(f'\n{"-" * 50}\nconcurrency: {concurrency}\n'
              f'elapsed_time: {elapsed_time:.3f}s\n')
        if stream_output:
            print(f'first token latency(s)(min, max, ave): '
                  f'{first_token_latency_min:.3f}, '
                  f'{first_token_latency_max:.3f}, '
                  f'{first_token_latency_ave:.3f}')
            print(f'per-token latency(s) percentile(50, 75, 95, 99): '
                  f'{percentiles}\n')
        print(
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
                    'batch', 'num_promts', 'RPS', 'RPM', 'FTL(ave)(s)',
                    'FTL(min)(s)', 'FTL(max)(s)', '50%(s)', '75%(s)', '95%(s)',
                    '99%(s)', 'throughput(out tok/s)',
                    'throughput(total tok/s)'
                ])
                writer.writerow([
                    concurrency,
                    len(requests), f'{rps:.3f}', f'{rpm:.3f}',
                    f'{first_token_latency_ave:.3f}' if stream_output else '-',
                    f'{first_token_latency_min:.3f}' if stream_output else '-',
                    f'{first_token_latency_max:.3f}' if stream_output else '-',
                    f'{percentiles[0]:.3f}' if stream_output else '-',
                    f'{percentiles[1]:.3f}' if stream_output else '-',
                    f'{percentiles[2]:.3f}' if stream_output else '-',
                    f'{percentiles[3]:.3f}' if stream_output else '-',
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
                        default='./profile_throughput.csv')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Seed used in sampling prompts from dataset')
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

    # turbomind engine args
    tb_group = parser.add_argument_group('TurboMind engine argument')
    tb_group._group_actions.append(tp_act)
    tb_group._group_actions.append(session_len_act)
    tb_group._group_actions.append(cache_count_act)
    tb_group._group_actions.append(cache_block_seq_len_act)
    ArgumentHelper.model_format(tb_group, default='hf')
    ArgumentHelper.quant_policy(tb_group, default=0)

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
            quant_policy=args.quant_policy)
    elif args.backend == 'pytorch':
        engine_config = PytorchEngineConfig(
            session_len=args.session_len,
            cache_max_entry_count=args.cache_max_entry_count,
            block_size=args.cache_block_seq_len,
            max_batch_size=args.concurrency,
            tp=args.tp,
            thread_safe=True)

    engine = Engine(args.model_path, engine_config, csv=args.csv)

    requests = sample_requests(args.dataset, args.num_prompts,
                               engine.tokenizer)

    engine.process_request(requests,
                           temperature=args.temperature,
                           top_p=args.top_p,
                           top_k=args.top_k,
                           concurrency=args.concurrency,
                           stream_output=True)


if __name__ == '__main__':
    main()
