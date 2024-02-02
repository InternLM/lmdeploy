# Copyright (c) OpenMMLab. All rights reserved.
import csv
import json
import os
import random
import time
from queue import Queue
from threading import Thread
from typing import List, Tuple

import fire
import numpy as np
from tqdm import tqdm

from lmdeploy.tokenizer import DetokenizeState, Tokenizer
from lmdeploy.turbomind import TurboMind


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

    def __init__(self, model_path: str, tp: int, csv: str, **kwargs):
        # avoid turbomind checking chat template name by setting
        # `model_name='llama'`
        tm_model = TurboMind(model_path=model_path,
                             model_name='llama',
                             tp=tp,
                             **kwargs)
        self.tm_model = tm_model
        self.tokenizer = tm_model.tokenizer
        self.csv = csv
        self.pbar = None

    def _inference(self, req_queue: Queue, res_queue: Queue, session_id: int,
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
                    request_output_len=output_seqlen,
                    temperature=1.0,
                    top_p=1.0,
                    sequence_start=True,
                    sequence_end=True,
                    ignore_eos=True,
                    stream_output=stream_output):
                _, res, n_token = outputs
                _, state = self.tokenizer.detokenize_incrementally(res, state)
                now = time.perf_counter()
                if n_prev_token != n_token:
                    _per_token_latency_stats[n_prev_token] = np.round(
                        now - prev, 3)
                    n_prev_token = n_token
                prev = now

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

    def process_request(self,
                        requests,
                        concurrency: int = 1,
                        stream_output: bool = True):
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
                       args=(req_queue, res_queue, i, stream_output))
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


def main(dataset: str,
         model_path: str,
         concurrency: int = 64,
         num_prompts: int = 2000,
         tp: int = 1,
         top_k: int = 1,
         top_p: float = 1.0,
         temperature: float = 1.0,
         stream_output: bool = True,
         csv: str = './profile_throughput.csv',
         log_level: str = 'ERROR',
         seed: int = 0):
    """Benchmark the request throughput of lmdeploy in localhost.

    Args:
        dataset (str): Path to the dataset
        model_path (str): Path to a model in localhost or a model_repo_id in huggingface.co
        concurrency (int, optional): Number of working threads to process the sampled prompts.
            Defaults to 64.
        num_prompts (int, optional): Number of prompts to process. Defaults to 2000.
        tp (int, optional): Number of GPUs for tensor parallel. Defaults to 1.
        top_k (int, optional): The number of highest probability vocabulary tokens
            to keep for top-k-filtering. Defaults to 1.
        top_p (float, optional): the set of most probable tokens with
            probabilities that add up to top_p or higher
            are kept for generation. Defaults to 1.0.
        temperature (float, optional): The value used to modulate the next token probabilities.
            Defaults to 1.0.
        stream_output (bool, optional): Indicator for streaming output. Defaults to True.
        csv (str, optional): The path to save the result.
        log_level(str, optional): The log level. Defaults to INFO
        seed (int, optional): Seed used in sampling prompts from dataset. Defaults to 0.
    """    # noqa
    random.seed(seed)
    os.environ['TM_LOG_LEVEL'] = log_level

    engine = Engine(model_path,
                    tp=tp,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    csv=csv)

    requests = sample_requests(dataset, num_prompts, engine.tokenizer)

    engine.process_request(requests, concurrency, stream_output)


if __name__ == '__main__':
    fire.Fire(main)
