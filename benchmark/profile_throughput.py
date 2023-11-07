import json
import os.path as osp
import random
import time
from queue import Queue
from threading import Thread
from typing import List, Tuple

import fire
import numpy as np

from lmdeploy.tokenizer import Tokenizer
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

    def __init__(self, model_path: str, tp: int = 1):
        tokenizer_model_path = osp.join(model_path, 'triton_models',
                                        'tokenizer')
        tokenizer = Tokenizer(tokenizer_model_path)
        tm_model = TurboMind(model_path=model_path, tp=tp)
        self.tm_model = tm_model
        self.tokenizer = tokenizer

    def _inference(self, req_queue: Queue, res_queue: Queue, session_id: int,
                   stream_output: bool):
        model_inst = self.tm_model.create_instance()
        stats = []
        timestamps = []
        tokens = []
        timestamps.append(time.perf_counter())
        for prompt, input_seqlen, output_seqlen in iter(
                req_queue.get, [None, None, None]):
            input_ids = self.tokenizer.encode(prompt)
            offset = 0
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
                res, token = outputs[0]
                self.tokenizer.decode(res, offset)
                offset = token
                timestamps.append(time.perf_counter())
                tokens.append(token)
            first_token_latency = np.round(timestamps[1] - timestamps[0], 3)
            token_latency = np.round(timestamps[-1] - timestamps[0], 3)
            completion_tokens = tokens[-1]
            total_tokens = tokens[-1] + len(input_ids)
            stats.append([
                first_token_latency, completion_tokens, output_seqlen,
                total_tokens, token_latency
            ])
            print(
                f'session {session_id}: '
                f'input_seqlen {input_seqlen}, output_seqlen {output_seqlen}, '
                f'completion_tokens {completion_tokens}')
        res_queue.put((session_id, stats))

    def process_request(self,
                        requests,
                        concurrency: int = 1,
                        stream_output: bool = True):
        res_queue = Queue()
        req_queue = Queue()
        threads = []

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
        while not res_queue.empty():
            session_id, _stats = res_queue.get()
            print(f'\n{"-" * 50}\n'
                  f'session {session_id} stats: \n{_stats}\n{"-" * 50}\n')
            stats.append(np.array(_stats))

        stats = np.concatenate(stats).reshape(-1, 5)

        first_token_latency_min = np.min(stats[:, 0], axis=0)
        first_token_latency_max = np.max(stats[:, 0], axis=0)
        first_token_latency_ave = np.mean(stats[:, 0], axis=0)
        completion_tokens = np.sum(stats[:, 1], axis=0)
        request_output_tokens = np.sum(stats[:, 2], axis=0)
        total_tokens = np.sum(stats[:, 3], axis=0)
        prompt_tokens = total_tokens - completion_tokens
        completion_token_throughput = completion_tokens / elapsed_time
        total_token_throughput = total_tokens / elapsed_time
        rqs = len(requests) / elapsed_time
        rqm = rqs * 60

        if completion_tokens != request_output_tokens:
            print(f'Did not generate requested number of tokens. '
                  f'Request {request_output_tokens:.0f}, '
                  f'but got {completion_tokens:.0f}')

        print(f'\n{"-" * 50}\nconcurrency: {concurrency}\n'
              f'elapsed_time: {elapsed_time:.3f}s\n')
        if stream_output:
            print(f'first_token latency(min, max, ave): '
                  f'{first_token_latency_min:.3f}s, '
                  f'{first_token_latency_max:.3f}s, '
                  f'{first_token_latency_ave:.3f}s\n')
        print(
            f'number of prompt tokens: {prompt_tokens:.0f}\n'
            f'number of completion tokens: {completion_tokens:.0f}\n'
            f'number of request completion tokens: {request_output_tokens:.0f}\n'  # noqa
            f'token throughput (completion token): {completion_token_throughput:.3f} token/s\n'  # noqa
            f'token throughput (prompt + completion token): {total_token_throughput:.3f} token/s\n'  # noqa
            f'PPS (request per second): {rqs:.3f} req/s\n'
            f'RPM (request per minute): {rqm:.3f} req/min\n'
            f'{"-" * 50}\n')


def main(dataset: str,
         model_path: str,
         concurrency: int = 1,
         num_prompts: int = 1000,
         tp: int = 1,
         stream_output: bool = True):

    engine = Engine(model_path, tp=tp)
    tokenizer = engine.tokenizer

    requests = sample_requests(dataset, num_prompts, tokenizer)

    engine.process_request(requests, concurrency, stream_output)


if __name__ == '__main__':
    fire.Fire(main)
