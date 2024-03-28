import csv
import json
import random
import threading
import time
from functools import partial
from queue import Queue
from threading import Thread
from typing import List, Tuple

import fire
import numpy as np
import tritonclient.grpc as grpcclient
from tqdm import tqdm

from lmdeploy.serve.turbomind.utils import prepare_tensor
from lmdeploy.tokenizer import Tokenizer


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


def stream_callback(que, result, error):
    """Callback function invoked by triton."""
    que.put((result, error))


class Engine:

    def __init__(self,
                 server_addr: str,
                 tokenzier_path: str,
                 temperature: float = 0.8,
                 top_k: int = 1,
                 top_p: float = 1.0,
                 csv: str = '',
                 log_level: str = 'ERROR',
                 **kwargs):
        self.server_addr = server_addr
        self.tokenizer = Tokenizer(tokenzier_path)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.csv = csv
        self.log_level = log_level
        self.pbar = None

    def _inference(self, req_queue: Queue, res_queue: Queue, session_id: int,
                   stream_output: bool):

        stats = []
        for prompt, input_seqlen, output_seqlen in iter(
                req_queue.get, [None, None, None]):
            res_que = Queue()
            t = threading.Thread(target=self._call_triton_server,
                                 args=(prompt, self.server_addr, session_id,
                                       output_seqlen, stream_output, res_que))
            t.start()
            timestamps = self._process_result(res_que)
            t.join()

            # record statistics
            first_token_latency = np.round(timestamps[1] - timestamps[0], 3)
            token_latency = np.round(timestamps[-1] - timestamps[0], 3)

            total_tokens = input_seqlen + output_seqlen
            stats.append([
                first_token_latency, output_seqlen, output_seqlen,
                total_tokens, token_latency
            ])
            self.pbar.update(1)
        res_queue.put((session_id, stats))

    def _process_result(self, que):
        timestamps = []
        timestamps.append(time.perf_counter())
        while True:
            res = que.get()
            if res is not None:
                _, err = res
                if err is not None:
                    print(err)
                else:
                    timestamps.append(time.perf_counter())
            else:
                return timestamps

    def _call_triton_server(self, prompts, tritonserver_addr, session_id,
                            request_output_len, stream_output, res_que):

        with grpcclient.InferenceServerClient(tritonserver_addr) as client:
            inputs = [
                prepare_tensor('prompt',
                               np.array([prompts.encode()], dtype=np.object_)),
                prepare_tensor('max_tokens',
                               np.array([request_output_len], dtype=np.int32)),
                prepare_tensor('temperature',
                               np.array([self.temperature], dtype=np.float_)),
                prepare_tensor('top_p', np.array([self.top_p],
                                                 dtype=np.float_)),
                prepare_tensor('top_k', np.array([self.top_k],
                                                 dtype=np.int32)),
                prepare_tensor('ignore_eos', np.array([True], dtype=np.bool_)),
                prepare_tensor('stream',
                               np.array([stream_output], dtype=np.bool_)),
                prepare_tensor('skip_special_tokens',
                               np.array([False], dtype=np.bool_)),
            ]

            # async_stream
            client.start_stream(partial(stream_callback, res_que))
            client.async_stream_infer('lmdeploy_model',
                                      inputs,
                                      sequence_id=session_id,
                                      sequence_start=True,
                                      sequence_end=True)
        res_que.put(None)
        return

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
        while not res_queue.empty():
            _, _stats = res_queue.get()
            if len(_stats) != 0:
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
        rps = len(requests) / elapsed_time
        rpm = rps * 60

        if (np.abs(stats[:, 1] - stats[:, 2]) <= 1).min() is False:
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
            f'token throughput (completion token): {completion_token_throughput:.3f} token/s\n'  # noqa
            f'token throughput (prompt + completion token): {total_token_throughput:.3f} token/s\n'  # noqa
            f'RPS (request per second): {rps:.3f} req/s\n'
            f'RPM (request per minute): {rpm:.3f} req/min\n'
            f'{"-" * 50}\n')

        if self.csv:
            with open(self.csv, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'batch', 'num_prompts', 'RPS', 'RPM', 'FTL(ave)(s)',
                    'FTL(min)(s)', 'FTL(max)(s)', 'throughput(out tok/s)',
                    'throughput(total tok/s)'
                ])
                writer.writerow([
                    concurrency,
                    len(requests), f'{rps:.3f}', f'{rpm:.3f}',
                    f'{first_token_latency_ave:.3f}' if stream_output else '-',
                    f'{first_token_latency_min:.3f}' if stream_output else '-',
                    f'{first_token_latency_max:.3f}' if stream_output else '-',
                    f'{completion_token_throughput:.3f}',
                    f'{total_token_throughput:.3f}'
                ])


def main(server_addr: str,
         tokenizer_path: str,
         dataset: str,
         concurrency: int = 32,
         num_prompts: int = 1000,
         top_k: int = 1,
         top_p: float = 1.0,
         temperature: float = 1.0,
         stream_output: bool = True,
         csv: str = './profile_tis_python.csv',
         seed: int = 0):
    """Benchmark the request througput of the triton inference server Python
    backend.

    Args:
        server_addr (str): Address of the triton inference server with format 0.0.0.0:0
        tokenizer_path (str): Path to the tokenizer model in localhost
        dataset (str): Path to the dataset
        concurrency (int, optional): Number of working threads to process the sampled prompts.
            Defaults to 32.
        num_prompts (int, optional): Number of prompts to process. Defaults to 1000.
        top_k (int, optional): The number of highest probability vocabulary tokens
            to keep for top-k-filtering. Defaults to 1.
        top_p (float, optional): the set of most probable tokens with
            probabilities that add up to top_p or higher
            are kept for generation. Defaults to 1.0.
        temperature (float, optional): The value used to modulate the next token probabilities.
            Defaults to 1.0.
        stream_output (bool, optional): Indicator for streaming output. Defaults to True.
        seed (int, optional): Seed used in sampling prompts from dataset. Defaults to 0.
    """    # noqa

    random.seed(seed)

    engine = Engine(server_addr,
                    tokenizer_path,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    log_level='ERROR',
                    csv=csv)

    requests = sample_requests(dataset, num_prompts, engine.tokenizer)

    engine.process_request(requests, concurrency, stream_output)


if __name__ == '__main__':
    fire.Fire(main)
