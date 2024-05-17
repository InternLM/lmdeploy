import csv
import json
import random
import time
from queue import Queue
from threading import Thread
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import fire
import numpy as np
from tqdm import tqdm

from lmdeploy.serve.openai.api_client import APIClient
from lmdeploy.tokenizer import Tokenizer


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: Tokenizer,
    role: str,
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
        tokenized_dataset.append(
            (
                [{'role': role, 'content': prompts[i]}],
                prompt_token_ids[i],
                output_len
            )
        )

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

    def __init__(self,
                 server_addr: str,
                 tokenzier_path: str,
                 temperature: float = 0.8,
                 top_p: float = 1.0,
                 csv: str = '',
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None,
                 **kwargs):
        self.tokenizer = Tokenizer(tokenzier_path)
        self.server_addr = server_addr
        self.temperature = temperature
        self.top_p = top_p
        self.csv = csv
        self.api_key = api_key
        client = APIClient(self.server_addr, api_key=self.api_key)
        if model_name is None:
            self.model_name = client.available_models[0]
            print(f'using model: {self.model_name}\n')
        else:
            self.model_name = model_name
        self.pbar = None

    def _inference(self, req_queue: Queue, res_queue: Queue, session_id: int,
                   stream_output: bool):

        stats = []
        client = APIClient(self.server_addr, api_key=self.api_key)

        for prompt, input_seqlen, output_seqlen in iter(
                req_queue.get, [None, None, None]):
            timestamps = []
            timestamps.append(time.perf_counter())
            full_output = ''
            failed = 0
            try:
                for output in client.chat_completions_v1(
                        model=self.model_name,
                        messages=prompt,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        n=1,
                        max_tokens=output_seqlen,
                        stream=stream_output,
                        session_id=session_id,
                        ignore_eos=True):
                    # Here we ignore the index of the multiple outputs and
                    # just put all of them together to compute tokens.
                    for choice in output.get('choices', []):
                        if stream_output:
                            full_output += choice['delta'].get('content', '')
                        else:
                            full_output += choice['message']['content']
                    timestamps.append(time.perf_counter())
            except Exception as e:
                print(f'inference failed: {e}')
                failed = 1
                timestamps.append(time.perf_counter())

            first_token_latency = np.round(timestamps[1] - timestamps[0], 3)
            token_latency = np.round(timestamps[-1] - timestamps[0], 3)
            # assert output.pop('finish_reason') == 'length', \
            #     f'Error. session_id({session_id}) request {output_seqlen} ' \
            #     f'tokens, but `finish_reason` is not `length`'
            tokenlizer_start = time.perf_counter()
            real_output_seqlen = len(self.tokenizer(full_output).input_ids)
            tokenlizer_finish = time.perf_counter()
            tokenlizer_time = tokenlizer_finish - tokenlizer_start
            total_tokens = input_seqlen + real_output_seqlen
            stats.append([
                first_token_latency, real_output_seqlen, output_seqlen,
                total_tokens, token_latency, tokenlizer_time, failed
            ])
            self.pbar.update(1)

        res_queue.put((session_id, stats))

    def process_request(self,
                        requests,
                        concurrency: int = 1,
                        stream_output: bool = False):
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
            session_id, _stats = res_queue.get()
            # print(f'\n{"-" * 50}\n'
            #       f'session {session_id} stats: \n{_stats}\n{"-" * 50}\n')
            stats.append(np.array(_stats))

        stats = np.concatenate(stats).reshape(-1, 7)

        tokenlizer_time = np.sum(stats[:, 5], axis=0) / concurrency
        elapsed_time -= tokenlizer_time

        first_token_latency_min = np.min(stats[:, 0], axis=0)
        first_token_latency_max = np.max(stats[:, 0], axis=0)
        first_token_latency_ave = np.mean(stats[:, 0], axis=0)
        failed_requests = np.sum(stats[:, 6], axis=0)
        completion_tokens = np.sum(stats[:, 1], axis=0)
        request_output_tokens = np.sum(stats[:, 2], axis=0)
        total_tokens = np.sum(stats[:, 3], axis=0)
        prompt_tokens = total_tokens - completion_tokens
        local_tokenlizer_throughput = completion_tokens / tokenlizer_time
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

        if failed_requests > 0:
            print(f'number of failed requests: {failed_requests:.0f}\n')

        print(
            f'number of prompt tokens: {prompt_tokens:.0f}\n'
            f'number of completion tokens: {completion_tokens:.0f}\n'
            f'local tokenlizer throughput (completion token): {local_tokenlizer_throughput:.3f} token/s\n'  # noqa
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
         api_key: Optional[str] = None,
         model_name: Optional[str] = None,
         concurrency: int = 128,
         num_prompts: int = 5000,
         top_p: float = 1.0,
         temperature: float = 1.0,
         stream_output: bool = False,
         csv: str = './profile_api_server.csv',
         seed: int = 0,
         role: str = 'user',
         ):
    """Benchmark the request througput of api server.

    Args:
        server_addr (str): http url of api_server with format http://0.0.0.0:0
        tokenizer_path (str): Path to the tokenizer model in localhost
        dataset (str): Path to the dataset
        concurrency (int, optional): Number of working threads to process the sampled prompts.
            Defaults to 128.
        num_prompts (int, optional): Number of prompts to process. Defaults to 5000.
        top_p (float, optional): the set of most probable tokens with
            probabilities that add up to top_p or higher
            are kept for generation. Defaults to 1.0.
        temperature (float, optional): The value used to modulate the next token probabilities.
            Defaults to 1.0.
        stream_output (bool, optional): Indicator for streaming output. Defaults to False.
        csv (str, optional): The path to save the result.
        seed (int, optional): Seed used in sampling prompts from dataset. Defaults to 0.
        role (str, optional): The role of the messages author in prompts. Defaults to 'user'
    """    # noqa
    addr_schem = urlparse(server_addr).scheme
    if addr_schem not in ['http', 'https']:
        print(f'[WARNING] server_addr of the api_server should '
              f'start with "http://" or "https://", but got "{server_addr}"')
        server_addr = 'http://' + server_addr.strip()
    print(f'[INFO] using server_addr: {server_addr}')

    random.seed(seed)

    engine = Engine(server_addr,
                    tokenizer_path,
                    top_p=top_p,
                    temperature=temperature,
                    csv=csv,
                    api_key=api_key,
                    model_name=model_name)

    requests = sample_requests(dataset, num_prompts, engine.tokenizer, role)

    engine.process_request(requests, concurrency, stream_output)


if __name__ == '__main__':
    fire.Fire(main)
