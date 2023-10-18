import json
import multiprocessing as mp
import random
import time
from typing import Iterable, List

import fire
import numpy as np
import requests

from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import get_logger


def get_streaming_response(prompt: str,
                           api_url: str,
                           session_id: int,
                           request_output_len: int,
                           stream: bool = True,
                           sequence_start: bool = True,
                           sequence_end: bool = False,
                           ignore_eos: bool = False) -> Iterable[List[str]]:
    headers = {'User-Agent': 'Test Client'}
    pload = {
        'prompt': prompt,
        'stream': stream,
        'session_id': session_id,
        'request_output_len': request_output_len,
        'sequence_start': sequence_start,
        'sequence_end': sequence_end,
        'ignore_eos': ignore_eos
    }
    response = requests.post(api_url,
                             headers=headers,
                             json=pload,
                             stream=stream)
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b'\n'):
        if chunk:
            data = json.loads(chunk.decode('utf-8'))
            output = data['text']
            tokens = data['tokens']
            yield output, tokens


def infer(server_addr: str, session_id: int, req_queue: mp.Queue,
          res_que: mp.Queue):
    stats = []
    while not req_queue.empty():
        prompt, input_seqlen, output_seqlen = req_queue.get()
        get_logger('profile_restful_api').info(
            f'request info: session {session_id}, '
            f'input_seqlen {input_seqlen}, output_seqlen {output_seqlen}')
        timestamps = []
        tokens = []
        start = time.perf_counter()
        for res, token in get_streaming_response(
                prompt,
                server_addr,
                session_id,
                request_output_len=output_seqlen,
                sequence_start=True,
                sequence_end=True):
            timestamps.append(time.perf_counter())
            tokens.append(token)

        first_token_latency = timestamps[1] - start
        token_latency = timestamps[-1] - timestamps[0]
        token = tokens[-1] - tokens[0]
        stats.append([first_token_latency, token, token_latency])
    res_que.put((session_id, stats))


def warmup(server_addr: str,
           concurrency: int,
           output_seqlen: int,
           warmup_round: int = 1):
    print('start to warmup ...')

    def _infer(server_addr, session_id):
        for _ in range(warmup_round):
            for _, _ in get_streaming_response(
                    '',
                    server_addr,
                    session_id,
                    request_output_len=output_seqlen,
                    sequence_start=True,
                    sequence_end=True):
                continue

    _start = time.perf_counter()
    procs = []
    for i in range(concurrency):
        proc = mp.Process(target=_infer, args=(server_addr, i + 1))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    _end = time.perf_counter()
    print(f'end warmup, elapsed time: {round(_end - _start, 2)} s')


def read_dataset(tokenizer_path: str, dataset_path: str, samples: int,
                 session_len: int):
    start = time.perf_counter()
    with open(dataset_path) as f:
        dataset = json.load(f)
        dataset = [data for data in dataset if len(data['conversations']) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [(data['conversations'][0]['value'],
                    data['conversations'][1]['value']) for data in dataset]
        prompts = [prompt for prompt, _ in dataset]
        completions = [completion for _, completion in dataset]
        print(f'elapsed time for read data: '
              f'{round(time.perf_counter() - start, 2)} s')

    start = time.perf_counter()
    tokenizer = Tokenizer(tokenizer_path)
    prompts_token_lens = [len(tokenizer.encode(prompt)) for prompt in prompts]
    completions_token_lens = [
        len(tokenizer.encode(prompt)) for prompt in completions
    ]
    print(f'elapsed time for tokenization: '
          f'{round(time.perf_counter() - start, 2)} s')

    start = time.perf_counter()
    filtered_dataset = []
    for (prompt, _), input_len, output_len in zip(dataset, prompts_token_lens,
                                                  completions_token_lens):
        if input_len + output_len > session_len:
            # ignore too long conversation
            continue
        filtered_dataset.append([prompt, input_len, output_len])

    if samples > 0:
        filtered_dataset = random.sample(filtered_dataset, samples)

    que = mp.Queue()
    for data in filtered_dataset:
        que.put(data)
    print(f'elapsed time for filtering: '
          f'{round(time.perf_counter() - start, 2)} s')
    return que, len(filtered_dataset)


def main(server_addr: str,
         tokenizer_path: str,
         dataset_path: str,
         concurrency: int = 1,
         session_len: int = 2048,
         samples: int = 1000):
    api_url = server_addr + '/generate'
    warmup(api_url, concurrency, session_len - 1)
    req_queue, n_req = read_dataset(tokenizer_path, dataset_path, samples,
                                    session_len)
    res_que = mp.Queue()
    procs = []
    _start = time.perf_counter()
    for i in range(concurrency):
        proc = mp.Process(target=infer,
                          args=(api_url, i + 1, req_queue, res_que))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    _end = time.perf_counter()
    elapsed_time = _end - _start

    stats = []
    while not res_que.empty():
        session_id, _stats = res_que.get()
        print(f'\n{"-" * 50}\n'
              f'session {session_id} stats: \n{_stats}\n{"-" * 50}\n')
        stats.append(np.array(_stats))

    stats = np.concatenate(stats).reshape(-1, 3)

    first_token_latency_min = np.min(stats[:, 0], axis=0)
    first_token_latency_max = np.max(stats[:, 0], axis=0)
    first_token_latency_ave = np.mean(stats[:, 0], axis=0)
    token_throughput = np.sum(stats[:, 1], axis=0) / elapsed_time
    req_throughput = n_req / elapsed_time

    print(f'\n{"-" * 50}\nconcurrency: {concurrency}\n'
          f'elapsed_time: {elapsed_time:.2f}s\n'
          f'first_token latency(min, max, ave): '
          f'{first_token_latency_min:.2f}s, {first_token_latency_max:.2f}s, '
          f'{first_token_latency_ave:.2f}s\n'
          f'token throughput: {token_throughput:.2f} token/s\n'
          f'req throughput: {req_throughput:.2f} req/s\n'
          f'{"-" * 50}\n')


if __name__ == '__main__':
    fire.Fire(main)
