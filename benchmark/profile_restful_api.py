import json
import random
import time
from queue import Queue
from threading import Thread

import fire
import numpy as np

from lmdeploy.serve.openai.api_client import get_streaming_response
from lmdeploy.tokenizer import Tokenizer


def infer(server_addr: str, session_id: int, req_queue: Queue, res_que: Queue,
          stream_output: bool):
    stats = []
    for prompt, input_seqlen, output_seqlen in iter(req_queue.get,
                                                    [None, None, None]):
        if prompt is None:
            break
        timestamps = []
        tokens = []
        timestamps.append(time.perf_counter())
        for res, token, status in get_streaming_response(
                prompt,
                server_addr,
                session_id,
                request_output_len=output_seqlen,
                interactive_mode=False,
                ignore_eos=True,
                stream=stream_output):
            timestamps.append(time.perf_counter())
            tokens.append(token)

        first_token_latency = np.round(timestamps[1] - timestamps[0], 3)
        token_latency = np.round(timestamps[-1] - timestamps[0], 3)
        completion_tokens = tokens[-1]
        total_tokens = tokens[-1] + input_seqlen
        stats.append([
            first_token_latency, completion_tokens, output_seqlen,
            total_tokens, token_latency
        ])
        print(f'session {session_id}: '
              f'input_seqlen {input_seqlen}, output_seqlen {output_seqlen}, '
              f'completion_tokens {completion_tokens}')
    res_que.put((session_id, stats))


def warmup(server_addr: str,
           concurrency: int,
           output_seqlen: int,
           warmup_round: int = 1,
           stream_output: bool = False):
    print('start to warmup ...')

    def _infer(server_addr, session_id):
        for _ in range(warmup_round):
            for _ in get_streaming_response('',
                                            server_addr,
                                            session_id,
                                            request_output_len=output_seqlen,
                                            interactive_mode=False,
                                            stream=stream_output,
                                            ignore_eos=True):
                continue

    _start = time.perf_counter()
    procs = []
    for i in range(concurrency):
        proc = Thread(target=_infer, args=(server_addr, i + 1))
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

    print('start tokenization. This takes a while, please wait...')
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

    que = Queue()
    for data in filtered_dataset:
        que.put(data)
    que.put((None, None, None))
    print(f'elapsed time for filtering: '
          f'{round(time.perf_counter() - start, 2)} s')
    return que, len(filtered_dataset)


def main(server_addr: str,
         tokenizer_path: str,
         dataset_path: str,
         concurrency: int = 1,
         session_len: int = 2048,
         samples: int = 1000,
         stream_output: bool = False,
         seed: int = 0):
    random.seed(seed)
    api_url = server_addr + '/v1/chat/interactive'
    warmup(api_url, concurrency, session_len - 1, 4, stream_output)
    req_queue, n_req = read_dataset(tokenizer_path, dataset_path, samples,
                                    session_len)
    for i in range(concurrency):
        req_queue.put([None, None, None])
    res_que = Queue()
    procs = []
    _start = time.perf_counter()
    for i in range(concurrency):
        proc = Thread(target=infer,
                      args=(api_url, i + 1, req_queue, res_que, stream_output))
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
    rqs = n_req / elapsed_time
    rqm = rqs * 60

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
        f'RPS (request per second): {rqs:.3f} req/s\n'
        f'RPM (request per minute): {rqm:.3f} req/min\n'
        f'{"-" * 50}\n')


if __name__ == '__main__':
    fire.Fire(main)
