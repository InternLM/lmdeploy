import json
import logging
import multiprocessing as mp
import random
import time

import fire
import numpy as np

from lmdeploy.serve.turbomind.chatbot import Chatbot
from lmdeploy.tokenizer import Tokenizer


def infer(chatbot, session_id: int, req_que: mp.Queue, res_que: mp.Queue):
    stats = []
    for prompt, input_seqlen, output_seqlen in iter(req_que.get,
                                                    [None, None, None]):
        timestamps = []
        tokens = []
        start = time.perf_counter()
        for status, res, token in chatbot.stream_infer(
                session_id,
                prompt,
                request_output_len=output_seqlen,
                sequence_start=True,
                sequence_end=True):
            timestamps.append(time.perf_counter())
            tokens.append(token)

        first_token_latency = np.round(timestamps[1] - start, 3)
        token_latency = np.round(timestamps[-1] - timestamps[0], 3)
        token = tokens[-1] - tokens[0]
        stats.append([first_token_latency, token, token_latency])
        print(f'session {session_id}: '
              f'input_seqlen {input_seqlen}, output_seqlen {output_seqlen}')
    res_que.put((session_id, stats))


def warmup(tritonserver_addr: str,
           concurrency: int,
           output_seqlen: int,
           warmup_round: int = 1):
    print('start to warmup ...')

    def _infer(_chatbot, session_id):
        for _ in range(warmup_round):
            for _, _, _ in _chatbot.stream_infer(
                    session_id,
                    prompt='',
                    request_output_len=output_seqlen,
                    sequence_start=True,
                    sequence_end=True):
                continue
            _chatbot.reset_session()

    _start = time.perf_counter()
    chatbots = [
        Chatbot(tritonserver_addr=tritonserver_addr,
                ignore_eos=True,
                log_level=logging.ERROR,
                profile_generation=True) for _ in range(concurrency)
    ]
    procs = []
    for i, chatbot in enumerate(chatbots):
        proc = mp.Process(target=_infer, args=(chatbot, i + 1))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    _end = time.perf_counter()
    print(f'end warmup, elapsed time: {round(_end - _start, 2)} s')


def read_dataset(tokenizer_path: str, dataset_path: str, samples: int,
                 session_len: int, que: mp.Queue):
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

    for data in filtered_dataset:
        que.put(data)
    print(f'elapsed time for filtering: '
          f'{round(time.perf_counter() - start, 2)} s')
    return len(filtered_dataset)


def main(tritonserver_addr: str,
         tokenizer_path: str,
         dataset_path: str,
         concurrency: int = 1,
         session_len: int = 2048,
         samples: int = 1000):
    warmup(tritonserver_addr, concurrency, session_len - 1)
    req_que = mp.Queue()
    res_que = mp.Queue()

    procs = []
    _start = time.perf_counter()
    for i in range(concurrency):
        chatbot = Chatbot(tritonserver_addr=tritonserver_addr,
                          display=False,
                          profile_serving=True,
                          ignore_eos=True,
                          log_level=logging.ERROR)
        proc = mp.Process(target=infer,
                          args=(chatbot, i + 1, req_que, res_que))
        procs.append(proc)
        proc.start()

    # read data and put it to queue
    n_req = read_dataset(tokenizer_path, dataset_path, samples, session_len,
                         req_que)
    for i in range(concurrency):
        req_que.put([None, None, None])

    stats = []
    for i in range(concurrency):
        session_id, _stats = res_que.get()
        print(f'\n{"-" * 50}\n'
              f'session {session_id}: processed reqs {len(_stats)}, '
              f'stats: \n{_stats}\n{"-" * 50}\n')
        stats.append(np.array(_stats))

    _end = time.perf_counter()
    elapsed_time = _end - _start

    stats = np.concatenate(stats).reshape(-1, 3)

    first_token_latency_min = np.min(stats[:, 0], axis=0)
    first_token_latency_max = np.max(stats[:, 0], axis=0)
    first_token_latency_ave = np.mean(stats[:, 0], axis=0)
    token_throughput = np.sum(stats[:, 1], axis=0) / elapsed_time
    req_throughput = n_req / elapsed_time

    print(f'\n{"-" * 50}\nconcurrency: {concurrency}\n'
          f'elapsed_time: {elapsed_time:.3f}s\n'
          f'first_token latency(min, max, ave): '
          f'{first_token_latency_min:.3f}s, {first_token_latency_max:.3f}s, '
          f'{first_token_latency_ave:.3f}s\n'
          f'token throughput: {token_throughput:.3f} token/s\n'
          f'req throughput: {req_throughput:.3f} req/s\n'
          f'{"-" * 50}\n')

    for proc in procs:
        proc.join()


if __name__ == '__main__':
    fire.Fire(main)
