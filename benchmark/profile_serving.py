import json
import multiprocessing as mp
import random
import time

import fire
import numpy as np

from llmdeploy.serve.fastertransformer.chatbot import Chatbot
from llmdeploy.serve.fastertransformer.utils import Preprocessor


def infer(chatbot, session_id: int, req_que: mp.Queue, res_que: mp.queues):
    stats = []
    while not req_que.empty():
        prompt, output_seqlen = req_que.get()
        timestamps = []
        tokens = []
        timestamps.append(time.perf_counter())
        tokens.append(0)
        for status, res, token in chatbot.generate(
                session_id, prompt, request_output_len=output_seqlen):
            timestamps.append(time.perf_counter())
            tokens.append(token)

        first_token_latency = timestamps[1] - timestamps[0]
        token_latency = timestamps[-1] - timestamps[1]
        token = tokens[-1] - tokens[1]
        stats.append([first_token_latency, token, token_latency])
    res_que.put((session_id, stats))


def warmup(tritonserver_addr: str,
           session_len: int,
           concurrency: int,
           output_seqlen: int,
           warmup_round: int = 4):
    print('start to warmup ...')

    def _infer(_chatbot, session_id):
        for _ in range(warmup_round):
            for _, _, _ in chatbot.stream_infer(
                    session_id,
                    prompt='',
                    request_output_len=output_seqlen,
                    sequence_start=True,
                    sequence_end=True):
                continue

    _start = time.perf_counter()
    chatbots = [
        Chatbot(tritonserver_addr=tritonserver_addr,
                session_len=session_len,
                ignore_eos=True) for _ in range(concurrency)
    ]
    procs = []
    for i, chatbot in enumerate(chatbots):
        proc = mp.Process(target=_infer, args=(chatbot, i + 1))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    _end = time.perf_counter()
    print(f'end warmup, elapsed time: {_end - _start}s')


def read_dataset(tritonserver_addr, dataset_path: str, samples: int,
                 test_round: int, session_len: int):
    with open(dataset_path) as f:
        dataset = json.load(f)
        dataset = [data for data in dataset if len(data['conversations']) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [(data['conversations'][0]['value'],
                    data['conversations'][1]['value']) for data in dataset]

    preprocessor = Preprocessor(tritonserver_addr)

    prompts = [prompt for prompt, _ in dataset]
    completions = [completion for _, completion in dataset]
    prompts_token_ids, _ = preprocessor(prompts)
    completions_token_ids, = preprocessor(completions)
    filtered_dataset = []
    for prompt, _, input_ids, output_ids in zip(dataset, prompts_token_ids,
                                                completions_token_ids):
        input_len = len(input_ids)
        output_len = len(output_ids)
        if input_len + output_len > session_len:
            # ignore too long conversation
            continue
        filtered_dataset.append([prompt, output_len])

    if samples > 0:
        filtered_dataset = random.sample(filtered_dataset, samples)

    filtered_dataset *= test_round
    random.shuffle(filtered_dataset)

    que = mp.Queue()
    que.put(data for data in filtered_dataset)
    return que


def main(tritonserver_addr: str,
         dataset_path: str,
         concurrency: int = 1,
         session_len: int = 2048,
         samples: int = 2000,
         test_round: int = 1):
    warmup(tritonserver_addr, concurrency, session_len, session_len)

    req_que = read_dataset(tritonserver_addr, dataset_path, samples,
                           test_round, session_len)

    res_que = mp.Queue()
    procs = []
    _start = time.perf_counter()
    for i in range(concurrency):
        chatbot = Chatbot(tritonserver_addr=tritonserver_addr,
                          session_len=session_len,
                          ignore_eos=True)
        proc = mp.Process(target=infer,
                          args=(chatbot, i + 1, req_que, res_que))
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
        stats.append(_stats)

    stats = np.array(stats)
    stats.reshape(-1, 3)

    first_token_latency_min = np.min(stats[:, 0], axis=0)
    first_token_latency_max = np.max(stats[:, 0], axis=0)
    first_token_latency_ave = np.mean(stats[:, 0], axis=0)
    throughput = np.sum(stats[:, 1], axis=0) / elapsed_time
    print(f'\n{"-" * 50}\nelapsed_time: {elapsed_time}s\n'
          f'first_token latency(min, max, ave):\n'
          f'{first_token_latency_min}, {first_token_latency_max}, '
          f'{first_token_latency_ave})\ntoken latency(min, max, ave):\n'
          f'throughput: {throughput} token/s\n{"-" * 50}')


if __name__ == '__main__':
    fire.Fire(main)
