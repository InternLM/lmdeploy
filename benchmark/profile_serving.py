import json
import multiprocessing as mp
import random
import time

import fire
import numpy as np

from llmdeploy.serve.fastertransformer.chatbot import Chatbot
from llmdeploy.serve.fastertransformer.utils import Preprocessor


def infer(chatbot, session_id: int, req_que: mp.Queue, res_que: mp.Queue):
    stats = []
    while not req_que.empty():
        prompt, output_seqlen = req_que.get()
        timestamps = []
        tokens = []
        timestamps.append(time.perf_counter())
        tokens.append(0)
        for status, res, token in chatbot.stream_infer(
                session_id,
                prompt,
                request_output_len=output_seqlen,
                sequence_start=True,
                sequence_end=True):
            timestamps.append(time.perf_counter())
            tokens.append(token)

        first_token_latency = timestamps[1] - timestamps[0]
        token_latency = timestamps[-1] - timestamps[1]
        token = tokens[-1]
        stats.append([first_token_latency, token, token_latency])
    res_que.put((session_id, stats))


def warmup(tritonserver_addr: str,
           model_name: str,
           concurrency: int,
           session_len: int,
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
            chatbot.reset_session()

    _start = time.perf_counter()
    chatbots = [
        Chatbot(tritonserver_addr=tritonserver_addr,
                model_name=model_name,
                session_len=session_len,
                ignore_eos=True,
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
    print(f'end warmup, elapsed time: {_end - _start}s')


def tokenize(tritonserver_addr, prompts):
    # TODO: make 'preprocessor' supports batch inference
    preprocessor = Preprocessor(tritonserver_addr)
    prompts_token_ids = []
    for prompt in prompts:
        token_ids, _ = preprocessor(prompt)
        token_ids = token_ids.tolist()
        prompts_token_ids.append(len(token_ids[0]))
    return prompts_token_ids


def read_dataset(tritonserver_addr, dataset_path: str, samples: int,
                 test_round: int, session_len: int):
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
    prompts_token_lens = tokenize(tritonserver_addr, prompts)
    completions_token_lens = tokenize(tritonserver_addr, completions)
    print(f'elapsed time for tokenization: '
          f'{round(time.perf_counter() - start, 2)} s')

    start = time.perf_counter()
    filtered_dataset = []
    for (prompt, _), input_len, output_len in zip(dataset, prompts_token_lens,
                                                  completions_token_lens):
        if input_len + output_len > session_len:
            # ignore too long conversation
            continue
        filtered_dataset.append([prompt, output_len])

    if samples > 0:
        filtered_dataset = random.sample(filtered_dataset, samples)

    filtered_dataset *= test_round
    random.shuffle(filtered_dataset)
    que = mp.Queue()
    for data in filtered_dataset:
        que.put(data)
    print(f'elapsed time for filtering: '
          f'{round(time.perf_counter() - start, 2)} s')
    return que


def main(tritonserver_addr: str,
         model_name: str,
         dataset_path: str,
         concurrency: int = 1,
         session_len: int = 2048,
         samples: int = 2000,
         test_round: int = 1):
    warmup(tritonserver_addr, model_name, concurrency, session_len,
           session_len)
    req_que = read_dataset(tritonserver_addr, dataset_path, samples,
                           test_round, session_len)
    res_que = mp.Queue()
    procs = []
    _start = time.perf_counter()
    for i in range(concurrency):
        chatbot = Chatbot(tritonserver_addr=tritonserver_addr,
                          model_name=model_name,
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

    stats = np.array(stats).reshape(-1, 3)

    first_token_latency_min = np.min(stats[:, 0], axis=0)
    first_token_latency_max = np.max(stats[:, 0], axis=0)
    first_token_latency_ave = np.mean(stats[:, 0], axis=0)
    throughput = np.sum(stats[:, 1], axis=0) / elapsed_time
    print(f'\n{"-" * 50}\ncocurrency: {concurrency}\n'
          f'elapsed_time: {elapsed_time}s\n'
          f'first_token latency(min, max, ave):\n'
          f'{first_token_latency_min}s, {first_token_latency_max}s, '
          f'{first_token_latency_ave}s)\n'
          f'throughput:\n{throughput} token/s\n{"-" * 50}')


if __name__ == '__main__':
    fire.Fire(main)
