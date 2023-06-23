import multiprocessing as mp
import time

import fire
import numpy as np

from llmdeploy.serve.fastertransformer.chatbot import Chatbot


def infer(chatbot, session_id: int, prompt: str, outseq_len: int,
          test_round: int, que: mp.Queue):
    stats = []
    for i in range(test_round):
        timestamps = []
        tokens = []
        for status, res, token in chatbot.generate(
                session_id, prompt, request_output_len=outseq_len):
            timestamps.append(time.perf_counter())
            tokens.append(token)

        first_token_latency = timestamps[1] - timestamps[0]
        token_latency = timestamps[-1] - timestamps[0]
        token = tokens[-1] - tokens[0]
        stats.append([first_token_latency, token, token_latency])
    que.put((session_id, stats))


def warmup(tritonserver_addr: str,
           model_name: str,
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
                model_name=model_name,
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


def main(tritonserver_addr: str,
         model_name: str,
         concurrency: int = 1,
         session_len: int = 2048,
         input_seqlen: int = 0,
         output_seqlen: int = 512,
         test_round: int = 10):
    warmup(tritonserver_addr, model_name, concurrency, session_len,
           output_seqlen)

    # make up a prompt that can be tokenized into {input_seqlen} tokens
    prompt = '' if input_seqlen == 0 else 'hi' + ' hi' * (input_seqlen - 1)
    que = mp.Queue()
    procs = []
    _start = time.perf_counter()
    for i in range(concurrency):
        chatbot = Chatbot(tritonserver_addr=tritonserver_addr,
                          session_len=session_len,
                          ignore_eos=True,
                          profile_generation=True)
        proc = mp.Process(target=infer,
                          args=(chatbot, i + 1, prompt, output_seqlen,
                                test_round, que))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    _end = time.perf_counter()
    elapsed_time = _end - _start

    stats = []
    while not que.empty():
        session_id, _stats = que.get()
        print(f'\n{"-" * 50}\n'
              f'session {session_id} stats: \n{_stats}\n{"-" * 50}\n')
        stats.append(_stats)

    stats = np.array(stats)
    stats.reshape(-1, 3)

    first_token_latency_min = np.min(stats[:, 0], axis=0)
    first_token_latency_max = np.max(stats[:, 0], axis=0)
    first_token_latency_ave = np.mean(stats[:, 0], axis=0)
    token_latency_min = np.min(stats[:, 2], axis=0)
    token_latency_max = np.max(stats[:, 2], axis=0)
    token_latency_ave = np.mean(stats[:, 2], axis=0)
    throughput = np.sum(stats[:, 1], axis=0) / np.sum(stats[:, 2], axis=0)
    print(f'\n{"-" * 50}\nelapsed_time: {elapsed_time}s\n'
          f'first_token latency(min, max, ave):\n'
          f'{first_token_latency_min}, {first_token_latency_max}, '
          f'{first_token_latency_ave})\ntoken latency(min, max, ave):\n'
          f'{token_latency_min}, {token_latency_max}, {token_latency_ave}\n'
          f'throughput: {throughput} token/s\n{"-" * 50}')


if __name__ == '__main__':
    fire.Fire(main)
