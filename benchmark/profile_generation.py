import multiprocessing as mp
import time

import fire
import numpy as np

from lmdeploy.serve.turbomind.chatbot import Chatbot


def infer(chatbot, session_id: int, prompt: str, output_seqlen: int,
          test_round: int, que: mp.Queue):
    stats = []
    for i in range(test_round):
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

        first_token_latency = timestamps[0] - start
        token_latency = timestamps[-1] - timestamps[0]
        token = tokens[-1] - tokens[0]
        stats.append([first_token_latency, token, token_latency])
        chatbot.reset_session()
    que.put((session_id, stats))


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
    print(f'end warmup, elapsed time: {round(_end - _start, 2)}s')


def main(tritonserver_addr: str,
         model_name: str,
         concurrency: int = 1,
         session_len: int = 2056,
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
                          model_name=model_name,
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

    stats = np.array(stats).reshape(-1, 3)

    first_token_latency_min = np.min(stats[:, 0], axis=0)
    first_token_latency_max = np.max(stats[:, 0], axis=0)
    first_token_latency_ave = np.mean(stats[:, 0], axis=0)
    token_latency_min = np.min(stats[:, 2], axis=0)
    token_latency_max = np.max(stats[:, 2], axis=0)
    token_latency_ave = np.mean(stats[:, 2], axis=0)
    throughput = np.sum(stats[:, 1], axis=0) / np.sum(stats[:, 2], axis=0)
    print(f'\n{"-" * 50}\nconcurrency: {concurrency}, input_tokens: '
          f'{input_seqlen}, output_tokens: {output_seqlen}\n'
          f'elapsed_time: {elapsed_time:.2f}s\n'
          f'first_token latency(min, max, ave): '
          f'{first_token_latency_min:.2f}s, {first_token_latency_max:.2f}s, '
          f'{first_token_latency_ave:.2f}s\ntoken latency(min, max, ave): '
          f'{token_latency_min:.2f}s, {token_latency_max:.2f}s, '
          f'{token_latency_ave:.2f}s\n'
          f'throughput: {throughput} token/s\n{"-" * 50}')


if __name__ == '__main__':
    fire.Fire(main)
