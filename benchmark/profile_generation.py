# import multiprocessing as mp
from threading import Thread
from queue import Queue
import time

import fire
import numpy as np

from lmdeploy.turbomind import TurboMind
from lmdeploy.model import MODELS
from transformers import AutoTokenizer


def infer(model, session_id: int, input_ids: str, output_seqlen: int,
          test_round: int, que: Queue):
    chatbot = model.create_instance()
    stats = []
    for i in range(test_round):
        start = time.perf_counter()
        timestamps = [start]
        tokens = [0]
        for outputs in chatbot.stream_infer(
                session_id,
                input_ids,
                request_output_len=output_seqlen,
                sequence_start=True,
                sequence_end=True,
                ignore_eos=True):
            res, token = outputs[0]
            timestamps.append(time.perf_counter())
            tokens.append(token)

        # TODO: ignore first token
        first_token_latency = timestamps[1] - start
        token_latency = timestamps[-1] - timestamps[0]
        token = tokens[-1] - tokens[0]
        stats.append([first_token_latency, token, token_latency])
    que.put((session_id, stats))


def warmup(model,
           concurrency: int,
           session_len: int,
           output_seqlen: int,
           warmup_round: int = 4):
    print('start to warmup ...')

    def _infer(model, session_id):
        chatbot = model.create_instance()
        for _ in range(warmup_round):
            for _ in chatbot.stream_infer(
                    session_id,
                    input_ids=[1],
                    request_output_len=output_seqlen,
                    sequence_start=True,
                    sequence_end=True,
                    ignore_eos=True):
                continue

    _start = time.perf_counter()
    procs = []
    for i in range(concurrency):
        proc = Thread(target=_infer, args=(model, i + 1))
        procs.append(proc)
        proc.start()

    try:
        for proc in procs:
            proc.join()
    except Exception:
        for proc in procs:
            proc.stop()
        exit(1)
    _end = time.perf_counter()
    print(f'end warmup, elapsed time: {round(_end - _start, 2)}s')


def main(model_path: str,
         model_name: str,
         tokenlizer: str,
         concurrency: int = 1,
         session_len: int = 2048,
         input_seqlen: int = 0,
         output_seqlen: int = 512,
         test_round: int = 10):
    tokenizer = AutoTokenizer.from_pretrained(tokenlizer)
    model = MODELS.get(model_name)()
    stop_words = model.stop_words
    tm_model = TurboMind(model_path=model_path, stop_words=stop_words)

    # warmup(tm_model, concurrency, session_len,
    #        output_seqlen)

    # make up a prompt that can be tokenized into {input_seqlen} tokens
    prompt = '' if input_seqlen == 0 else 'hi' + ' hi' * (input_seqlen - 1)
    input_ids = tokenizer.encode(prompt)
    que = Queue()
    procs = []
    _start = time.perf_counter()

    # TODO: update to the multithread version
    # for i in range(concurrency):
    #     proc = Thread(target=infer,
    #                       args=(tm_model, i + 1, input_ids, output_seqlen,
    #                             test_round, que))
    #     procs.append(proc)
    #     proc.start()

    batched_session_id = tuple(range(1, concurrency + 1))
    batched_input_ids = [input_ids] * concurrency
    proc = Thread(
        target=infer,
        args=(tm_model, batched_session_id, batched_input_ids, output_seqlen,
              test_round, que))
    procs.append(proc)
    proc.start()
    try:
        for proc in procs:
            proc.join()
    except Exception:
        for proc in procs:
            proc.stop()
        exit(1)
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
    print(f'\n{"-" * 50}\ncocurrency: {concurrency}, input_tokens: '
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
