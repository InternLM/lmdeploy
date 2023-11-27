# Copyright (c) OpenMMLab. All rights reserved.
# import multiprocessing as mp
import argparse
import csv
import logging
import os
import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import List

import numpy as np
from pynvml import (NVMLError, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetMemoryInfo, nvmlDeviceGetName,
                    nvmlDeviceGetPowerState, nvmlDeviceGetTemperature,
                    nvmlInit, nvmlShutdown, nvmlSystemGetDriverVersion)
from tqdm import tqdm

from lmdeploy.turbomind import TurboMind


def infer(model, session_id: int, input_ids: List, output_seqlen: int,
          test_round: int, que: Queue):
    chatbot = model.create_instance()
    stats = []
    for _ in range(test_round):
        token_latency_stats = [0] * (output_seqlen + 1)
        prev = time.perf_counter()
        n_pre_token = 0
        """
        The iterator provided by `stream_infer` denotes the number of generated tokens so far,
        which is represented by the variable `n_token`.
        Please note that `n_token` is not a continuous value. In other words, during the iteration,
        its value might be 5, 7, 8, 16, and so on, rather than 1, 2, 3, 4, etc.
        So, it is quite difficult to get the latency of each generated token.
        As a work-around, we set the latency `new-prev` of each iteration to the first token of
        the new generated tokens, and leave the latency of the rest tokens being 0.
        For example, in the first iteration, 5 tokens are generated.
        The time elapsing in this iteration `now-prev` is set to the latency of first token of
        the 5 tokens, i.e. `token_latency_stats[0]`, and `token_latency_stats[1:4]` is set 0`
        """   # noqa: E501
        for outputs in chatbot.stream_infer(session_id,
                                            input_ids,
                                            request_output_len=output_seqlen,
                                            sequence_start=True,
                                            sequence_end=True,
                                            ignore_eos=True,
                                            stream_output=True):
            _, n_token = outputs[0]
            now = time.perf_counter()
            if n_pre_token != n_token:
                token_latency_stats[n_pre_token] = np.round(now - prev, 3)
                n_pre_token = n_token
            prev = now

        assert output_seqlen <= n_token <= output_seqlen + 1, \
            f'Error. session_id({session_id}) request {output_seqlen} ' \
            f'tokens, but generate {n_token} tokens'
        stats.append(token_latency_stats[:output_seqlen])
    que.put((session_id, stats))


def warmup(model,
           concurrency: int,
           input_ids: List[int],
           output_seqlen: int,
           warmup_round: int = 2):
    print('start to warmup ...')

    def _infer(model, session_id):
        chatbot = model.create_instance()
        for _ in range(warmup_round):
            for _ in chatbot.stream_infer(session_id,
                                          input_ids=input_ids,
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


def profile_throughput(model_path: str,
                       concurrency: int = 1,
                       input_seqlen: int = 1,
                       output_seqlen: int = 512,
                       test_round: int = 10,
                       tp: int = 1,
                       **kwargs):
    # avoid turbomind checking chat template name by setting
    # `model_name='llama'`
    tm_model = TurboMind(model_path=model_path,
                         tp=tp,
                         model_name='llama',
                         **kwargs)
    tokenizer = tm_model.tokenizer

    # make up a prompt that can be tokenized into {input_seqlen} tokens
    assert input_seqlen > 0, 'input_seqlen should > 0'
    input_ids = tokenizer('hi').input_ids
    input_ids = input_ids * input_seqlen

    warmup(tm_model, concurrency, input_ids, output_seqlen)

    que = Queue()
    procs = []
    _start = time.perf_counter()

    # TODO: update to the multithread version
    for i in range(concurrency):
        proc = Thread(target=infer,
                      args=(tm_model, i + 1, input_ids, output_seqlen,
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

    token_latency_stats = []
    while not que.empty():
        _, _stats = que.get()
        token_latency_stats += _stats

    # The shape is [concurrency*test_round, output_seqlen]
    token_latency_stats = np.stack(token_latency_stats, axis=0)

    first_token_latency_min = np.round(
        np.min(token_latency_stats[:, 0], axis=0), 3)
    first_token_latency_max = np.round(
        np.max(token_latency_stats[:, 0], axis=0), 3)
    first_token_latency_ave = np.round(
        np.mean(token_latency_stats[:, 0], axis=0), 3)
    token_latency_max = np.round(np.max(np.sum(token_latency_stats, axis=1)),
                                 3)
    token_latency_min = np.round(np.min(np.sum(token_latency_stats, axis=1)),
                                 3)
    token_latency_ave = np.round(np.mean(np.sum(token_latency_stats, axis=1)),
                                 3)
    # sort token_latency without the first token's latency
    sorted_token_latency = np.sort(token_latency_stats[:, 1:].flatten())
    percentiles = [
        np.round(
            sorted_token_latency[int(percent * len(sorted_token_latency))], 3)
        for percent in [0.5, 0.75, 0.95, 0.99]
    ]

    throughput = np.round(token_latency_stats.size / elapsed_time, 2)
    print(f'\n{"-" * 50}\ntotal time: {elapsed_time:.2f}s\n'
          f'concurrency: {concurrency}, test_round: {test_round}\n'
          f'input_tokens: {input_seqlen}, output_tokens: {output_seqlen}\n'
          f'first_token latency(min, max, ave): '
          f'{first_token_latency_min}s, {first_token_latency_max}s, '
          f'{first_token_latency_ave}s\ntotal_token latency(min, max, ave): '
          f'{token_latency_min}s, {token_latency_max}s, '
          f'{token_latency_ave}s\n'
          f'token_latency percentiles(50%,75%,95%,99%)(s): {percentiles}\n'
          f'throughput: {throughput} token/s\n{"-" * 50}')
    return tm_model.model_name, \
        [first_token_latency_min, first_token_latency_max,
         first_token_latency_ave], \
        percentiles, throughput, tm_model.gpu_count


class MemoryMonitor:
    from multiprocessing import Manager
    max_mem = Manager().Value('f', 0)  # GB
    device_count = Manager().Value('f', 0)

    @staticmethod
    def nvidia_info():
        # pip install nvidia-ml-py
        nvidia_dict = {
            'state': True,
            'nvidia_version': '',
            'nvidia_count': 0,
            'gpus': []
        }
        try:
            nvmlInit()
            nvidia_dict['nvidia_version'] = nvmlSystemGetDriverVersion()
            nvidia_dict['nvidia_count'] = nvmlDeviceGetCount()
            for i in range(nvidia_dict['nvidia_count']):
                handle = nvmlDeviceGetHandleByIndex(i)
                memory_info = nvmlDeviceGetMemoryInfo(handle)
                gpu = {
                    'gpu_name': nvmlDeviceGetName(handle),
                    'total': memory_info.total,
                    'free': memory_info.free,
                    'used': memory_info.used,
                    'temperature': f'{nvmlDeviceGetTemperature(handle, 0)}℃',
                    'powerStatus': nvmlDeviceGetPowerState(handle)
                }
                nvidia_dict['gpus'].append(gpu)
        except NVMLError as _:  # noqa
            nvidia_dict['state'] = False
        except Exception as _:  # noqa
            nvidia_dict['state'] = False
        finally:
            try:
                nvmlShutdown()
            except:  # noqa
                pass
        return nvidia_dict

    @classmethod
    def mem_monitor(cls):
        info = cls.nvidia_info()
        max_mem = 0
        mem_start = 0
        cls.device_count.value = len(info['gpus'])
        for used_total in info['gpus']:
            mem_start += used_total['used']
        while True:
            info = cls.nvidia_info()
            used = 0
            for used_total in info['gpus']:
                used += used_total['used']
            if used > max_mem:
                max_mem = used
                cls.max_mem.value = (max_mem - mem_start) / (1 << 30)

    @classmethod
    def start(cls):
        cls._running = True
        from multiprocessing import Process
        cls.proc = Process(target=cls.mem_monitor)
        cls.proc.start()

    @classmethod
    def terminate(cls) -> float:
        """Terminate the subprocess and return maximum memory."""
        cls.proc.kill()
        return cls.max_mem.value


@dataclass
class ProfileResult:
    model_name: str
    batch: int
    prompt_tokens: int
    completion_tokens: int
    first_token_latency: List
    percentiles: List
    throughput_per_proc: float
    throughput_per_node: float
    mem_per_proc: float
    mem_per_gpu: float
    mem_per_node: float


def parse_args():
    parser = argparse.ArgumentParser(description='Regression Test')
    parser.add_argument('model_path',
                        type=str,
                        help='the path of the model in localhost or '
                        'the repo_id of the model in huggingface.co')
    parser.add_argument('--concurrency',
                        nargs='+',
                        type=int,
                        help='how many requests launched concurrently',
                        default=[1, 16, 32, 64])
    parser.add_argument(
        '--prompt-tokens',
        nargs='+',
        type=int,
        help='how many requests launched concurrently. One-to-one'
        'correspondence with completion-tokens',
        default=[1, 128, 128, 2048, 2048])
    parser.add_argument('--completion-tokens',
                        nargs='+',
                        type=int,
                        help='how many tokens to be generated. One-to-one'
                        'correspondence with prompt-tokens',
                        default=[128, 128, 2048, 128, 2048])
    parser.add_argument('--tp', type=int, help='Tensor parallel', default=1)
    parser.add_argument('--top_k',
                        type=int,
                        help='The number of highest probability vocabulary '
                        'tokens to keep for top-k-filtering',
                        default=1)
    parser.add_argument('--top_p',
                        type=float,
                        help='the set of most probable tokens with '
                        'probabilities that add up to top_p or higher '
                        'are kept for generation',
                        default=1.0)
    parser.add_argument('--temperature',
                        type=float,
                        help='The value used to modulate the next token '
                        'probabilities',
                        default=0.8)
    parser.add_argument('--csv',
                        type=str,
                        help='Where to save the result.',
                        default='profile_generation.csv')
    parser.add_argument('--log-level',
                        help='set log level',
                        default='ERROR',
                        choices=list(logging._nameToLevel.keys()))
    parser.add_argument('--test-round',
                        type=int,
                        help='number of test rounds',
                        default=10)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ['TM_LOG_LEVEL'] = args.log_level
    results: List[ProfileResult] = []
    for batch in tqdm(args.concurrency):
        for prompt_tokens, completion_tokens in tqdm(
                zip(args.prompt_tokens, args.completion_tokens)):
            MemoryMonitor.start()
            from functools import partial
            from multiprocessing import Pool
            profile_target = partial(profile_throughput,
                                     concurrency=batch,
                                     input_seqlen=prompt_tokens,
                                     output_seqlen=completion_tokens,
                                     tp=args.tp,
                                     top_k=args.top_k,
                                     top_p=args.top_p,
                                     temperature=args.temperature,
                                     test_round=args.test_round)
            output = Pool(1).map(profile_target, (args.model_path, ))
            model_name, first_token_latency, percentiles, \
                throughput_per_proc, tp = output[0]
            time.sleep(5)  # wait a while for releasing GPU mem
            memory = MemoryMonitor.terminate()
            device_count = MemoryMonitor.device_count.value
            results.append(
                ProfileResult(model_name=model_name,
                              batch=batch,
                              prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens,
                              first_token_latency=first_token_latency,
                              percentiles=percentiles,
                              throughput_per_proc=throughput_per_proc,
                              throughput_per_node=throughput_per_proc / tp *
                              device_count,
                              mem_per_proc=memory,
                              mem_per_gpu=memory / tp,
                              mem_per_node=memory / tp * device_count))
    with open(args.csv, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'batch', 'prompt_tokens', 'completion_tokens',
            '1st_token_latency(min)(s)', '1st_token_latency(max)(s)',
            '1st_token_latency(ave)(s)', 'percentile50(s)', 'percentile75(s)',
            'percentile95(s)', 'percentile99(s)', 'throughput(token/s)',
            'mem_per_proc(GB)', 'mem_per_gpu(GB)'
        ])
        for re in results:
            writer.writerow([
                re.batch, re.prompt_tokens, re.completion_tokens,
                re.first_token_latency[0], re.first_token_latency[1],
                re.first_token_latency[2], re.percentiles[0],
                re.percentiles[1], re.percentiles[2], re.percentiles[3],
                f'{re.throughput_per_proc:.2f}', f'{re.mem_per_proc:.2f}',
                f'{re.mem_per_gpu:.2f}'
            ])


if __name__ == '__main__':
    main()
