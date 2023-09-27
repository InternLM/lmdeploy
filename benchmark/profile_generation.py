# Copyright (c) OpenMMLab. All rights reserved.
# import multiprocessing as mp
import argparse
import csv
import logging
import os
import os.path as osp
import random
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


def infer(model,
          session_id: int,
          input_ids: str,
          output_seqlen: int,
          test_round: int,
          que: Queue,
          sampling_param=None):
    chatbot = model.create_instance()
    stats = []
    for i in range(test_round):
        start = time.perf_counter()
        timestamps = []
        tokens = []
        for outputs in chatbot.stream_infer(session_id,
                                            input_ids,
                                            request_output_len=output_seqlen,
                                            sequence_start=True,
                                            sequence_end=True,
                                            ignore_eos=True,
                                            sampling_param=sampling_param):
            if len(outputs) > 1:
                res, token = outputs[-2:]
            else:
                res, token = outputs[0]
            timestamps.append(time.perf_counter())
            tokens.append(token)

        # TODO: ignore first token
        first_token_latency = np.round(timestamps[0] - start, 2)
        if len(timestamps) == 1:
            token_latency = np.round(timestamps[0] - start, 2)
            token = tokens[0]
        else:
            token_latency = np.round(timestamps[-1] - timestamps[0], 2)
            token = tokens[-1] - tokens[0]
        stats.append([first_token_latency, token, token_latency])
    que.put((session_id, stats))


def warmup(model,
           concurrency: int,
           input_ids: List[int],
           output_seqlen: int,
           warmup_round: int = 2,
           sampling_param=None):
    print('start to warmup ...')

    def _infer(model, session_id):
        chatbot = model.create_instance()
        for _ in range(warmup_round):
            for _ in chatbot.stream_infer(session_id,
                                          input_ids=input_ids,
                                          request_output_len=output_seqlen,
                                          sequence_start=True,
                                          sequence_end=True,
                                          ignore_eos=True,
                                          sampling_param=sampling_param):
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
                       input_seqlen: int = 0,
                       output_seqlen: int = 512,
                       test_round: int = 10,
                       tp: int = 1,
                       sampling_param=None):
    from lmdeploy.pytorch_poc.engine import Engine
    from lmdeploy.pytorch_poc.messages import SamplingParam
    from lmdeploy.turbomind.tokenizer import Tokenizer
    tokenizer_model_path = osp.join(model_path, 'triton_models', 'tokenizer')
    if os.path.exists(tokenizer_model_path):
        from lmdeploy.turbomind import TurboMind
        tokenizer = Tokenizer(tokenizer_model_path)
        tm_model = TurboMind(model_path=model_path, tp=tp)
    else:
        tokenizer = Tokenizer(model_path)
        tm_model = Engine(model_path, tp=tp)

    sampling_param = SamplingParam(
        top_k=40,
        top_p=0.8,
        temperature=0.8,
        repetition_penalty=1.0,
        ignore_eos=True,
        random_seed=random.getrandbits(64),
    )

    # make up a prompt that can be tokenized into {input_seqlen} tokens
    prompt = '' if input_seqlen == 0 else 'hi' + ' hi' * (input_seqlen - 1)
    input_ids = tokenizer.encode(prompt)

    warmup(tm_model,
           concurrency,
           input_ids,
           output_seqlen,
           sampling_param=sampling_param)

    que = Queue()
    procs = []
    _start = time.perf_counter()

    # TODO: update to the multithread version
    for i in range(concurrency):
        proc = Thread(target=infer,
                      args=(tm_model, i + 101, input_ids, output_seqlen,
                            test_round, que, sampling_param))
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
    throughput = np.sum(stats[:, 1], axis=0) / np.sum(stats[:, 2],
                                                      axis=0) * concurrency
    print(f'\n{"-" * 50}\nconcurrency: {concurrency}, input_tokens: '
          f'{input_seqlen}, output_tokens: {output_seqlen}\n'
          f'elapsed_time: {elapsed_time:.2f}s\n'
          f'first_token latency(min, max, ave): '
          f'{first_token_latency_min:.2f}s, {first_token_latency_max:.2f}s, '
          f'{first_token_latency_ave:.2f}s\ntoken latency(min, max, ave): '
          f'{token_latency_min:.2f}s, {token_latency_max:.2f}s, '
          f'{token_latency_ave:.2f}s\n'
          f'throughput: {throughput:.2f} token/s\n{"-" * 50}')
    return throughput, tm_model.gpu_count


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
    batch: int
    prompt_tokens: int
    completion_tokens: int
    throughput_per_proc: float
    throughput_per_node: float
    mem_per_proc: float
    mem_per_gpu: float
    mem_per_node: float


def parse_args():
    parser = argparse.ArgumentParser(description='Regression Test')
    parser.add_argument('--model-path',
                        type=str,
                        help='benchmark test model path')
    parser.add_argument('--concurrency',
                        nargs='+',
                        type=int,
                        help='how many requests launched concurrently',
                        default=[1, 8, 16, 32])
    parser.add_argument(
        '--prompt-tokens',
        nargs='+',
        type=int,
        help='how many requests launched concurrently. One-to-one'
        'correspondence with completion-tokens',
        default=[64, 512, 512, 1024])
    parser.add_argument('--completion-tokens',
                        nargs='+',
                        type=int,
                        help='how many tokens to be generated. One-to-one'
                        'correspondence with prompt-tokens',
                        default=[512, 512, 1024, 1024])
    parser.add_argument('--tp', type=int, help='Tensor parallel', default=1)
    parser.add_argument('--dst-csv',
                        type=str,
                        help='Where to save the result.',
                        default='profile_generation.csv')
    parser.add_argument('--log-level',
                        help='set log level',
                        default='INFO',
                        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()
    return args


def main():
    import multiprocessing as mp
    args = parse_args()
    os.environ['TM_LOG_LEVEL'] = args.log_level
    results: List[ProfileResult] = []
    for batch in tqdm(args.concurrency):
        for prompt_tokens, completion_tokens in tqdm(
                zip(args.prompt_tokens, args.completion_tokens)):
            from functools import partial
            MemoryMonitor.start()
            profile_target = partial(profile_throughput,
                                     concurrency=batch,
                                     input_seqlen=prompt_tokens,
                                     output_seqlen=completion_tokens,
                                     tp=args.tp)
            output = mp.Pool(1).map(profile_target, (args.model_path, ))
            throughput_per_proc, tp = output[0]
            time.sleep(5)  # wait a while for releasing GPU mem
            memory = MemoryMonitor.terminate()
            device_count = MemoryMonitor.device_count.value
            results.append(
                ProfileResult(batch=batch,
                              prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens,
                              throughput_per_proc=throughput_per_proc,
                              throughput_per_node=throughput_per_proc / tp *
                              device_count,
                              mem_per_proc=memory,
                              mem_per_gpu=memory / tp,
                              mem_per_node=memory / tp * device_count))
    with open(args.dst_csv, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'batch', 'prompt_tokens', 'completion_tokens',
            'throughput_per_proc(token/s)', 'throughput_per_node(token/s)',
            'mem_per_proc(GB)', 'mem_per_gpu(GB)', 'mem_per_node(GB)'
        ])
        for re in results:
            writer.writerow([
                re.batch, re.prompt_tokens, re.completion_tokens,
                f'{re.throughput_per_proc:.2f}',
                f'{re.throughput_per_node:.2f}', f'{re.mem_per_proc:.2f}',
                f'{re.mem_per_gpu:.2f}', f'{re.mem_per_node:.2f}'
            ])


if __name__ == '__main__':
    main()
