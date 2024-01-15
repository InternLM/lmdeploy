# Copyright (c) OpenMMLab. All rights reserved.
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


def infer(model, session_id: int, input_ids: List, output_seqlen: int,
          top_k: int, top_p: float, temperature: float, test_round: int,
          que: Queue):
    from lmdeploy.messages import EngineGenerationConfig

    if session_id == 1:
        pbar = tqdm(total=test_round)
    chatbot = model.create_instance()
    stats = []
    for _ in range(test_round):
        token_latency_stats = [0] * (output_seqlen + 1)
        prev = time.perf_counter()
        n_prev_token = 0

        """
        The iterator provided by `stream_infer` denotes the number of generated tokens so far,
        which is represented by the variable `n_token`.
        Please note that `n_token` is not a continuous value. In other words, during the iteration,
        its value might be 5, 7, 8, 16, and so on, rather than 1, 2, 3, 4, etc.
        So, it is quite difficult to get the latency of each generated token.
        As a work-around, we set the latency `now-prev` of each iteration to the first token of
        the new generated tokens, and leave the latency of the rest tokens being 0.
        For example, in the first iteration, 5 tokens are generated.
        The time elapsing in this iteration `now-prev` is set to the latency of first token of
        the 5 tokens, i.e. `token_latency_stats[0]`, and `token_latency_stats[1:4]` is set 0`
        """   # noqa: E501
        # TODO: use same inference interface
        gen_config = EngineGenerationConfig(max_new_tokens=output_seqlen,
                                            top_k=top_k,
                                            top_p=top_p,
                                            temperature=temperature,
                                            ignore_eos=True)
        for outputs in chatbot.stream_infer(session_id,
                                            input_ids=input_ids,
                                            gen_config=gen_config):
            _, n_token = outputs[-2:]
            now = time.perf_counter()
            if n_prev_token != n_token:
                token_latency_stats[n_prev_token] = np.round(now - prev, 3)
                n_prev_token = n_token
            prev = now
        chatbot.end(session_id)
        if session_id == 1:
            pbar.update(1)

        assert output_seqlen <= n_token <= output_seqlen + 1, \
            f'Error. session_id({session_id}) request {output_seqlen} ' \
            f'tokens, but generate {n_token} tokens'
        stats.append(token_latency_stats[:output_seqlen])
    que.put((session_id, stats))


def warmup(model, concurrency: int, input_ids: List[int], output_seqlen: int,
           warmup_round: int):
    if not warmup_round:
        return

    print('start to warmup ...')

    def _infer(model, session_id):
        from lmdeploy.messages import EngineGenerationConfig
        chatbot = model.create_instance()
        for _ in range(warmup_round):
            # TODO: use same inference interface
            gen_config = EngineGenerationConfig(max_new_tokens=output_seqlen,
                                                top_k=1,
                                                top_p=1.0,
                                                temperature=0.8,
                                                repetition_penalty=1.0,
                                                ignore_eos=True)
            generator = chatbot.stream_infer(session_id,
                                             input_ids=input_ids,
                                             gen_config=gen_config)
            for _ in generator:
                continue
            # for pytorch engine to restart a session
            if hasattr(chatbot, 'end'):
                chatbot.end(session_id)

    _start = time.perf_counter()
    procs = []
    for i in range(concurrency):
        proc = Thread(target=_infer, args=(model, i + 1), daemon=True)
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    _end = time.perf_counter()
    print(f'end warmup, elapsed time: {round(_end - _start, 2)}s')


def profile_throughput(model_path: str, concurrency: int, input_seqlen: int,
                       output_seqlen: int, tp: int, top_k: int, top_p: float,
                       temperature: float, test_round: int, warmup_round: int,
                       **kwargs):

    print(f'profiling ... concurrency: {concurrency}, '
          f'n_prompt_token: {input_seqlen}, '
          f'n_completion_token: {output_seqlen}, '
          f'test_round: {test_round}, warmup_round: {warmup_round}')

    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.pytorch.engine import Engine

    tm_model = Engine(model_path, PytorchEngineConfig(model_name='llama',
                                                      tp=tp))

    # make up a dummy `input_ids` with the length of `input_seqlen` exactly
    assert input_seqlen > 0, 'input_seqlen should > 0'
    input_ids = np.random.randint(low=0, high=101, size=input_seqlen).tolist()
    warmup(tm_model, concurrency, input_ids, output_seqlen, warmup_round)

    que = Queue()
    procs = []
    _start = time.perf_counter()

    for i in range(concurrency):
        proc = Thread(target=infer,
                      args=(tm_model, i + 1, input_ids, output_seqlen, top_k,
                            top_p, temperature, test_round, que),
                      daemon=True)
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

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
        cls.proc = Process(target=cls.mem_monitor, daemon=True)
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
                        default=1.0)
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
                        default=6)
    parser.add_argument('--warmup-round',
                        type=int,
                        help='number of warmuop rounds',
                        default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert len(args.prompt_tokens) == len(args.completion_tokens), \
        f'mismatched size between `prompt-tokens` and `completion-tokenes`' \
        f', {len(args.prompt_tokens)} vs {len(args.completion_tokens)}'

    os.environ['TM_LOG_LEVEL'] = args.log_level
    results: List[ProfileResult] = []
    for batch in args.concurrency:
        for prompt_tokens, completion_tokens in zip(args.prompt_tokens,
                                                    args.completion_tokens):
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
                                     test_round=args.test_round,
                                     warmup_round=args.warmup_round)
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
    if args.csv:
        with open(args.csv, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'batch',
                'prompt_tokens',
                'completion_tokens',
                'throughput(out tok/s)',
                'mem(GB)',
                'FTL(ave)(s)',
                'FTL(min)(s)',
                'FTL(max)(s)',
                '50%(s)',
                '75%(s)',
                '95%(s)',
                '99%(s)',
            ])
            for re in results:
                writer.writerow([
                    re.batch, re.prompt_tokens, re.completion_tokens,
                    f'{re.throughput_per_proc:.2f}', f'{re.mem_per_gpu:.2f}',
                    re.first_token_latency[2], re.first_token_latency[0],
                    re.first_token_latency[1], re.percentiles[0],
                    re.percentiles[1], re.percentiles[2], re.percentiles[3]
                ])


if __name__ == '__main__':
    main()
