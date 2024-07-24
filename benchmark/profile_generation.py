# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import csv
import os
import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import List, Union

import numpy as np
from pynvml import (NVMLError, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetMemoryInfo, nvmlDeviceGetName,
                    nvmlDeviceGetPowerState, nvmlDeviceGetTemperature,
                    nvmlInit, nvmlShutdown, nvmlSystemGetDriverVersion)
from tqdm import tqdm

from lmdeploy.cli.utils import ArgumentHelper, DefaultsAndTypesHelpFormatter
from lmdeploy.messages import (EngineGenerationConfig, PytorchEngineConfig,
                               TurbomindEngineConfig)
from lmdeploy.utils import get_logger

get_logger('lmdeploy').setLevel('ERROR')
os.environ['TM_LOG_LEVEL'] = 'ERROR'


def infer(model, session_id: int, input_ids: List,
          gen_config: EngineGenerationConfig, test_round: int, que: Queue):
    if session_id == 1:
        pbar = tqdm(total=test_round)
    chatbot = model.create_instance()
    output_seqlen = gen_config.max_new_tokens
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
        for outputs in chatbot.stream_infer(session_id,
                                            input_ids,
                                            gen_config=gen_config,
                                            sequence_start=True,
                                            sequence_end=True,
                                            stream_output=True):
            n_token = outputs.num_token
            now = time.perf_counter()
            if n_prev_token != n_token:
                token_latency_stats[n_prev_token] = np.round(now - prev, 3)
                n_prev_token = n_token
            prev = now
        # for pytorch engine to restart a session
        if hasattr(chatbot, 'end'):
            chatbot.end(session_id)
        if session_id == 1:
            pbar.update(1)

        assert output_seqlen <= n_token <= output_seqlen + 1, \
            f'Error. session_id({session_id}) request {output_seqlen} ' \
            f'tokens, but generate {n_token} tokens'
        stats.append(token_latency_stats[:output_seqlen])
    que.put((session_id, stats))


def warmup(model, concurrency: int, input_ids: List[int], warmup_round: int,
           gen_config: EngineGenerationConfig):
    if not warmup_round:
        return

    print('start to warmup ...')
    output_seqlen = gen_config.max_new_tokens

    def _infer(model, session_id):
        chatbot = model.create_instance()
        for _ in range(warmup_round):
            for _ in chatbot.stream_infer(session_id,
                                          input_ids=input_ids,
                                          request_output_len=output_seqlen,
                                          sequence_start=True,
                                          sequence_end=True,
                                          ignore_eos=True,
                                          gen_config=gen_config):
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
                       engine_config: Union[PytorchEngineConfig,
                                            TurbomindEngineConfig],
                       gen_config: EngineGenerationConfig, test_round: int,
                       warmup_round: int):
    output_seqlen = gen_config.max_new_tokens
    print(f'profiling ... concurrency: {concurrency}, '
          f'n_prompt_token: {input_seqlen}, '
          f'n_completion_token: {output_seqlen}, '
          f'test_round: {test_round}, warmup_round: {warmup_round}')
    if isinstance(engine_config, TurbomindEngineConfig):
        from lmdeploy.turbomind import TurboMind
        tm_model = TurboMind.from_pretrained(model_path,
                                             engine_config=engine_config)
    elif isinstance(engine_config, PytorchEngineConfig):
        from lmdeploy.pytorch.engine import Engine
        tm_model = Engine(model_path, engine_config)

    # make up a dummy `input_ids` with the length of `input_seqlen` exactly
    assert input_seqlen > 0, 'input_seqlen should > 0'
    input_ids = np.random.randint(low=0, high=101, size=input_seqlen).tolist()
    warmup(tm_model, concurrency, input_ids, warmup_round, gen_config)

    que = Queue()
    procs = []
    _start = time.perf_counter()

    for i in range(concurrency):
        proc = Thread(target=infer,
                      args=(tm_model, i + 1, input_ids, gen_config, test_round,
                            que))
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
    if output_seqlen > 1:
        # sort token_latency without the first token's latency
        sorted_token_latency = np.sort(token_latency_stats[:, 1:].flatten())
        percentiles = [
            np.round(
                sorted_token_latency[int(percent * len(sorted_token_latency))],
                3) for percent in [0.5, 0.75, 0.95, 0.99]
        ]
    else:
        percentiles = [
            first_token_latency_ave,
        ] * 4

    out_token_throughput = np.round(token_latency_stats.size / elapsed_time, 2)
    total_token_throughput = np.round(
        concurrency * test_round * (input_seqlen + output_seqlen) /
        elapsed_time, 2)
    print(f'\n{"-" * 50}\ntotal time: {elapsed_time:.2f}s\n'
          f'concurrency: {concurrency}, test_round: {test_round}\n'
          f'input_tokens: {input_seqlen}, output_tokens: {output_seqlen}\n'
          f'first_token latency(min, max, ave): '
          f'{first_token_latency_min}s, {first_token_latency_max}s, '
          f'{first_token_latency_ave}s\ntotal_token latency(min, max, ave): '
          f'{token_latency_min}s, {token_latency_max}s, '
          f'{token_latency_ave}s\n'
          f'token_latency percentiles(50%,75%,95%,99%)(s): {percentiles}\n'
          f'throughput(output): {out_token_throughput} token/s\n'
          f'throughput(total): {total_token_throughput} token/s\n{"-" * 50}')
    return tm_model.model_name, \
        [first_token_latency_min, first_token_latency_max,
         first_token_latency_ave], \
        percentiles, out_token_throughput, total_token_throughput, \
        tm_model.gpu_count


class MemoryMonitor:

    @classmethod
    def init(cls):
        from multiprocessing import Manager
        cls.max_mem = Manager().Value('f', 0)  # GB
        cls.device_count = Manager().Value('f', 0)

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
    output_throughput: float
    total_throughput: float
    mem_per_gpu: float


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile the token generation performance with'
        ' pytorch or turbomind engine',
        formatter_class=DefaultsAndTypesHelpFormatter)
    parser.add_argument('model_path',
                        type=str,
                        help='the path of the model in localhost or '
                        'the repo_id of the model in huggingface.co')
    parser.add_argument('-c',
                        '--concurrency',
                        nargs='+',
                        type=int,
                        help='how many requests launched concurrently',
                        default=[1, 16, 32, 64])
    parser.add_argument(
        '-pt',
        '--prompt-tokens',
        nargs='+',
        type=int,
        help='how many requests launched concurrently. One-to-one '
        'correspondence with completion-tokens',
        default=[1, 128, 128, 2048, 2048])
    parser.add_argument('-ct',
                        '--completion-tokens',
                        nargs='+',
                        type=int,
                        help='how many tokens to be generated. One-to-one'
                        'correspondence with prompt-tokens',
                        default=[128, 128, 2048, 128, 2048])
    parser.add_argument('--csv',
                        type=str,
                        help='Where to save the result.',
                        default='profile_generation.csv')
    parser.add_argument('-tr',
                        '--test-round',
                        type=int,
                        help='number of test rounds',
                        default=3)
    parser.add_argument('-w',
                        '--warmup-round',
                        type=int,
                        help='number of warmup rounds',
                        default=1)

    # other args
    ArgumentHelper.top_p(parser)
    ArgumentHelper.temperature(parser)
    ArgumentHelper.top_k(parser)
    ArgumentHelper.backend(parser)
    # pytorch engine args
    pt_group = parser.add_argument_group('PyTorch engine arguments')
    tp_act = ArgumentHelper.tp(pt_group)
    cache_count_act = ArgumentHelper.cache_max_entry_count(pt_group)
    cache_block_seq_len_act = ArgumentHelper.cache_block_seq_len(pt_group)
    session_len_act = ArgumentHelper.session_len(pt_group, default=2048)
    prefix_caching_act = ArgumentHelper.enable_prefix_caching(pt_group)
    rope_scaling_factor_act = ArgumentHelper.rope_scaling_factor(pt_group)

    # turbomind engine args
    tb_group = parser.add_argument_group('TurboMind engine argument')
    tb_group._group_actions.append(tp_act)
    tb_group._group_actions.append(session_len_act)
    tb_group._group_actions.append(cache_count_act)
    tb_group._group_actions.append(cache_block_seq_len_act)
    tb_group._group_actions.append(prefix_caching_act)
    tb_group._group_actions.append(rope_scaling_factor_act)
    ArgumentHelper.model_format(tb_group, default='hf')
    args = parser.parse_args()
    return args


def __proc_cb(*args, ret_pipe, target):
    try:
        ret = target(*args)
        ret_pipe[1].send(ret)
    except Exception as e:
        ret_pipe[1].send(e)


def _process_map(target, iterable):
    from multiprocessing import Pipe, get_context

    pipe = Pipe(False)
    spawn_context = get_context('spawn')
    proc = spawn_context.Process(target=__proc_cb,
                                 args=iterable,
                                 kwargs=dict(ret_pipe=pipe, target=target))
    proc.start()
    proc.join()

    ret = pipe[0].recv()
    if isinstance(ret, Exception):
        raise ret

    return ret


def main():
    args = parse_args()
    assert len(args.prompt_tokens) == len(args.completion_tokens), \
        f'mismatched size between `prompt-tokens` and `completion-tokenes`' \
        f', {len(args.prompt_tokens)} vs {len(args.completion_tokens)}'

    results: List[ProfileResult] = []

    MemoryMonitor.init()
    for batch in args.concurrency:
        for prompt_tokens, completion_tokens in zip(args.prompt_tokens,
                                                    args.completion_tokens):
            MemoryMonitor.start()
            from functools import partial

            # make sure session_len >= prompt_tokens + completion_tokens
            session_len = max(args.session_len,
                              prompt_tokens + completion_tokens)
            if args.backend == 'turbomind':
                engine_config = TurbomindEngineConfig(
                    cache_max_entry_count=args.cache_max_entry_count,
                    cache_block_seq_len=args.cache_block_seq_len,
                    model_format=args.model_format,
                    session_len=session_len,
                    rope_scaling_factor=args.rope_scaling_factor,
                    tp=args.tp,
                    enable_prefix_caching=args.enable_prefix_caching,
                )
            elif args.backend == 'pytorch':
                engine_config = PytorchEngineConfig(
                    cache_max_entry_count=args.cache_max_entry_count,
                    block_size=args.cache_block_seq_len,
                    session_len=session_len,
                    tp=args.tp,
                    thread_safe=True,
                    enable_prefix_caching=args.enable_prefix_caching,
                )
            gen_config = EngineGenerationConfig(
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                max_new_tokens=completion_tokens,
                ignore_eos=True)
            profile_target = partial(
                profile_throughput,
                concurrency=batch,
                input_seqlen=prompt_tokens,
                engine_config=engine_config,
                gen_config=gen_config,
                test_round=args.test_round,
                warmup_round=args.warmup_round,
            )
            output = _process_map(profile_target, (args.model_path, ))
            model_name, first_token_latency, percentiles, \
                output_throughput, total_throughput, tp = output
            time.sleep(5)  # wait a while for releasing GPU mem
            memory = MemoryMonitor.terminate()
            results.append(
                ProfileResult(model_name=model_name,
                              batch=batch,
                              prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens,
                              first_token_latency=first_token_latency,
                              percentiles=percentiles,
                              output_throughput=output_throughput,
                              total_throughput=total_throughput,
                              mem_per_gpu=memory / tp))
    if args.csv:
        with open(args.csv, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'batch',
                'prompt_tokens',
                'completion_tokens',
                'throughput(total tok/s)',
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
                    f'{re.total_throughput:.2f}',
                    f'{re.output_throughput:.2f}', f'{re.mem_per_gpu:.2f}',
                    re.first_token_latency[2], re.first_token_latency[0],
                    re.first_token_latency[1], re.percentiles[0],
                    re.percentiles[1], re.percentiles[2], re.percentiles[3]
                ])


if __name__ == '__main__':
    main()
