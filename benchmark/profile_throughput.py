# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import asyncio
import json
import os
import random
from queue import Queue
from typing import List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from lmdeploy.cli.utils import ArgumentHelper, DefaultsAndTypesHelpFormatter
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.profiler import Profiler, Session
from lmdeploy.tokenizer import DetokenizeState, Tokenizer
from lmdeploy.utils import get_logger

get_logger('lmdeploy').setLevel('ERROR')
os.environ['TM_LOG_LEVEL'] = 'ERROR'


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError('output_len too small')
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data['conversations']) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data['conversations'][0]['value'], data['conversations'][1]['value']) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (len(completion_token_ids) if fixed_output_len is None else fixed_output_len)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or (prompt_len + output_len > 2048 and fixed_output_len is None):
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    print(f'#Input tokens: {np.sum([x[1] for x in filtered_dataset])}')
    print(f'#Output tokens: {np.sum([x[2] for x in filtered_dataset])}')
    return filtered_dataset


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
) -> List[Tuple[str, int, int]]:

    input_lens = np.random.randint(
        max(int(input_len * range_ratio), 1),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )

    if True:
        # Sample token ids from ShareGPT and repeat/truncate them to
        # satisfy the input_lens

        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data['conversations']) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [(data['conversations'][0]['value'], data['conversations'][1]['value']) for data in dataset]
        # remove the empty prompt
        dataset = [(query, answer) for query, answer in dataset if len(query) > 0]

        # Shuffle the dataset.
        random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        input_requests: List[Tuple[str, int, int]] = []
        for i in range(num_prompts):
            # Tokenize the prompts and completions.
            prompt = dataset[i][0]
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

            if prompt_len > input_lens[i]:
                input_ids = prompt_token_ids[:input_lens[i]]
            else:
                ratio = (input_lens[i] + prompt_len - 1) // prompt_len
                input_ids = (prompt_token_ids * ratio)[:input_lens[i]]
            prompt = tokenizer.decode(input_ids)
            input_requests.append((prompt, int(input_lens[i]), int(output_lens[i])))
    else:
        # Sample token ids from random integers.
        # This can cause some NaN issues.
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        input_requests = []
        for i in range(num_prompts):
            prompt = tokenizer.decode([(offsets[i] + i + j) % tokenizer.vocab_size for j in range(input_lens[i])])
            input_requests.append((prompt, int(input_lens[i]), int(output_lens[i])))

    print(f'#Input tokens: {np.sum(input_lens)}')
    print(f'#Output tokens: {np.sum(output_lens)}')
    return input_requests


class Engine:

    def __init__(self, model_path: str, engine_config: Union[PytorchEngineConfig, TurbomindEngineConfig]):
        self.tokenizer = Tokenizer(model_path)
        if isinstance(engine_config, TurbomindEngineConfig):
            from lmdeploy.turbomind import TurboMind
            tm_model = TurboMind.from_pretrained(model_path, engine_config=engine_config)
            self.backend = 'turbomind'
        elif isinstance(engine_config, PytorchEngineConfig):
            from lmdeploy.pytorch.engine import Engine as PytorchEngine
            tm_model = PytorchEngine.from_pretrained(model_path, engine_config=engine_config)
            self.backend = 'pytorch'

        self.tm_model = tm_model
        self.pbar = None

    async def _inference(self, req_queue: Queue, session_id: int, temperature: float, top_p: float, top_k: int,
                         stream_output: bool, skip_tokenize: bool, skip_detokenize: bool, concurrency: int):
        model_inst = self.tm_model.create_instance()
        sess: Session = None
        for prompt, _, output_seqlen, cancel_after, sess in iter(req_queue.get_nowait, None):

            sess.tick(0)

            if skip_tokenize:
                input_ids = prompt
            else:
                input_ids = self.tokenizer(prompt).input_ids

            state = DetokenizeState(len(input_ids))

            n_token = 0
            token_ids = input_ids.copy()

            generator = model_inst.async_stream_infer(session_id,
                                                      input_ids=input_ids,
                                                      gen_config=GenerationConfig(max_new_tokens=output_seqlen,
                                                                                  temperature=temperature,
                                                                                  top_p=top_p,
                                                                                  top_k=top_k,
                                                                                  ignore_eos=True),
                                                      sequence_start=True,
                                                      sequence_end=True,
                                                      stream_output=stream_output)
            try:
                async for outputs in generator:
                    n_token += len(outputs.token_ids)
                    token_ids += outputs.token_ids
                    if not skip_detokenize:
                        _, state = self.tokenizer.detokenize_incrementally(token_ids, state)
                    sess.tick(n_token)
                    if n_token > cancel_after:
                        break
                sess.finish(Session.SUCCESS)
            finally:
                await generator.aclose()

            # for pytorch engine to restart a session
            if self.backend == 'pytorch':
                await model_inst.async_end(session_id)

            self.pbar.update(1)
            session_id += concurrency

    def process_request(self, requests, profiler: Profiler, concurrency, temperature, top_p, top_k, stream_output,
                        skip_tokenize, skip_detokenize, cancel_rate):
        req_queue = Queue()

        # feed request to q
        for prompt, input_len, output_len in requests:
            cancel_after = output_len + 1
            if cancel_rate > 0:
                if random.random() < cancel_rate:
                    cancel_after = random.randint(0, cancel_after)
            sess = profiler.new_session(input_len, output_len)
            req = [prompt, input_len, output_len, cancel_after, sess]
            if skip_tokenize:
                req[0] = self.tokenizer.encode(prompt)
            req_queue.put(req)
        for i in range(concurrency):
            req_queue.put(None)

        # start threads
        tasks = []
        for i in range(concurrency):
            task = self._inference(req_queue, i, temperature, top_p, top_k, stream_output, skip_tokenize,
                                   skip_detokenize, concurrency)
            tasks.append(task)

        async def _gather_tasks(tasks):
            return await asyncio.gather(*tasks)

        self.pbar = tqdm(total=len(requests))

        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

        profiler.start()

        asyncio.run(_gather_tasks(tasks))

        profiler.finish()

        self.pbar.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark the request throughput of lmdeploy '
                                     'in localhost',
                                     formatter_class=DefaultsAndTypesHelpFormatter)
    parser.add_argument('dataset', type=str, help='the path dataset')
    parser.add_argument('model_path',
                        type=str,
                        help='the path of the model in localhost or '
                        'the repo_id of the model in huggingface.co')
    parser.add_argument('-c',
                        '--concurrency',
                        type=int,
                        help='Number of working threads to process the sampled prompts',
                        default=256)
    parser.add_argument('-n', '--num-prompts', type=int, help='Number of prompts to process', default=5000)
    parser.add_argument('--no-stream-output', action='store_true', help='Use stream output')
    parser.add_argument('--skip-tokenize', action='store_true', help='Pre-tokenize input prompts before starting')
    parser.add_argument('--skip-detokenize', action='store_true', help='Skip detokenizing output tokens')
    parser.add_argument('--cancel-rate', type=float, help='Possibility of a request being canceled', default=0)
    parser.add_argument('--use-uvloop', action='store_true')
    parser.add_argument('--csv', type=str, help='Where to save the result.', default='./profile_throughput.csv')
    parser.add_argument('--seed', type=int, default=0, help='Seed used in sampling prompts from dataset')
    parser.add_argument('--distributed-executor-backend',
                        type=str,
                        default=None,
                        choices=['uni', 'mp', 'ray'],
                        help='backend of executor backend')
    parser.add_argument('--dataset-name',
                        type=str,
                        default='sharegpt',
                        choices=['sharegpt', 'random'],
                        help='Name of the dataset to benchmark on.')
    parser.add_argument(
        '--sharegpt-output-len',
        type=int,
        default=None,
        help='Output length for each request. Overrides the output length '
        'from the ShareGPT dataset.',
    )
    parser.add_argument(
        '--random-input-len',
        type=int,
        help='Number of input tokens per request, used only for random '
        'dataset.',
    )
    parser.add_argument(
        '--random-output-len',
        type=int,
        help='Number of output tokens per request, used only for random '
        'dataset.',
    )
    parser.add_argument(
        '--random-range-ratio',
        type=float,
        default=0.0,
        help='Range of sampled ratio of input/output length, '
        'used only for random dataset.',
    )
    # other args
    ArgumentHelper.top_p(parser)
    ArgumentHelper.temperature(parser)
    ArgumentHelper.top_k(parser)
    ArgumentHelper.backend(parser)

    # pytorch engine args
    pt_group = parser.add_argument_group('PyTorch engine arguments')
    ArgumentHelper.eager_mode(pt_group)
    ArgumentHelper.dllm_block_length(pt_group)
    ArgumentHelper.dllm_unmasking_strategy(pt_group)
    ArgumentHelper.dllm_denoising_steps(pt_group)
    ArgumentHelper.dllm_confidence_threshold(pt_group)

    tp_act = ArgumentHelper.tp(pt_group)
    cache_count_act = ArgumentHelper.cache_max_entry_count(pt_group)
    cache_block_seq_len_act = ArgumentHelper.cache_block_seq_len(pt_group)
    prefix_caching_act = ArgumentHelper.enable_prefix_caching(pt_group)
    quant_policy_act = ArgumentHelper.quant_policy(pt_group, default=0)
    dtype_act = ArgumentHelper.dtype(pt_group)

    # turbomind engine args
    tb_group = parser.add_argument_group('TurboMind engine argument')
    tb_group._group_actions.append(tp_act)
    tb_group._group_actions.append(cache_count_act)
    tb_group._group_actions.append(cache_block_seq_len_act)
    tb_group._group_actions.append(prefix_caching_act)
    tb_group._group_actions.append(quant_policy_act)
    tb_group._group_actions.append(dtype_act)

    ArgumentHelper.dp(tb_group)
    ArgumentHelper.model_format(tb_group, default='hf')
    ArgumentHelper.num_tokens_per_iter(tb_group)
    ArgumentHelper.max_prefill_iters(tb_group)
    ArgumentHelper.communicator(tb_group)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    if args.backend == 'turbomind':
        engine_config = TurbomindEngineConfig(
            max_batch_size=args.concurrency // args.dp,
            tp=args.tp,
            dp=args.dp,
            cache_max_entry_count=args.cache_max_entry_count,
            cache_block_seq_len=args.cache_block_seq_len,
            model_format=args.model_format,
            quant_policy=args.quant_policy,
            num_tokens_per_iter=args.num_tokens_per_iter,
            max_prefill_iters=args.max_prefill_iters,
            enable_prefix_caching=args.enable_prefix_caching,
            dtype=args.dtype,
            communicator=args.communicator,
        )
    elif args.backend == 'pytorch':
        engine_config = PytorchEngineConfig(
            cache_max_entry_count=args.cache_max_entry_count,
            block_size=args.cache_block_seq_len,
            max_batch_size=args.concurrency,
            tp=args.tp,
            eager_mode=args.eager_mode,
            enable_prefix_caching=args.enable_prefix_caching,
            quant_policy=args.quant_policy,
            dtype=args.dtype,
            distributed_executor_backend=args.distributed_executor_backend,
            dllm_block_length=args.dllm_block_length,
            dllm_unmasking_strategy=args.dllm_unmasking_strategy,
            dllm_denoising_steps=args.dllm_denoising_steps,
            dllm_confidence_threshold=args.dllm_confidence_threshold,
        )

    if args.use_uvloop:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    engine = Engine(args.model_path, engine_config)

    if args.dataset_name == 'sharegpt':
        assert args.random_input_len is None and args.random_output_len is None
        requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=engine.tokenizer.model.model,
            fixed_output_len=args.sharegpt_output_len,
        )
    elif args.dataset_name == 'random':
        assert args.random_input_len is not None and \
            args.random_output_len is not None
        requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=engine.tokenizer.model.model,
            dataset_path=args.dataset,
        )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset_name}')

    stream_output = not args.no_stream_output

    profiler = Profiler(stream_output, [50, 75, 95, 99])

    engine.process_request(requests,
                           profiler,
                           temperature=args.temperature,
                           top_p=args.top_p,
                           top_k=args.top_k,
                           concurrency=args.concurrency if args.concurrency < args.num_prompts else args.num_prompts,
                           stream_output=not args.no_stream_output,
                           skip_tokenize=args.skip_tokenize,
                           skip_detokenize=args.skip_detokenize,
                           cancel_rate=args.cancel_rate)

    hyperparams = [('Concurrency', args.concurrency), ('Cancel rate', args.cancel_rate),
                   ('Stream output', str(stream_output).lower()), ('Skip tokenize', str(args.skip_tokenize).lower()),
                   ('Skip detokenize', str(args.skip_detokenize).lower())]
    profiler.compute_metrics()
    profiler.summarize(title='Profile Throughput', hyperparams=hyperparams)
    if args.csv:
        profiler.save_csv(args.csv, (
            ('backend', args.backend),
            ('bs', args.concurrency),
            ('dataset_name', args.dataset_name),
            ('sharegpt_output_len', args.sharegpt_output_len),
            ('random_input_len', args.random_input_len),
            ('random_output_len', args.random_output_len),
            ('random_range_ratio', args.random_range_ratio),
            ('num_prompts', args.num_prompts),
        ))


if __name__ == '__main__':
    main()
