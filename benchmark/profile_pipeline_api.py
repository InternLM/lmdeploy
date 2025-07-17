# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from lmdeploy.cli.utils import ArgumentHelper, DefaultsAndTypesHelpFormatter
from lmdeploy.profiler import Profiler, Session
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


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

    def __init__(self, model_path: str, engine_config, csv: str):
        self.pipe = pipeline(model_path, backend_config=engine_config, log_level='ERROR')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.csv = csv

    def process_request(self, requests, profiler: Profiler, temperature, top_p, top_k, stream_output):

        prompts = [prompt for prompt, _, _ in requests]
        gen_configs = [
            GenerationConfig(temperature=temperature,
                             top_p=top_p,
                             top_k=top_k,
                             ignore_eos=True,
                             do_sample=False,
                             max_new_tokens=output_len) for _, _, output_len in requests
        ]

        sess: List[Session] = []
        for _, input_len, output_len in requests:
            sess.append(profiler.new_session(input_len, output_len))

        def _to_status(finish_reason):
            if finish_reason == 'length':
                return Session.SUCCESS
            else:
                return Session.FAIL

        profiler.start()

        for s in sess:
            s.tick(0)

        if stream_output:
            pbar = tqdm(total=len(requests))
            for output in self.pipe.stream_infer(prompts, gen_configs, do_preprocess=False):
                index = output.index
                n_token = output.generate_token_len
                finish_reason = output.finish_reason
                sess[index].tick(n_token)
                if finish_reason is not None:
                    sess[index].finish(_to_status(finish_reason))
                    pbar.update(1)
            pbar.close()
        else:
            for output in self.pipe(prompts, gen_configs, do_preprocess=False, use_tqdm=True):
                index = output.index
                n_token = output.generate_token_len
                finish_reason = output.finish_reason
                sess[index].tick(n_token)
                sess[index].finish(_to_status(finish_reason))

        profiler.finish()

        # report first failure
        for i, s in enumerate(sess):
            if s.status != Session.SUCCESS or s.ns[-1] < s.req_output_len:
                logger.error(f'Request {i} failed with {s.ns[-1]}/{s.req_output_len} tokens generated'  # noqa: E501
                             )
                logger.error(f'Prompt: {prompts[i]}')
                logger.warning('Got failed requests, metrics may be invalid')
                break


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
    parser.add_argument('--csv', type=str, help='Where to save the result.', default='./profile_pipeline_api.csv')
    parser.add_argument('--seed', type=int, default=0, help='Seed used in sampling prompts from dataset')
    parser.add_argument('--stream-output', action='store_true', help='Trust remote code for loading hf models')
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
    ArgumentHelper.log_level(parser)
    ArgumentHelper.backend(parser)

    # pytorch engine args
    pt_group = parser.add_argument_group('PyTorch engine arguments')
    ArgumentHelper.eager_mode(pt_group)

    tp_act = ArgumentHelper.tp(pt_group)
    cache_count_act = ArgumentHelper.cache_max_entry_count(pt_group)
    cache_block_seq_len_act = ArgumentHelper.cache_block_seq_len(pt_group)
    prefix_caching_act = ArgumentHelper.enable_prefix_caching(pt_group)

    # turbomind engine args
    tb_group = parser.add_argument_group('TurboMind engine argument')
    tb_group._group_actions.append(tp_act)
    tb_group._group_actions.append(cache_count_act)
    tb_group._group_actions.append(cache_block_seq_len_act)
    tb_group._group_actions.append(prefix_caching_act)
    ArgumentHelper.model_format(tb_group, default='hf')
    ArgumentHelper.quant_policy(tb_group, default=0)
    ArgumentHelper.num_tokens_per_iter(tb_group)
    ArgumentHelper.max_prefill_iters(tb_group)
    ArgumentHelper.communicator(tb_group)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    os.environ['TM_LOG_LEVEL'] = args.log_level
    if args.backend == 'turbomind':
        engine_config = TurbomindEngineConfig(
            max_batch_size=args.concurrency,
            tp=args.tp,
            cache_max_entry_count=args.cache_max_entry_count,
            cache_block_seq_len=args.cache_block_seq_len,
            model_format=args.model_format,
            quant_policy=args.quant_policy,
            num_tokens_per_iter=args.num_tokens_per_iter,
            max_prefill_iters=args.max_prefill_iters,
            enable_prefix_caching=args.enable_prefix_caching,
            communicator=args.communicator,
        )
    elif args.backend == 'pytorch':
        engine_config = PytorchEngineConfig(
            cache_max_entry_count=args.cache_max_entry_count,
            block_size=args.cache_block_seq_len,
            max_batch_size=args.concurrency,
            tp=args.tp,
            thread_safe=False,
            eager_mode=args.eager_mode,
            enable_prefix_caching=args.enable_prefix_caching,
        )

    engine = Engine(args.model_path, engine_config, csv=args.csv)

    profiler = Profiler(args.stream_output, [50, 75, 95, 99])

    if args.dataset_name == 'sharegpt':
        assert args.random_input_len is None and args.random_output_len is None
        requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=engine.tokenizer,
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
            tokenizer=engine.tokenizer,
            dataset_path=args.dataset,
        )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset_name}')

    engine.process_request(requests,
                           profiler,
                           temperature=args.temperature,
                           top_p=args.top_p,
                           top_k=args.top_k,
                           stream_output=args.stream_output)

    hyperparams = [('Concurrency', args.concurrency), ('Stream output', str(args.stream_output).lower())]

    profiler.compute_metrics()
    profiler.summarize(title='Profile Pipeline API', hyperparams=hyperparams)

    if args.csv:
        # profiler.save_csv(args.csv, (('batch', args.concurrency), ('num_prompts', args.num_prompts)))
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
