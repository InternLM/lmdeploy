# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import random
from typing import List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from lmdeploy.cli.utils import ArgumentHelper, DefaultsAndTypesHelpFormatter
from lmdeploy.profiler import Profiler, Session
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def sample_requests(dataset_path: str, num_requests: int, tokenizer) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data['conversations']) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data['conversations'][0]['value'], data['conversations'][1]['value']) for data in dataset]
    # remove the empty prompts
    dataset = [(query, answer) for query, answer in dataset if len(query) > 0]
    # pre-sample to avoid go through all the dataset
    dataset = random.sample(dataset, max(int(num_requests * 1.2), 1000))

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


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
    session_len_act = ArgumentHelper.session_len(pt_group, default=4096)
    cache_count_act = ArgumentHelper.cache_max_entry_count(pt_group)
    cache_block_seq_len_act = ArgumentHelper.cache_block_seq_len(pt_group)
    prefix_caching_act = ArgumentHelper.enable_prefix_caching(pt_group)

    # turbomind engine args
    tb_group = parser.add_argument_group('TurboMind engine argument')
    tb_group._group_actions.append(tp_act)
    tb_group._group_actions.append(session_len_act)
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
            session_len=args.session_len,
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
            session_len=args.session_len,
            cache_max_entry_count=args.cache_max_entry_count,
            block_size=args.cache_block_seq_len,
            max_batch_size=args.concurrency,
            tp=args.tp,
            thread_safe=False,
            eager_mode=args.eager_mode,
            enable_prefix_caching=args.enable_prefix_caching,
        )

    engine = Engine(args.model_path, engine_config, csv=args.csv)

    requests = sample_requests(args.dataset, args.num_prompts, engine.tokenizer)

    profiler = Profiler(args.stream_output, [50, 75, 95, 99])

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
        profiler.save_csv(args.csv, (('batch', args.concurrency), ('num_prompts', args.num_prompts)))


if __name__ == '__main__':
    main()
