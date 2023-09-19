import json
import os.path as osp
import random
import time
from queue import Queue
from threading import Thread
from typing import List, Tuple

import fire

from lmdeploy.turbomind import Tokenizer, TurboMind


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: Tokenizer,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    print('load test data...')
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data['conversations']) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data['conversations'][0]['value'],
                data['conversations'][1]['value']) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    start = time.perf_counter()
    prompt_token_ids = tokenizer(prompts).input_ids
    end = time.perf_counter()
    print(f'tokenizer encode prompts cost: {end-start} s, '
          f'len(prompts): {len(prompts)}')

    completions = [completion for _, completion in dataset]
    start = time.perf_counter()
    completion_token_ids = tokenizer(completions).input_ids
    end = time.perf_counter()
    print(f'tokenizer encode completions cost: {end-start} s, '
          f'len(prompts): {len(completions)}')

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
    print('load test data successfully')
    return sampled_requests


class Engine:

    def __init__(self, model_path: str, tp: int = 1):
        tokenizer_model_path = osp.join(model_path, 'triton_models',
                                        'tokenizer')
        tokenizer = Tokenizer(tokenizer_model_path)
        tm_model = TurboMind(model_path=model_path, tp=tp)
        self.tm_model = tm_model
        self.tokenizer = tokenizer

    def _inference(self, queue, session_id: int):

        model_inst = self.tm_model.create_instance()
        while True:
            request = queue.get()
            if request is None:
                # stop signal
                queue.put(None)
                return
            else:
                prompt, _, output_seqlen = request
                input_ids = self.tokenizer.encode(prompt)

                for outputs in model_inst.stream_infer(
                        session_id,
                        input_ids=input_ids,
                        request_output_len=output_seqlen,
                        temperature=1.0,
                        top_p=1.0,
                        sequence_start=True,
                        sequence_end=True,
                        ignore_eos=True):
                    res, tokens = outputs[0]
                    self.tokenizer.decode(res)

    def process_request(self, requests, concurrency: int = 1):
        q = Queue()
        threads = []

        start = time.time()

        # start threads
        for i in range(concurrency):
            t = Thread(target=self._inference, args=(q, i))
            t.start()
            threads.append(t)

        # feed request to q
        for req in requests:
            q.put(req)

        q.put(None)

        # wait for finish
        for t in threads:
            t.join()

        end = time.time()

        return end - start


def warmup(model, tokenizer, concurrency: int, warmup_round: int = 2):
    print('start to warmup ...')

    # make up a prompt that can be tokenized into {input_seqlen} tokens
    prompt = 'hi ' * 128
    input_ids = tokenizer.encode(prompt)

    def _infer(model, session_id):
        chatbot = model.create_instance()
        for _ in range(warmup_round):
            for _ in chatbot.stream_infer(session_id,
                                          input_ids=input_ids,
                                          request_output_len=512,
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


def main(dataset: str,
         model_path: str,
         concurrency: int = 1,
         num_prompts: int = 1000,
         tp: int = 1):

    engine = Engine(model_path, tp=tp)
    tokenizer = engine.tokenizer
    print('load model successfully')
    warmup(engine.tm_model, tokenizer, concurrency)

    requests = sample_requests(dataset, num_prompts, tokenizer)

    elapsed_time = engine.process_request(requests, concurrency)
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    total_num_out_tokens = sum(output_len for _, _, output_len in requests)
    print(f'Throughput requests: {len(requests) / elapsed_time:.2f} req/s')
    print(
        f'Throughput requests: {len(requests) * 60 / elapsed_time:.2f} req/min'
    )
    print(f'Throughput tokens: {total_num_tokens / elapsed_time:.2f} tokens/s')
    print('Throughput tokens(output only):'
          f'{total_num_out_tokens / elapsed_time:.2f} tokens/s')


if __name__ == '__main__':
    fire.Fire(main)
