# Copyright (c) OpenMMLab. All rights reserved.
import json
import pickle
import queue
import time
import warnings
from typing import List, Optional

import numpy as np
import pynvml
import torch
# import multiprocessing as mp
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase)

from .model import accel_model, init_model


def safe_numel(free_mem, model_size, max_intermediate):
    return int(free_mem - model_size) // max_intermediate


def avail_gpus(percentage=0.96):
    gpus = []
    mems = []
    pynvml.nvmlInit()
    for i in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(i))
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free, total = int(mem_info.free), int(mem_info.total)
        # print(free, total)
        # free, total = torch.cuda.mem_get_info(i)
        # However, this will allocate 500MB memory on each gpus
        if free / total > percentage:
            gpus.append(i)
            mems.append(free)
    pynvml.nvmlShutdown()
    return gpus, sum(mems) / len(mems)


@torch.no_grad()
def decode_single(model: PreTrainedModel,
                  input_ids: torch.Tensor,
                  attention_mask: torch.Tensor = None):
    # input_ids = input_ids.cuda()
    output = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   output_hidden_states=False,
                   output_attentions=False,
                   use_cache=False,
                   return_dict=True)
    logits = output.logits
    # fp32, [bs, seq_len, vocab_size]
    torch.softmax(logits, dim=-1, out=logits)
    # inplace to save memory

    # print(input_ids)
    # print(attention_mask)

    shift_labels = input_ids[..., 1:].contiguous()
    shift_probs = logits[..., :-1, :].contiguous()
    probs = torch.gather(shift_probs, -1, shift_labels.unsqueeze(-1))

    probs = probs.squeeze(-1)

    if attention_mask is not None:
        probs *= attention_mask[..., 1:]

    probs = probs.cpu()

    return probs


def worker_fn(model_path: str,
              inq: mp.Queue,
              outq: mp.Queue,
              accel: Optional[str] = None,
              gpu_id=0):
    torch.set_default_device(gpu_id)
    model, _ = init_model(model_path)
    model = model.eval()
    # model = accel_model(model, accel)

    while True:
        try:
            idx, inputs = inq.get(timeout=1)
        except queue.Empty:
            continue
        if inputs is None:
            break

        input_ids = inputs['input_ids'].cuda(gpu_id)
        attention_mask = inputs['attention_mask'].cuda(gpu_id)
        try:
            probs = decode_single(model, input_ids, attention_mask)
        except torch.cuda.OutOfMemoryError as e:
            warnings.warn(
                f'OOM on GPU {gpu_id}, discard prompts at indics {idx}.')
            probs = torch.empty((input_ids.size(0), 0),
                                dtype=torch.float32,
                                device='cpu')
        outq.put((idx, probs))


class Engine:

    def __init__(self,
                 model_path: str,
                 tokenizer_path: Optional[str] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 accel: Optional[str] = None):

        gpu_ids, mem = avail_gpus()
        print(f'Available GPUs are: {gpu_ids}, ', end='')
        print(f'with {mem/2**30:.2f} GiB free.')

        ctx = mp.get_context('spawn')
        inq = ctx.Queue()
        outq = ctx.Queue()

        ps = []
        for id in gpu_ids:
            p = ctx.Process(target=worker_fn,
                            args=(model_path, inq, outq, accel, id))
            p.start()
            ps.append(p)

        if tokenizer is None:

            if tokenizer_path is None:
                tokenizer_path = model_path

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.gpu_ids = gpu_ids
        self.inq = inq
        self.outq = outq
        self.ps = ps
        self.tokenizer = tokenizer
        self.safe_numel = safe_numel(mem, 14e9, 2e6)

    def clear_queue(self):
        for q in self.inq, self.outq:
            while not q.empty():
                q.get()

    def decode(self,
               prompts: List[str],
               sort=False,
               max_bs: int = 1024,
               pad=False):
        """Inference the model to compute probabilities."""

        self.clear_queue()

        # sort to achieve better efficiency
        if sort:
            prompts_and_indicis = sorted(enumerate(prompts),
                                         key=lambda i_and_x: len(i_and_x[1]))
        else:
            prompts_and_indicis = list(enumerate(prompts))

        left = 0
        bs = max_bs

        while left < len(prompts):

            if not sort:
                bs = max_bs

            right = min(left + bs, len(prompts))

            # batch of prompts
            sub_p_and_i = prompts_and_indicis[left:right]
            idx, sub_p = zip(*sub_p_and_i)

            # batch of input_ids and attn_masks
            inputs = self.tokenizer(sub_p, return_tensors='pt', padding=True)

            # Dynamic batch size based on save memory
            while inputs.input_ids.numel() > self.safe_numel:
                if bs == 1:
                    break
                bs = max(1, round(bs / 1.5))
                print(
                    f'\nReduce bs to {bs} when seq len reaches {inputs.input_ids.shape[-1]}'
                )
                idx = idx[:bs]
                inputs['input_ids'] = inputs['input_ids'][:bs]
                inputs['attention_mask'] = inputs['attention_mask'][:bs]

            # Send to worker
            self.inq.put((idx, inputs))

            left += bs

            print(
                f'Tokenizing and distributing prompts {right}/{len(prompts)},'
                f' {right/len(prompts):.0%}',
                end='\r')

        print()

        # Collect outputs from workers
        all_probs = [None] * len(prompts)
        count = 0

        while count < len(prompts):
            idx, probs = self.outq.get()
            for i, p in zip(idx, probs):
                assert all_probs[i] is None
                all_probs[i] = p

            count += len(idx)
            print(
                f'Decoding and collecting outputs {count}/{len(prompts)}'
                f', {count/len(prompts):.0%}',
                end='\r')

        if pad:
            all_probs = pad_sequence(all_probs, batch_first=True)

        return all_probs

    def __del__(self):
        print('Exit engine')
        for _ in self.ps:
            self.inq.put((None, None))
        for p in self.ps:
            p.join(timeout=1)
        for p in self.ps:
            p.terminate()


def benchmark(path='llama2/huggingface/llama-2-7b',
              share_gpt='ShareGPT_V3_unfiltered_cleaned_split.json'):

    start = time.monotonic()
    content = json.load(open(share_gpt, 'r'))

    texts = []
    for c in content:
        for cc in c['conversations']:
            texts.append(cc['value'])

    print(f'Parse json in {time.monotonic() - start} seconds.')

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    start = time.monotonic()
    engine = Engine(path, tokenizer=tokenizer)
    probs = engine.decode(texts, sort=True)
    total_tokens = sum(p.numel() for p in probs)

    elapsed = time.monotonic() - start
    print(
        f'Decoded {total_tokens} tokens in {elapsed:.1f} seconds, {total_tokens / elapsed} tokens/s.'
    )

    pickle.dump(probs, open('decode_result.pkl', 'wb'))


def test_decode_dist(path='llama2/huggingface/llama-2-7b'):
    np.set_printoptions(linewidth=200)

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    engine = Engine(path, tokenizer=tokenizer)
    prompt = [
        'I believe the meaning of life is to find your gift. The purpose of life is to give it away.'
    ] * 2
    probs = engine.decode(prompt, sort=False, max_bs=4, pad=True)

    return probs


def test_decode_single():
    gpu_id = 0
    torch.set_default_device(gpu_id)
    torch.set_printoptions(linewidth=200, edgeitems=5)
    np.set_printoptions(linewidth=200, edgeitems=5)
    model, tokenizer = init_model('llama2/huggingface/llama-2-7b')
    model = model.eval()

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    prompt = [
        'I believe the meaning of life is to find your gift. The purpose of life is to give it away.'
    ] * 2

    inputs = tokenizer(prompt, return_tensors='pt', padding=True)

    input_ids = inputs.input_ids.cuda(gpu_id)
    attention_mask = None
    attention_mask = inputs.attention_mask.cuda(gpu_id)
    probs = decode_single(model, input_ids, attention_mask)

    return probs


if __name__ == '__main__':
    benchmark()
    # # p_single = test_decode_single()
    # # p_dist = test_decode_dist()

    # # print(p_single[0])
    # # print(p_dist[0])
    # assert torch.allclose(p_single, p_dist, rtol=1e-3, atol=1e-3)
