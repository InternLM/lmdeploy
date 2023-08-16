# Copyright (c) OpenMMLab. All rights reserved.
import queue
import warnings
from typing import List, Optional

import pynvml
import torch
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase)

from .model import accel_model, init_model


def safe_numel(free_mem, model_size, max_intermediate):
    """Number of elements without out-of-memory."""
    return int(free_mem - model_size) // max_intermediate


def avail_gpus(percentage=0.96):
    """Detect available gpus.

    Args:
        percentage (float): The minimum percentage of free memory to be
            considered as available.

    Return:
       A list of gpu ids.
       average free memory on single gpu.
    """

    gpus = []
    mems = []
    pynvml.nvmlInit()
    for i in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(i))
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free, total = int(mem_info.free), int(mem_info.total)

        if free / total > percentage:
            gpus.append(i)
            mems.append(free)
    pynvml.nvmlShutdown()

    if len(gpus) == 0:
        raise RuntimeError('No GPU available.')

    return gpus, sum(mems) / len(mems)


@torch.no_grad()
def decode_single(model: PreTrainedModel,
                  input_ids: torch.Tensor,
                  attention_mask: torch.Tensor = None):
    """Decode a single batch.

    Args:
        model (PreTrainedModel): Pretrained model.
        input_ids (torch.Tensor): A batch of input ids.
        attention_mask (torch.Tensor): A batch of attention masks.

    Returns:
        torch.Tensor: A batch of probabilities (on CPU).


    Note:
        This function assume input_ids[i] = [bos, x1, x2, ..., xn]
        and return prob = [p(x1|bos), p(x2|bos,x1), ..., p(xn|bos..xn-1)]
        So prob is shorter than input_ids by 1.
    """

    # Call Causal LM forward
    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True)
    # fp32, [bs, seq_len, vocab_size]
    logits = outputs.logits

    # inplace softmax to save memory
    # probs is logits
    torch.softmax(logits, dim=-1, out=logits)

    # Shift to fetch probabilities
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
    # torch.set_default_device(gpu_id)
    model, _ = init_model(model_path)
    model = model.eval()
    model = accel_model(model, accel, gpu_id=gpu_id)

    while True:
        try:
            idx, input_ids, input_lens = inq.get(timeout=1)
        except queue.Empty:
            continue

        if input_ids is None:
            break

        input_ids = input_ids.cuda(gpu_id)
        max_len = max(input_lens)
        assert max_len == input_ids.size(-1), \
            f'input_ids.shape = {input_ids.shape}, max_len = {max_len}'

        input_lens = torch.tensor(input_lens, device=gpu_id)
        attention_mask = torch.arange(
            max_len, device=gpu_id)[None, :] < input_lens[:, None]
        # attention_mask = inputs['attention_mask'].cuda(gpu_id)

        assert attention_mask.shape == input_ids.shape, \
            f'attention_mask.shape = {attention_mask.shape}'
        try:
            probs = decode_single(model, input_ids, attention_mask)
        except torch.cuda.OutOfMemoryError:
            warnings.warn(
                f'OOM on GPU {gpu_id}, discard prompts at indics {idx}.')
            probs = torch.empty((input_ids.size(0), 0),
                                dtype=torch.float32,
                                device='cpu')

        outq.put((idx, probs))

    inq.close()
    outq.close()
    print(f'Worker {gpu_id} finished.')


class Engine:
    """Multi-GPU deciding engine.

    Args:
        model_path (str): Path to the pretrained model.
        tokenizer_path (str, optional): Path to the pretrained tokenizer.
            Defaults to None.
            Either tokenizer_path or tokenizer should be provided.
        tokenizer (PreTrainedTokenizerBase, optional): Pre-configured tokenizer.
            Defaults to None.
            Either tokenizer_path or tokenizer should be provided.
        accel (str, optional): Acceleration method.
            Defaults to None. 'deepspeed' is not tested.
        gpu_mem_percentage (float, optional): GPU with memory larger than this value
            are considered available and be used as decode device.
            Defaults to 0.96.
        model_size_byte (float, optional): (Approximate) model size in bytes.
            Defaults to 14e9 (7B model in FP16).
        bytes_per_token (float, optional): (Approximate) memory cost per token in bytes.
            Defaults to 2e6 (2MB).
            ``bytes_per_token`` and ``model_size_byte`` are used to compute
            the maximum batch size for given seq_length
    """  # noqa: E501

    def __init__(self,
                 model_path: str,
                 tokenizer_path: Optional[str] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 accel: Optional[str] = None,
                 gpu_mem_percentage: float = 0.96,
                 model_size_byte=14e9,
                 bytes_per_token=2e6):

        gpu_ids, mem = avail_gpus(gpu_mem_percentage)
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
        self.safe_numel = safe_numel(mem, model_size_byte, bytes_per_token)

    def clear_queue(self):
        for q in self.inq, self.outq:
            while not q.empty():
                q.get()

    def decode(self,
               prompt_token_ids: List[List[int]],
               sort=True,
               max_bs: int = 1024,
               pad=True):
        """Inference the model to compute probabilities.

        Args:
            prompts (List[List[int]]): List of list of token ids.
            sort (bool, optional): Internally sort the prompts by length to achieve better efficiency.
                Defaults to True.
                Note: orders of returned probabilities are always the same as the input.
            max_bs (int, optional): Maximum batch size.
                Defaults to 1024.
            pad (bool, optional): Pad the prompts in every mini batch to the same length.
                Defaults to True. Set to False to save memory.

        Returns:
            numpy.ndarray: List of probabilities of all prompts, with prob=0 padded.
                If pad is True
            List[numpy.ndarray]: List of probabilities of all prompts without padding.
                If pad is False, the result if a batch of tensor.

        Note:
            This function will accept input token_ids = [x0(=bos), x1, x2, ..., xn]
            and compute prob = [p(x1|x0), p(x2|x0,x1), ..., p(xn|x0..xn-1)]
            So prob is shorter than input_ids by 1.
        """  # noqa: E501

        self.clear_queue()

        # sort to achieve better efficiency
        if sort:
            prompts_and_indicis = sorted(enumerate(prompt_token_ids),
                                         key=lambda i_and_x: len(i_and_x[1]))
        else:
            prompts_and_indicis = list(enumerate(prompt_token_ids))

        left = 0
        bs = max_bs

        while left < len(prompt_token_ids):

            if not sort:
                bs = max_bs

            right = min(left + bs, len(prompt_token_ids))

            # batch of prompts
            sub_p_and_i = prompts_and_indicis[left:right]
            idx, sub_p = zip(*sub_p_and_i)

            # batch of input_ids and attn_masks
            # inputs = self.tokenizer(sub_p, return_tensors='pt', padding=True)
            input_ids = [torch.tensor(p) for p in sub_p]
            input_ids = pad_sequence(input_ids, batch_first=True)
            input_lens = [len(p) for p in sub_p]

            # Dynamic batch size based on safe memory
            while input_ids.numel() > self.safe_numel:
                if bs == 1:
                    break
                bs = max(1, round(bs / 1.5))
                print(f'\nReduce bs to {bs} when seq len reaches '
                      f'{input_ids.shape[-1]}')
                idx = idx[:bs]
                input_lens = input_lens[:bs]
                input_ids = input_ids[:bs, :max(input_lens)]

            # Send to worker
            self.inq.put((idx, input_ids, input_lens))

            left += bs

            print(
                f'Distributing prompts {right}/{len(prompt_token_ids)},'
                f' {right/len(prompt_token_ids):.0%}',
                end='\r')

        print()

        # Collect outputs from workers
        all_probs = [None] * len(prompt_token_ids)
        count = 0

        while count < len(prompt_token_ids):
            idx, probs = self.outq.get()
            for i, p in zip(idx, probs):
                assert all_probs[i] is None
                all_probs[i] = p

            count += len(idx)
            print(
                f'Decoding and collecting outputs '
                f'{count}/{len(prompt_token_ids)}, '
                f'{count/len(prompt_token_ids):.0%}',
                end='\r')

        if pad:
            all_probs = pad_sequence(all_probs, batch_first=True)
            all_probs = all_probs.cpu().numpy()
        else:
            all_probs = [p.cpu().numpy() for p in all_probs]

        return all_probs

    def __del__(self):
        print('Exiting engine ...')
        for _ in self.ps:
            self.inq.put((None, None, None))
        for p in self.ps:
            p.join(timeout=1)
