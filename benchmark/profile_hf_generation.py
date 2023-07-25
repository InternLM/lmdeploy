"""Benchmark huggingface models and maybe speedup by deepspeed.

Theoretically, this tool is compatible with all huggingface models.

Example 1: Test huggingface llama2 with simulated 128 input token and 128 generated token

```shell
python profile_hf_generation.py \
    --model_path $PATH_TO_HF_LLAMA2 \
    --batch_size 16 \
    --input_seqlen 128 \
    --gen_seqlen 128 \
    --out_file profile_hf.csv
```

Example 2: Same test above but accelerated with DeepSpeed inference and more round to test

```shell
python profile_hf_generation.py \
    --model_path $PATH_TO_HF_LLAMA2 \
    --batch_size 16 \
    --input_seqlen 128 \
    --gen_seqlen 128 \
    --test_round 2 \
    --accel deepspeed \
    --out_file profile_hf.csv
```

Example 3: Same test above but do not use streamer to measure time of every token
        Only only overall time is measured but a little bit faster

```shell
python profile_hf_generation.py \
    --model_path $PATH_TO_HF_LLAMA2 \
    --batch_size 16 \
    --input_seqlen 128 \
    --gen_seqlen 128 \
    --test_round 2 \
    --accel deepspeed \
    --no-streamer \
    --out_file profile_hf.csv
```

Result will be saved in `profile_hf.csv`, which is a comma-separated file
with the following fields.

1. model: name of model, specified with --model_log_name
2. batch_size: as name
3. input_seqlen: as name
4. gen_len: length of sequence length to generate
5. total_len: gen_len + input_len

In total, the model will take a random input ids of shape (batch_size, input_seqlen) and
run forward `gen_len` times to generate output ids of shape (batch_size, gen_len)

6. first_time: latency to forward the first (batch_size, input_seqlen) ids
    to get `input_seqlen+1`-th batch of output of shape (batch_size, 1)
7. next_time: average latency of the next samples (averaged of 5 sample),
    this measure latency when context length is short
8. last_time: average latency of the last samples (averaged of 5 sample),
    this measure latency when context length is long
9. total time: total time to generate all tokens
10. throughput(total): bs * total_len / total_time (same as vllm)
11. throughput(gen): bs * gen_len / total_time (same as vllm)
"""   # noqa: E501

import csv
import logging
import os
import time
from typing import Optional

import fire
import torch
from transformers import AutoModelForCausalLM, GenerationConfig

from lmdeploy.pytorch.accel import LoadNoInit

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
info = logger.info
warning = logger.warning
debug = logger.debug
cinfo = lambda x: info('\033[1;32m' + x + '\033[0m')  # noqa E731
cprint = lambda x: print('\033[1;32m' + x + '\033[0m')  # noqa E731
avg = lambda x: sum(x) / len(x)  # noqa E731


class TimingStreamer:
    """Timing helper for HuggingFace models."""

    def __init__(self) -> None:
        # self.token_cache = []
        # self.tokens = None

        torch.cuda.synchronize()

        self.evts = []
        # self.value = 0

    def put(self, value):
        """
        Notes:
            When `put` is called for the first time, no prompt is feed to the model yet.
            When `put` is called later, event is recorded for the previous generation,
                which means the second event records the time for the first prompt.

            GenerationMixin will call `.cpu()` on output token which implies a sync.
        """  # noqa: E501
        # self.value += 1
        # self.token_cache.append(value)
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self.evts.append(evt)

    def end(self):
        torch.cuda.synchronize()
        # self.tokens = torch.hstack([_atleast_2d(v) for v in self.token_cache])    # noqa: E501

    def get_times(self):
        """Maybe deprecated.

        Returns:
          a list of times in ms for (first prompt, avg next token, total time)
        """
        first = self.evts[0].elapsed_time(self.evts[1])
        rest = [
            self.evts[i].elapsed_time(self.evts[i + 1])
            for i in range(1,
                           len(self.evts) - 1)
        ]
        avg = sum(rest) / len(rest)
        return first + sum(rest), first, avg

    def raw_times(self):
        """
        Returns:
          a list of times in ms.
        """
        evts = self.evts
        r = [evts[i].elapsed_time(evts[i + 1]) for i in range(len(evts) - 1)]
        return r


class CSVWRitter:

    def __init__(
        self,
        file='unnamed.csv',
        header=[
            'model',
            'batch_size',
            'input_seqlen',
            'gen_len',
            'total_len',
            'first_time',
            'next_time',
            'last_time',
            'total time',
            'throughput(total)',
            'throughput(gen)',
        ],
    ):
        if self.on_master:
            self.file = file
            csv.writer(open(file, 'a')).writerow(header)

    def write(self, line):
        if self.on_master:
            csv.writer(open(self.file, 'a')).writerow(line)

    @property
    def on_master(self):
        # return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0  # noqa: E501
        rank = int(os.environ.get('RANK', 0))
        return rank == 0


def init_hf_model(model_path: str):
    start = time.monotonic()
    with LoadNoInit():
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True)
    print(f'load model in {time.monotonic() -start} s')
    return model


def accel_deepspeed(model, max_out_tokens, tp_size=1):
    import deepspeed
    ds_model = deepspeed.init_inference(
        model=model,  # Transformers models
        tensor_parallel={'tp_size': tp_size},
        dtype=torch.float16,  # dtype of the weights (fp16)
        replace_with_kernel_inject=True,
        max_out_tokens=max_out_tokens,
    )

    return ds_model


def main(model_path: str,
         batch_size: int,
         input_seqlen: int,
         gen_seqlen: int,
         test_round: int = 1,
         accel: Optional[str] = None,
         out_file: Optional[str] = 'profile_hf.csv',
         model_log_name: Optional[str] = None,
         no_streamer: bool = False):

    total_seqlen = input_seqlen + gen_seqlen

    model = init_hf_model(model_path)

    vocab_size = model.config.vocab_size
    if model_log_name is None:
        model_log_name = model.__class__.__name__

    if accel is None:
        model = model.cuda()
    elif accel == 'deepspeed':
        model = accel_deepspeed(model, total_seqlen + 6)
        # longer total seqlen for fault tolerance
    else:
        raise NotImplementedError(f'accel {accel} not supported.')

    # log to file
    csvwritter = CSVWRitter(out_file)

    cprint('Benchmarking {} '
           f'with batch_size={batch_size}, input_seqlen={input_seqlen}, '
           f'gen_seqlen={gen_seqlen}, accel={accel}')
    for r in range(test_round):
        # TODO: now write every round to csv
        # Use external tool for analysis

        cprint(f'Test round {r}')
        # input_id = 0 sometimes cause some cuda error
        fake_inputs = torch.randint(10, vocab_size, (batch_size, input_seqlen))
        fake_inputs = fake_inputs.cuda()

        ts = TimingStreamer() if not no_streamer else None

        torch.cuda.synchronize()
        start = time.monotonic()
        fake_outputs = model.generate(
            fake_inputs,
            GenerationConfig(max_new_tokens=gen_seqlen,
                             do_sample=False,
                             eos_token_id=[-1]),
            streamer=ts,
        )
        torch.cuda.synchronize()
        end = time.monotonic()
        assert fake_outputs.size() == (batch_size,
                                       total_seqlen), fake_outputs.size()

        # total_time, first_time, _ = ts.get_times()
        if no_streamer:
            total_time = (end - start) * 1000
            first_time = next_time = last_time = 0
        else:
            raw_times = ts.raw_times()  # You may further analyze this
            total_time = sum(raw_times)
            first_time = raw_times[0]
            next_time = avg(raw_times[1:6])
            last_time = avg(raw_times[-5:])
        tt = batch_size * total_seqlen * 1000 / total_time
        tg = batch_size * gen_seqlen * 1000 / total_time
        cprint(f'First token/ms: {first_time:.1f}, '
               f'Next tokens/ms: {next_time:.1f}, '
               f'Last tokens/ms: {last_time:.1f}, '
               f'Total Time/ms: {total_time:5.3f}, '
               f'Throughput Total(tok/s): {tt:5.3f}, '
               f'Throughput Gen(tok/s): {tg:5.3f}')

        if test_round > 1 and r > 0:
            # First round is warm up
            csvwritter.write([
                model_log_name, batch_size, input_seqlen, gen_seqlen,
                total_seqlen, first_time, next_time, last_time, total_time, tt,
                tg
            ])


if __name__ == '__main__':
    fire.Fire(main)
