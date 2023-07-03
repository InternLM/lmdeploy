# Copyright (c) OpenMMLab. All rights reserved.

import os
import warnings

import fire
import torch

try:
    import deepspeed

    _is_deepspeed_available = True
except ImportError:
    _is_deepspeed_available = False

try:
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              GenerationConfig)

    from .streamer import DecodeOutputStreamer

    _is_transformers_available = True
except ImportError:
    _is_transformers_available = False


def input_prompt():
    print('\ndouble enter to end input >>> ', end='')
    sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def init_model(
    model_path: str,
    tokenizer_path: str,
    use_fast_tokenizer=True,
    local_rank=0,
    world_size=1,
):
    """Note:
    If the model is converted from new version of transformers,
        use_fast_tokenizer should be True.
    If using depodaca/llama-xb-hf, use_fast_tokenizer should be False.
    """

    if not _is_transformers_available:
        raise ImportError('transformers is not installed.\n'
                          'Please install with `pip install transformers`.\n')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              use_fast=use_fast_tokenizer)

    torch.set_default_device(local_rank)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float16)

    if not _is_deepspeed_available:
        warnings.warn('deepspeed is not installed, ',
                      'use plain huggingface model.')
    else:
        model = deepspeed.init_inference(
            model=model,  # Transformers models
            mp_size=world_size,  # Number of GPU
            dtype=torch.float16,  # dtype of the weights (fp16)
            replace_with_kernel_inject=True,
            # replace the model with the kernel injector
            max_out_tokens=2048,
        )

    # print(f"model is loaded on device {model.device}")

    return tokenizer, model


def main(
    model_path: str,
    tokenizer_path: str = None,
    max_new_tokens=64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 1,
    use_fast_tokenizer=True,
):
    torch.manual_seed(seed)

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    if not tokenizer_path:
        tokenizer_path = model_path

    tokenizer, model = init_model(
        model_path,
        tokenizer_path,
        use_fast_tokenizer=use_fast_tokenizer,
        local_rank=local_rank,
        world_size=world_size,
    )

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
    )

    # warmup
    warmup_config = GenerationConfig(
        max_new_tokens=1,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
    )
    model.generate(torch.tensor([[1]]), warmup_config)

    # print("READY ...")
    _on_master = local_rank == 0
    _is_dist = world_size > 1

    while True:
        # Receive prompt on master
        if _on_master:
            prompt = input_prompt()
        else:
            prompt = None
        # Broadcast prompt to all workers
        if _is_dist:
            prompt = [prompt]
            torch.distributed.broadcast_object_list(prompt, src=0)
            prompt = prompt[0]

        if prompt == 'exit':
            exit(0)

        # Re-config during runtime
        if prompt.startswith('config set'):
            try:
                keqv = prompt.split()[-1]
                k, v = keqv.split('=')
                v = eval(v)
                gen_config.__setattr__(k, v)
                print(f'Worker {local_rank} set {k} to {repr(v)}')
            except:  # noqa
                print('illegal instruction')
        else:
            if _on_master:
                streamer = DecodeOutputStreamer(tokenizer)
            else:
                streamer = None
            ids = tokenizer.encode(prompt, return_tensors='pt')
            model.generate(ids, gen_config, streamer=streamer)


if __name__ == '__main__':
    fire.Fire(main)
