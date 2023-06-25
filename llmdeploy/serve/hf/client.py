# Copyright (c) OpenMMLab. All rights reserved.\

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


def init_model(model_path: str,
               tokenizer_path: str,
               tp: int = 1,
               use_fast_tokenizer=True):
    """Note:
    If the model is converted from new version of transformers, use_fast_tokenizer should be True.
    If using depodaca/llama-xb-hf, use_fast_tokenizer should be False.
    """
    if not _is_transformers_available:
        raise ImportError('transformers is not installed.\n'
                          'Please install with `pip install transformers`.\n')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              use_fast=use_fast_tokenizer)

    torch.set_default_device('cuda')
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float16)

    if not _is_deepspeed_available:
        warnings.warn('deepspeed is not installed, ',
                      'use plain huggingface model.')
    else:
        model = deepspeed.init_inference(
            model=model,  # Transformers models
            mp_size=tp,  # Number of GPU
            dtype=torch.float16,  # dtype of the weights (fp16)
            replace_with_kernel_inject=
            True,  # replace the model with the kernel injector
            max_out_tokens=2048,
        )

    # print(f"model is loaded on device {model.device}")

    return tokenizer, model


def main(model_path: str, tokenizer_path: str = None, tp: int = 1):
    if not tokenizer_path:
        tokenizer_path = model_path

    tokenizer, model = init_model(model_path, tokenizer_path, tp)

    gen_config = GenerationConfig(max_new_tokens=64, do_sample=False)

    # warmup
    warmup_config = GenerationConfig(max_new_tokens=1, do_sample=False)
    model.generate(torch.tensor([[1]]), warmup_config)

    print('READY ...')
    while True:
        prompt = input_prompt()
        if prompt == 'exit':
            exit(0)
        elif prompt.startswith('config set'):
            try:
                keqv = prompt.split()[-1]
                k, v = keqv.split('=')
                v = eval(v)
                gen_config.__setattr__(k, v)
                print(f'set {k} to {repr(v)}')
            except:
                print('illegal instruction')
        else:
            ids = tokenizer.encode(prompt, return_tensors='pt')
            model.generate(ids,
                           gen_config,
                           streamer=DecodeOutputStreamer(tokenizer))


if __name__ == '__main__':
    fire.Fire(main)
