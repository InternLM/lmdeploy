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

    _is_transformers_available = True
except ImportError:
    _is_transformers_available = False
else:
    from .accel import LoadNoInit
    from .utils import get_utils


def input_prompt():
    """Helper function for getting input from users."""

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
    """Initialize model and tokenizer from given path.

    Args:
        model_path (str): Path to model.
        tokenizer_path (str): Path to tokenizer.
        use_fast_tokenizer (bool): Whether to use fast tokenizer.
        local_rank (int): Local rank of current process.
        world_size (int): World size of current process.

    Note:
        If the model is converted from new version of transformers,
            use_fast_tokenizer should be True.
        If using depodaca/llama-xb-hf, use_fast_tokenizer should be False.
    """

    if not _is_transformers_available:
        raise ImportError('transformers is not installed.\n'
                          'Please install with `pip install transformers`.\n')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              use_fast=use_fast_tokenizer,
                                              trust_remote_code=True)

    with LoadNoInit():
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True)

    if not _is_deepspeed_available:
        warnings.warn('deepspeed is not installed, '
                      'use plain huggingface model.')

        model = model.cuda(local_rank)
    else:
        config = dict(
            tensor_parallel=dict(tp_size=world_size),  # Number of GPU
            dtype=torch.float16,  # dtype of the weights (fp16)
            replace_with_kernel_inject=True,
            # replace the model with the kernel injector
            max_out_tokens=2048,
        )
        # For internlm model not supported by DeepSpeed,
        # set replace_with_kernel_inject=False to use AutoTP.
        # It's a hotfix before the progress is merged
        # https://github.com/InternLM/lmdeploy/issues/136
        if 'InternLM' in model.__class__.__name__:
            try:
                # Use customized deepspeed supporting internlm
                # https://github.com/wangruohui/DeepSpeed/tree/support_internlm_0.10.0 (commit cdef2ce)  # noqa: E501
                from deepspeed.module_inject.containers.internlm import \
                    InternLMLayerPolicy
            except ImportError:
                # use stock deepspeed
                config.update({'replace_with_kernel_inject': False})
            else:
                for module in model.modules():
                    if module.__class__.__name__ == 'InternLMDecoderLayer':
                        InternLMLayerPolicy._orig_layer_class = module.__class__  # noqa: E501
                        break

        model = deepspeed.init_inference(
            model=model,  # Transformers models
            config=config,
        )

    return tokenizer, model


def main(
    model_path: str,
    tokenizer_path: str = None,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    use_fast_tokenizer: bool = True,
):
    """Start chat session with given model.

    Args:
        model_path (str): Path to model.
        tokenizer_path (str): Path to tokenizer.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Temperature for sampling.
        top_p (float): Top p for sampling.
        seed (int): Random seed.
        use_fast_tokenizer (bool): Whether to use fast tokenizer.
    """

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

    Decorator, Streamer, stop_criteria = get_utils(model)

    # warmup
    warmup_config = GenerationConfig(
        max_new_tokens=1,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
    )
    model.generate(torch.tensor([[1]], device=local_rank), warmup_config)

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
                streamer = Streamer(tokenizer)
            else:
                streamer = None

            prompt = Decorator.decorate(prompt)
            ids = tokenizer.encode(prompt, return_tensors='pt')
            model.generate(ids.cuda(local_rank),
                           gen_config,
                           streamer=streamer,
                           stopping_criteria=stop_criteria)


if __name__ == '__main__':
    fire.Fire(main)
