# Copyright (c) OpenMMLab. All rights reserved.

import os
import os.path as osp
from typing import Literal

import fire
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmdeploy.lite.quantization.weight.quant_utils import quant_blocked_fp8
from lmdeploy.lite.utils import collect_target_modules
from lmdeploy.pytorch.models import QLinear


def blocked_fp8(model: str,
                work_dir: str = './work_dir',
                quant_dtype: Literal['float8_e4m3fn', 'float8_e5m2'] = 'float8_e4m3fn',
                block_size: int = 128,
                revision: str = None,
                download_dir: str = None):
    if quant_dtype == 'fp8':
        quant_dtype = 'float8_e4m3fn'

    q_dtype = getattr(torch, quant_dtype, None)
    assert q_dtype is not None

    if not osp.exists(model):
        print(f'can\'t find model from local_path {model}, '
              'try to download from remote')
        from lmdeploy.utils import get_model
        model = get_model(model, revision=revision, download_dir=download_dir)

    model_path = model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.eval().cuda()

    # collect all linear layers
    fcs = collect_target_modules(model, nn.Linear)
    modules_to_not_convert = [
        'language_model.lm_head',
        'language_model.model.embed_tokens',
        'vision_model',  # don't quantize vision part in internvl3.5
        'mlp1'  # don't quantize internvl3.5 mlp1
    ]

    # quantize and replace linear layers
    for name, linear in fcs.items():
        # skip not to convert modules
        if any([x in name for x in modules_to_not_convert]):
            print(f'skip: {name}')
            continue

        print(f'quantize: {name}')
        linear.to('cuda')
        # quantize weight
        q_weight, scales = quant_blocked_fp8(linear.weight, q_dtype, block_size=block_size)

        # create and replace with QLinear
        q_linear = QLinear.from_float(linear, quant_dtype=q_dtype, initialization=False)
        q_linear.weight.data = q_weight
        q_linear.weight_scale_inv.data = scales
        if linear.bias is not None:
            q_linear.bias.data = linear.bias.detach()
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, q_linear)

        # move original layer to CPU to free GPU memory
        linear.to('cpu')
        torch.cuda.empty_cache()

    model.to('cpu')

    # update model config
    quant_config = dict(activation_scheme='dynamic',
                        modules_to_not_convert=modules_to_not_convert,
                        fmt='e4m3',
                        quant_method='fp8',
                        weight_block_size=[block_size, block_size])
    model.config.update(dict(quantization_config=quant_config))

    # save model and tokenizer
    if not osp.exists(work_dir):
        os.makedirs(work_dir)
    model.save_pretrained(work_dir, safe_serialization=True)
    tokenizer.save_pretrained(work_dir)
    print(f'Blocked FP8 model saved to {work_dir}')


if __name__ == '__main__':
    fire.Fire(blocked_fp8)
