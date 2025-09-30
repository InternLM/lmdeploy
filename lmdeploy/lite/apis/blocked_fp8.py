# Copyright (c) OpenMMLab. All rights reserved.

import os
import os.path as osp
from typing import Literal

import fire
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmdeploy.lite.quantization.weight.quant_utils import quant_blocked_fp8
from lmdeploy.lite.utils import collect_target_modules
from lmdeploy.pytorch.models import QLinear


def blocked_fp8(model: str,
                work_dir: str = './work_dir',
                quant_dtype: Literal['fp8', 'float8_e4m3fn', 'float8_e5m2'] = 'float8_e4m3fn',
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
        model_path = get_model(model, revision=revision, download_dir=download_dir)
    else:
        model_path = model

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16)
    model = model.eval().cuda()

    # collect all linear layers
    fcs = collect_target_modules(model, nn.Linear)
    skip_patterns = [
        'lm_head',
        'embed_tokens',
        'mlp.gate',  # sparse MOE router gate
        'vision_model',  # non-HF InternVL, vision part
        'mlp1',  # non-HF InternVL, projector
        'mlp2',  # non-HF InternVL-Flash, projector
        'vision_tower',  # HF InternVL, vision part
        'multi_modal_projector',  # HF InternVL, projector
    ]
    modules_to_not_convert = []

    # quantize and replace linear layers
    for name, linear in tqdm(fcs.items(), desc='Quantizing'):
        # skip not to convert modules
        if any([x in name for x in skip_patterns]):
            modules_to_not_convert.append(name)
            continue

        linear.to('cuda')
        # quantize weight
        q_weight, scales = quant_blocked_fp8(weight=linear.weight, fp8_dtype=q_dtype, block_size=block_size)

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
    if quant_dtype == 'float8_e4m3fn':
        fmt = 'e4m3'
    elif quant_dtype == 'float8_e5m2':
        fmt = 'e5m2'
    quant_config = dict(activation_scheme='dynamic',
                        modules_to_not_convert=modules_to_not_convert,
                        fmt=fmt,
                        quant_method='fp8',
                        weight_block_size=[block_size, block_size])
    model.config.update(dict(quantization_config=quant_config))

    # save model and tokenizer
    if not osp.exists(work_dir):
        os.makedirs(work_dir)
    print('Saving the quantized model ...')
    model.save_pretrained(work_dir, safe_serialization=True)
    tokenizer.save_pretrained(work_dir)
    print(f'Blocked FP8 model successfully saved to {work_dir}')


if __name__ == '__main__':
    fire.Fire(blocked_fp8)
