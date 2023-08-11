# Copyright (c) OpenMMLab. All rights reserved.

from pathlib import Path

import fire
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmdeploy.lite.quantization.awq import (FC_FCS_MAP, NORM_FCS_MAP,
                                            quant_weights, smooth_layers)
from lmdeploy.lite.utils import collect_target_modules

LAYER_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMDecoderLayer',
    'QWenLMHeadModel': 'QWenBlock',
    'BaiChuanForCausalLM': 'DecoderLayer',
    'LlamaForCausalLM': 'LlamaDecoderLayer',
}
NORM_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMRMSNorm',
    'QWenLMHeadModel': 'RMSNorm',
    'BaiChuanForCausalLM': 'RMSNorm',
    'LlamaForCausalLM': 'LlamaRMSNorm',
}


def main(model: str,
         w_bits: int = 4,
         w_sym: bool = False,
         w_group_size: int = 128,
         work_dir: str = './work_dir',
         device: str = 'cuda'):

    tokenizer = AutoTokenizer.from_pretrained(model,
                                              use_fast=False,
                                              trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True)

    layer_type = LAYER_TYPE_MAP[type(model).__name__]

    fc2fcs = FC_FCS_MAP[layer_type]
    norm2fcs = NORM_FCS_MAP[layer_type]
    work_dir = Path(work_dir)

    act_scales = torch.load(work_dir / 'inputs_stats.pth')['absmean']
    layers = collect_target_modules(model, layer_type)
    fcs = {}
    for l_name, layer in layers.items():
        name2fc = collect_target_modules(layer, nn.Linear, prefix=l_name)
        fcs.update(name2fc)

    smooth_layers(layers, fc2fcs, norm2fcs, act_scales, w_group_size, device)
    quant_weights(model, fcs, w_bits, w_sym, w_group_size, device)

    model.save_pretrained(work_dir)
    tokenizer.save_pretrained(work_dir)


if __name__ == '__main__':

    fire.Fire(main)
