# Copyright (c) OpenMMLab. All rights reserved.

from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from lmdeploy.lite.quantization.awq import (FC_FCS_MAP, NORM_FCS_MAP,
                                            quant_weights, smooth_layers)
from lmdeploy.lite.utils import collect_target_modules, load_hf_from_pretrained

# from lmdeploy.lite.utils.export_turbomind import export_turbomind_config

LAYER_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMDecoderLayer',
    'QWenLMHeadModel': 'QWenBlock',
    'BaiChuanForCausalLM': 'DecoderLayer',  # Baichuan 7B
    'BaichuanForCausalLM': 'DecoderLayer',  # Baichuan2 7B
    'LlamaForCausalLM': 'LlamaDecoderLayer',
}
NORM_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMRMSNorm',
    'QWenLMHeadModel': 'RMSNorm',
    'BaiChuanForCausalLM': 'RMSNorm',  # Baichuan 7B
    'BaichuanForCausalLM': 'RMSNorm',  # Baichuan2 7B
    'LlamaForCausalLM': 'LlamaRMSNorm',
}


def auto_awq(model: str,
             work_dir: str,
             w_bits: int = 4,
             w_sym: bool = False,
             w_group_size: int = 128,
             device: str = 'cuda'):

    assert model != work_dir, '$WORK_DIR and $HF_MODEL should be different'
    model_path = model  # noqa

    # Load tokenizer and configuration
    tokenizer = AutoTokenizer.from_pretrained(model,
                                              use_fast=False,
                                              trust_remote_code=True)

    model = load_hf_from_pretrained(model,
                                    torch_dtype=torch.float16,
                                    trust_remote_code=True)

    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    fc2fcs = FC_FCS_MAP[layer_type]
    norm2fcs = NORM_FCS_MAP[layer_type]

    work_dir = Path(work_dir)

    act_scales = torch.load(work_dir / 'inputs_stats.pth')['absmax']
    layers = collect_target_modules(model, layer_type)
    fcs = {}
    for l_name, layer in layers.items():
        name2fc = collect_target_modules(layer, nn.Linear, prefix=l_name)
        fcs.update(name2fc)

    smooth_layers(layers, fc2fcs, norm2fcs, act_scales, w_group_size, device)
    quant_weights(model, fcs, w_bits, w_sym, w_group_size, device)

    model.save_pretrained(work_dir, max_shard_size='2GB')
    tokenizer.save_pretrained(work_dir)

    # export_turbomind_config(model_name,
    #                         model_path,
    #                         work_dir,
    #                         group_size=w_group_size)


if __name__ == '__main__':
    import fire

    fire.Fire(auto_awq)
