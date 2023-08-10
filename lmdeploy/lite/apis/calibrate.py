# Copyright (c) OpenMMLab. All rights reserved.

from pathlib import Path
from typing import Optional

import fire
import torch
from accelerate import (infer_auto_device_map, init_empty_weights,
                        load_checkpoint_in_model)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lmdeploy.lite.quantization import QuantizeContext
from lmdeploy.lite.utils import collect_target_modules, get_calib_loaders

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
         calib_dataset: str = 'c4',
         calib_samples: int = 128,
         calib_seqlen: int = 2048,
         use_awq: bool = True,
         use_i8_kv: bool = False,
         w_bits: int = 4,
         w_sym: bool = False,
         w_granularity: str = 'per_group',
         w_group_size: int = 128,
         kv_bits: int = 8,
         kv_sym: bool = False,
         kv_num_tp: int = 1,
         pytorch_dir: Optional[str] = None,
         turbomind_dir: Optional[str] = None,
         device: str = 'cuda'):

    assert calib_dataset in ['c4', 'ptb', 'wikitext2', 'pileval'], \
        'Currently, only support `c4`, `ptb`, `wikitext2`, or `pileval`.'

    if use_awq:
        assert w_granularity == 'per_group'
        assert pytorch_dir

    tokenizer = AutoTokenizer.from_pretrained(model,
                                              use_fast=False,
                                              trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    checkpoint = hf_config._name_or_path

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model,
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True)
        model.config.use_cache = False

    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    norm_type = NORM_TYPE_MAP[type(model).__name__]

    decoder_layers = collect_target_modules(model, layer_type)
    device_map = infer_auto_device_map(model,
                                       no_split_module_classes=[layer_type])
    for name in device_map.keys():
        if name in decoder_layers:
            device_map[name] = 'cpu'
        else:
            device_map[name] = 0
    load_checkpoint_in_model(model, checkpoint, device_map)

    print('Loading calibrate dataset ...')
    calib_loader, _ = get_calib_loaders(calib_dataset,
                                        tokenizer,
                                        nsamples=calib_samples,
                                        seqlen=calib_seqlen)

    quant_ctx = QuantizeContext(model,
                                tokenizer,
                                layer_type=layer_type,
                                norm_type=norm_type,
                                device=device)

    with quant_ctx:
        all_data = torch.cat([
            data if isinstance(data, torch.Tensor) else data[0]
            for data in calib_loader
        ]).to(device)
        quant_ctx.calibrate(all_data)

    if pytorch_dir:
        pytorch_dir = Path(pytorch_dir)
        pytorch_dir.mkdir(parents=True, exist_ok=True)
        quant_ctx.export_stats(pytorch_dir)
        if use_awq:
            quant_ctx.auto_awq(w_bits, w_sym, w_group_size, pytorch_dir)

    if turbomind_dir and use_i8_kv:
        turbomind_dir = Path(turbomind_dir)
        turbomind_dir.mkdir(parents=True, exist_ok=True)
        quant_ctx.export_turbomind_kv_qparams(kv_bits, kv_sym, turbomind_dir,
                                              kv_num_tp)


if __name__ == '__main__':

    fire.Fire(main)
