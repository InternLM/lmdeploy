# Copyright (c) OpenMMLab. All rights reserved.

import shutil

import fire
import torch
from accelerate import (infer_auto_device_map, init_empty_weights,
                        load_checkpoint_in_model)
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lmdeploy.lite.quantization import CalibrationContext
from lmdeploy.lite.quantization.awq import (FC_FCS_MAP, NORM_FCS_MAP,
                                            smooth_layers)
from lmdeploy.lite.utils import collect_target_modules, get_calib_loaders
from lmdeploy.pytorch_poc.models import QLinear, QRMSNorm

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

MODEL_PATH_MAP = {
    'InternLMForCausalLM':
    './lmdeploy/pytorch_poc/modeling/modeling_internlm.py',
    'LlamaForCausalLM': './lmdeploy/pytorch_poc/modeling/modeling_llama.py',
    'BaiChuanForCausalLM':
    './lmdeploy/pytorch_poc/modeling/modeling_baichuan.py'
}

AUTO_MAP = {
    'InternLMForCausalLM': {
        'AutoConfig': 'configuration_internlm.InternLMConfig',
        'AutoModel': 'modeling_internlm.InternLMForCausalLM',
        'AutoModelForCausalLM': 'modeling_internlm.InternLMForCausalLM'
    },
    'LlamaForCausalLM': {
        'AutoModel': 'modeling_llama.LlamaForCausalLM',
        'AutoModelForCausalLM': 'modeling_llama.LlamaForCausalLM'
    },
    'BaiChuanForCausalLM': {
        'AutoConfig': 'configuration_baichuan.BaiChuanConfig',
        'AutoModelForCausalLM': 'modeling_baichuan.BaiChuanForCausalLM'
    }
}


def calibrate(model,
              tokenizer,
              calib_dataset: str = 'c4',
              calib_samples: int = 128,
              calib_seqlen: int = 2048,
              device: str = 'cuda') -> None:
    """The main function for loading the model and performing calibration on a
    given dataset.

    Args:
        model (nn.Module): The transformers model.
        tokenizer: The corresponding tokenizer.
        calib_dataset (str, optional): The calibration dataset name.
            Defaults to 'c4'.
        calib_samples (int, optional): The number of samples for calibration.
            Defaults to 128.
        calib_seqlen (int, optional): The sequence length for calibration.
            Defaults to 2048.
        device (str, optional): The device to be used for calculation.
            Defaults to 'cuda'.
    """

    assert calib_dataset in ['c4', 'ptb', 'wikitext2', 'pileval'], \
        'Support only `c4`, `ptb`, `wikitext2` or `pileval`.'

    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    norm_type = NORM_TYPE_MAP[type(model).__name__]

    print('Loading calibrate dataset ...')
    calib_loader, _ = get_calib_loaders(calib_dataset,
                                        tokenizer,
                                        nsamples=calib_samples,
                                        seqlen=calib_seqlen)

    # Initialize calibration context
    calib_ctx = CalibrationContext(model,
                                   tokenizer,
                                   layer_type=layer_type,
                                   norm_type=norm_type,
                                   device=device)

    with calib_ctx:
        all_data = torch.cat([
            data if isinstance(data, torch.Tensor) else data[0]
            for data in calib_loader
        ]).to(device)
        calib_ctx.calibrate(all_data)

    inp_stats = calib_ctx.collect_inputs_stats()
    return inp_stats


def smooth_quant(model: str,
                 work_dir: str = './work_dir',
                 calib_dataset: str = 'c4',
                 calib_samples: int = 128,
                 calib_seqlen: int = 2048,
                 device: str = 'cuda'):

    # Load tokenizer and configuration
    tokenizer = AutoTokenizer.from_pretrained(model,
                                              use_fast=False,
                                              trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    checkpoint = hf_config._name_or_path

    with init_empty_weights():
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model,
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True)
        model.config.use_cache = False

    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    norm_type = NORM_TYPE_MAP[type(model).__name__]
    fc2fcs = FC_FCS_MAP[layer_type]
    norm2fcs = NORM_FCS_MAP[layer_type]

    decoder_layers = collect_target_modules(model, layer_type)

    # Infer device map
    device_map = infer_auto_device_map(model,
                                       no_split_module_classes=[layer_type])
    for name in device_map.keys():
        if name in decoder_layers or 'lm_head' in name:
            device_map[name] = 'cpu'
        else:
            device_map[name] = 0
    load_checkpoint_in_model(model,
                             checkpoint,
                             device_map,
                             dtype=torch.float16)

    inp_stats = calibrate(model, tokenizer, calib_dataset, calib_samples,
                          calib_seqlen, device)
    act_scales = inp_stats['absmax']

    layers = collect_target_modules(model, layer_type)
    fcs = {}
    for l_name, layer in layers.items():
        name2fc = collect_target_modules(layer, nn.Linear, prefix=l_name)
        fcs.update(name2fc)

    smooth_layers(layers, fc2fcs, norm2fcs, act_scales, -1, device)

    rmsnorms = collect_target_modules(model, norm_type)

    for name, linear in fcs.items():
        linear.to(device)
        q_linear = QLinear.from_float(linear)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, q_linear)
        linear.to('cpu')

    for name, norm in rmsnorms.items():
        norm.to(device)
        q_norm = QRMSNorm.from_float(norm)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, q_norm)
        norm.to('cpu')

    if hasattr(model.config, 'auto_map'):
        model.config.auto_map.update(AUTO_MAP[type(model).__name__])
    else:
        model.config.auto_map = AUTO_MAP[type(model).__name__]

    model.save_pretrained(work_dir)
    tokenizer.save_pretrained(work_dir)

    shutil.copy(MODEL_PATH_MAP[type(model).__name__], work_dir)


if __name__ == '__main__':
    fire.Fire(smooth_quant)
