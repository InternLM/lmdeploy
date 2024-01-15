# Copyright (c) OpenMMLab. All rights reserved.

import shutil

import fire
import torch
from torch import nn

from lmdeploy.lite.apis.calibrate import calibrate
from lmdeploy.lite.quantization.awq import (FC_FCS_MAP, NORM_FCS_MAP,
                                            smooth_layers)
from lmdeploy.lite.utils import collect_target_modules
from lmdeploy.pytorch.models import QLinear, QRMSNorm

LAYER_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMDecoderLayer',
    'InternLM2ForCausalLM': 'InternLM2DecoderLayer',
    'QWenLMHeadModel': 'QWenBlock',
    'BaiChuanForCausalLM': 'DecoderLayer',
    'LlamaForCausalLM': 'LlamaDecoderLayer',
}
NORM_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMRMSNorm',
    'InternLM2ForCausalLM': 'InternLM2RMSNorm',
    'QWenLMHeadModel': 'RMSNorm',
    'BaiChuanForCausalLM': 'RMSNorm',
    'LlamaForCausalLM': 'LlamaRMSNorm',
}

MODEL_PATH_MAP = {
    'InternLMForCausalLM': './lmdeploy/pytorch/modeling/modeling_internlm.py',
    'InternLM2ForCausalLM':
    './lmdeploy/pytorch/modeling/modeling_internlm2.py',
    'LlamaForCausalLM': './lmdeploy/pytorch/modeling/modeling_llama.py',
    'BaiChuanForCausalLM': './lmdeploy/pytorch/modeling/modeling_baichuan.py'
}

AUTO_MAP = {
    'InternLMForCausalLM': {
        'AutoConfig': 'configuration_internlm.InternLMConfig',
        'AutoModel': 'modeling_internlm.InternLMForCausalLM',
        'AutoModelForCausalLM': 'modeling_internlm.InternLMForCausalLM'
    },
    'InternLM2ForCausalLM': {
        'AutoConfig': 'configuration_internlm.InternLMConfig',
        'AutoModelForCausalLM': 'modeling_internlm2.InternLM2ForCausalLM',
        'AutoModel': 'modeling_internlm2.InternLM2ForCausalLM'
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


def smooth_quant(model: str,
                 work_dir: str = './work_dir',
                 calib_dataset: str = 'ptb',
                 calib_samples: int = 128,
                 calib_seqlen: int = 2048,
                 device: str = 'cuda'):

    model, tokenizer, work_dir = calibrate(model, calib_dataset, calib_samples,
                                           calib_seqlen, work_dir, device)

    # calibrate function exports the calibration statistics
    # (inputs, outputs, keys and values) to `work_dir`.
    inp_stats = torch.load(work_dir / 'inputs_stats.pth')
    act_scales = inp_stats['absmax']

    model_type = type(model).__name__
    if model_type not in LAYER_TYPE_MAP or model_type not in NORM_TYPE_MAP:
        raise RuntimeError(
            f'Currently, quantification and calibration of {model_type} are '
            f'not supported. The supported model types are '
            f"{', '.join(LAYER_TYPE_MAP.keys())}.")

    if model_type == 'QWenLMHeadModel':
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            raise RuntimeError(
                'When using Qwen, you need to `pip install flash-attn` first, '
                'otherwise calibration and quantification will not work '
                'properly.')

    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    norm_type = NORM_TYPE_MAP[type(model).__name__]
    fc2fcs = FC_FCS_MAP[layer_type]
    norm2fcs = NORM_FCS_MAP[layer_type]

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

    model.save_pretrained(work_dir,
                          max_shard_size='2GB',
                          safe_serialization=False)
    tokenizer.save_pretrained(work_dir)

    shutil.copy(MODEL_PATH_MAP[type(model).__name__], work_dir)


if __name__ == '__main__':
    fire.Fire(smooth_quant)
