# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
from typing import Literal

import fire
import torch
from torch import nn

from lmdeploy.lite.apis.calibrate import LAYER_TYPE_MAP, NORM_TYPE_MAP, calibrate
from lmdeploy.lite.quantization.awq import FC_FCS_MAP, NORM_FCS_MAP, awq_layers, skipped_module, smooth_layers
from lmdeploy.lite.utils import collect_target_modules
from lmdeploy.pytorch.models import QLinear, QRMSNorm
from lmdeploy.utils import try_import_deeplink


def smooth_quant(model: str,
                 work_dir: str = './work_dir',
                 calib_dataset: str = 'ptb',
                 calib_samples: int = 128,
                 calib_seqlen: int = 2048,
                 search_scale: bool = False,
                 batch_size: int = 1,
                 w_bits: int = 8,
                 dtype: Literal['float16', 'bfloat16', 'auto'] = 'auto',
                 device: str = 'cuda',
                 quant_dtype: Literal['int8', 'fp8', 'float8_e4m3fn', 'float8_e5m2'] = 'int8',
                 revision: str = None,
                 download_dir: str = None):
    try_import_deeplink(device)
    if quant_dtype == 'fp8':
        quant_dtype = 'float8_e4m3fn'

    quant_dtype = getattr(torch, quant_dtype, torch.int8)
    if quant_dtype.is_floating_point:
        q_dtype_info = torch.finfo(quant_dtype)
    else:
        q_dtype_info = torch.iinfo(quant_dtype)

    assert q_dtype_info.bits == w_bits
    if not osp.exists(model):
        print(f'can\'t find model from local_path {model}, '
              'try to download from remote')
        from lmdeploy.utils import get_model
        model = get_model(model, revision=revision, download_dir=download_dir)
    model_path = model
    vl_model, model, tokenizer, work_dir = calibrate(model,
                                                     calib_dataset,
                                                     calib_samples,
                                                     calib_seqlen,
                                                     work_dir,
                                                     device,
                                                     w_bits=w_bits,
                                                     w_group_size=-1,
                                                     search_scale=search_scale,
                                                     dtype=dtype,
                                                     batch_size=batch_size)

    # calibrate function exports the calibration statistics
    # (inputs, outputs, keys and values) to `work_dir`.
    inp_stats = torch.load(work_dir / 'inputs_stats.pth', weights_only=True)
    act_scales = inp_stats['absmax']

    model_type = type(model).__name__
    if model_type not in LAYER_TYPE_MAP or model_type not in NORM_TYPE_MAP:
        raise RuntimeError(f'Currently, quantification and calibration of {model_type} are '
                           f'not supported. The supported model types are '
                           f"{', '.join(LAYER_TYPE_MAP.keys())}.")

    if model_type == 'QWenLMHeadModel':
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            raise RuntimeError('When using Qwen, you need to `pip install flash-attn` first, '
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

    if search_scale:
        awq_ratios = inp_stats['ratios']
        act_scales = inp_stats['absmean']
        awq_layers(layers, fc2fcs, norm2fcs, act_scales, awq_ratios, -1, device)
    else:
        smooth_layers(layers, fc2fcs, norm2fcs, act_scales, -1, device)

    rmsnorms = collect_target_modules(model, norm_type)

    for name, linear in fcs.items():
        if skipped_module(name):
            continue
        linear.to(device)
        q_linear = QLinear.from_float(linear, quant_dtype=quant_dtype)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, q_linear)
        linear.to('cpu')
        q_linear.to('cpu')
        torch.cuda.empty_cache()

    for name, norm in rmsnorms.items():
        if skipped_module(name):
            continue
        norm.to(device)
        q_norm = QRMSNorm.from_float(norm, quant_dtype=quant_dtype)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, q_norm)
        norm.to('cpu')
        q_linear.to('cpu')
        torch.cuda.empty_cache()

    quant_dtype_s = str(quant_dtype).split('.')[1]
    model.config.update(dict(quantization_config=dict(quant_method='smooth_quant', quant_dtype=f'{quant_dtype_s}')))

    if vl_model:
        from .auto_awq import save_vl_model
        save_vl_model(vl_model, model_path, work_dir)
    else:
        model.save_pretrained(work_dir, safe_serialization=True)
    tokenizer.save_pretrained(work_dir)


if __name__ == '__main__':
    fire.Fire(smooth_quant)
