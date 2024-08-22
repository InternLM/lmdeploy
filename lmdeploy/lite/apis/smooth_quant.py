# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional

import fire
import torch
from torch import nn

from lmdeploy.lite.apis.calibrate import calibrate
from lmdeploy.lite.quantization.awq import (FC_FCS_MAP, NORM_FCS_MAP,
                                            smooth_layers)
from lmdeploy.lite.utils import collect_target_modules
from lmdeploy.pytorch.models import QLayerNorm, QLinear, QRMSNorm

from .calibrate import LAYER_TYPE_MAP, NORM_TYPE_MAP


def smooth_quant(model: str,
                 work_dir: str = './work_dir',
                 calib_dataset: str = 'ptb',
                 calib_samples: int = 128,
                 calib_seqlen: int = 2048,
                 batch_size: int = 1,
                 w_bits: int = 8,
                 device: str = 'cuda',
                 calib_image: Optional[str] = None):

    model_path = model
    vl_model, model, tokenizer, work_dir = calibrate(model,
                                                     calib_dataset,
                                                     calib_samples,
                                                     calib_seqlen,
                                                     work_dir,
                                                     device,
                                                     w_bits=w_bits,
                                                     w_group_size=-1,
                                                     batch_size=batch_size,
                                                     calib_image=calib_image)

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

    _smooth_quant(model, act_scales, device)
    if calib_image is not None and vl_model is not None:
        # TODO models other than InternVL
        act_scales = torch.load(work_dir / 'vision_inputs_stats.pth')['absmax']
        _smooth_quant(vl_model.vision_model, act_scales, device)
        vl_model.vision_model.config.update(
            dict(lmdeploy_quant_config=dict(quant_method='smooth_quant',
                                            bits=w_bits)))

    if vl_model:
        from .auto_awq import save_vl_model
        save_vl_model(vl_model, model_path, work_dir)
    else:
        model.config.update(
            dict(lmdeploy_quant_config=dict(quant_method='smooth_quant',
                                            bits=w_bits)))
        model.save_pretrained(work_dir,
                              max_shard_size='2GB',
                              safe_serialization=False)
    tokenizer.save_pretrained(work_dir)


def _smooth_quant(model, act_scales, device):
    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    norm_type = NORM_TYPE_MAP[type(model).__name__]
    fc2fcs = FC_FCS_MAP[layer_type]
    norm2fcs = NORM_FCS_MAP[layer_type]

    layers = collect_target_modules(model, layer_type)
    norms = collect_target_modules(model, norm_type)
    fcs = {}
    for l_name, layer in layers.items():
        name2fc = collect_target_modules(layer, nn.Linear, prefix=l_name)
        fcs.update(name2fc)

    smooth_layers(layers, fc2fcs, norm2fcs, act_scales, -1, device)

    for name, linear in fcs.items():
        linear.to(device)
        q_linear = QLinear.from_float(linear)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, q_linear)
        linear.to('cpu')

    for name, norm in norms.items():
        norm.to(device)
        if norm_type == 'LayerNorm':
            q_norm = QLayerNorm.from_float(norm)
        else:
            q_norm = QRMSNorm.from_float(norm)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, q_norm)
        norm.to('cpu')


if __name__ == '__main__':
    fire.Fire(smooth_quant)
