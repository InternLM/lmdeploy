# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
from typing import Optional

import torch
from torch import nn

from lmdeploy.lite.quantization.awq import (FC_FCS_MAP, NORM_FCS_MAP,
                                            awq_layers, quant_weights,
                                            smooth_layers)
from lmdeploy.lite.utils import collect_target_modules

from .calibrate import LAYER_TYPE_MAP, NORM_TYPE_MAP, calibrate

# from lmdeploy.lite.utils.export_turbomind import export_turbomind_config

NORM_TYPE_MAP = NORM_TYPE_MAP  # legacy


def save_vl_model(vl_model, model_path, dst_path):
    safe_serialization = type(vl_model).__name__ == 'MGMLlamaForCausalLM'
    vl_model.save_pretrained(dst_path,
                             max_shard_size='2GB',
                             safe_serialization=safe_serialization)
    candidate = [
        'preprocessor_config.json', 'processor_config.json', 'vit',
        'generation_config.json', 'added_tokens.json'
    ]
    for name in candidate:
        tmp_path = osp.join(model_path, name)
        if osp.exists(tmp_path):
            if osp.isfile(tmp_path):
                shutil.copy(tmp_path, osp.join(dst_path, name))
            elif osp.isdir(tmp_path):
                shutil.copytree(tmp_path, osp.join(dst_path, name))


def auto_awq(model: str,
             work_dir: str = './work_dir',
             calib_dataset: str = 'ptb',
             calib_samples: int = 128,
             batch_size: int = 1,
             calib_seqlen: int = 2048,
             w_bits: int = 4,
             w_sym: bool = False,
             w_group_size: int = 128,
             search_scale: bool = False,
             device: str = 'cuda',
             revision: str = None,
             download_dir: str = None,
             calib_image: Optional[str] = None):
    """Perform weight quantization using AWQ algorithm.

    Args:
        model (str): The path of model in hf format.
        work_dir (str): The working directory to save results.
        calib_dataset (str): The calibration dataset name.
        calib_samples (int): The number of samples for calibration.
        batch_size (int): The batch size for running the calib samples.
            Low GPU mem requires small batch_size. Large batch_size
            reduces the calibration time while costs more VRAM.
        calib_seqlen (int): The sequence length for calibration.
        w_bits (int): Bit number for weight quantization.
        w_sym (bool): Whether to do symmetric quantization.
        w_group_size (int): Group size for weight quantization statistics.
        search_scale (bool): Whether search scale ratio. Default to False,
            which means only smooth quant with 0.5 ratio will be applied.
        device (str): Device type of running.
        revision (str): The specific model version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified,
            will use the default version.
        download_dir (str): Directory to download and load the weights,
            default to the default cache directory of huggingface.
    """
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
                                                     w_group_size=w_group_size,
                                                     search_scale=search_scale,
                                                     batch_size=batch_size,
                                                     calib_image=calib_image)

    if calib_image is not None and vl_model is not None:
        # TODO models other than InternVL
        act_scales = torch.load(work_dir / 'vision_inputs_stats.pth')['absmax']
        from .smooth_quant import _smooth_quant
        _smooth_quant(vl_model.vision_model, act_scales, device)
        vl_model.vision_model.config.update(
            dict(quantization_config=dict(quant_method='smooth_quant',
                                          bits=w_bits)))

    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    fc2fcs = FC_FCS_MAP[layer_type]
    norm2fcs = NORM_FCS_MAP[layer_type]
    input_stats = torch.load(work_dir / 'inputs_stats.pth')
    layers = collect_target_modules(model, layer_type)
    fcs = {}
    for l_name, layer in layers.items():
        name2fc = collect_target_modules(layer, nn.Linear, prefix=l_name)
        fcs.update(name2fc)

    if search_scale:
        awq_ratios = input_stats['ratios']
        act_scales = input_stats['absmean']
        awq_layers(layers, fc2fcs, norm2fcs, act_scales, awq_ratios,
                   w_group_size, device)
    else:
        act_scales = input_stats['absmax']
        smooth_layers(layers, fc2fcs, norm2fcs, act_scales, w_group_size,
                      device)
    quant_weights(model,
                  fcs,
                  w_bits,
                  w_sym,
                  w_group_size,
                  device,
                  skip_if_contains='lora')  # TODO quant lora weight
    quantization_config = dict(quant_method='awq',
                               version='gemm',
                               bits=w_bits,
                               group_size=w_group_size,
                               zero_point=not w_sym)
    model.config.update(dict(quantization_config=quantization_config))

    if vl_model:
        save_vl_model(vl_model, model_path, work_dir)
    else:
        model.save_pretrained(work_dir,
                              max_shard_size='2GB',
                              safe_serialization=False)
    tokenizer.save_pretrained(work_dir)


if __name__ == '__main__':
    import fire

    fire.Fire(auto_awq)
