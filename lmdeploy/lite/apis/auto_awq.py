# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.lite.quantization.awq import (FC_FCS_MAP, NORM_FCS_MAP,
                                            awq_layers, quant_weights,
                                            smooth_layers)
from lmdeploy.lite.utils import collect_target_modules

from .calibrate import calibrate

# from lmdeploy.lite.utils.export_turbomind import export_turbomind_config

LAYER_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMDecoderLayer',
    'InternLM2ForCausalLM': 'InternLM2DecoderLayer',
    'QWenLMHeadModel': 'QWenBlock',
    'Qwen2ForCausalLM': 'Qwen2DecoderLayer',
    'BaiChuanForCausalLM': 'DecoderLayer',  # Baichuan 7B
    'BaichuanForCausalLM': 'DecoderLayer',  # Baichuan2 7B
    'LlamaForCausalLM': 'LlamaDecoderLayer',
}
NORM_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMRMSNorm',
    'InternLM2ForCausalLM': 'InternLM2RMSNorm',
    'QWenLMHeadModel': 'RMSNorm',
    'Qwen2ForCausalLM': 'Qwen2RMSNorm',
    'BaiChuanForCausalLM': 'RMSNorm',  # Baichuan 7B
    'BaichuanForCausalLM': 'RMSNorm',  # Baichuan2 7B
    'LlamaForCausalLM': 'LlamaRMSNorm',
}


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
             device: str = 'cuda'):
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
    """
    model, tokenizer, work_dir = calibrate(model,
                                           calib_dataset,
                                           calib_samples,
                                           calib_seqlen,
                                           work_dir,
                                           device,
                                           w_bits=w_bits,
                                           w_group_size=w_group_size,
                                           search_scale=search_scale,
                                           batch_size=batch_size)

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
    quant_weights(model, fcs, w_bits, w_sym, w_group_size, device)
    quantization_config = dict(quant_method='awq',
                               version='gemm',
                               bits=w_bits,
                               group_size=w_group_size,
                               zero_point=not w_sym)
    model.config.update(dict(quantization_config=quantization_config))

    model.save_pretrained(work_dir,
                          max_shard_size='2GB',
                          safe_serialization=False)
    tokenizer.save_pretrained(work_dir)


if __name__ == '__main__':
    import fire

    fire.Fire(auto_awq)
