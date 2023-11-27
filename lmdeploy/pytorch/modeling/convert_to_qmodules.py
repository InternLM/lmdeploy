# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from ..models import QLinear, QRMSNorm

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


def convert_decoder_layer(module, norm_type):
    """convert decoder layer."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_child = QLinear.from_float(child)
            setattr(module, name, new_child)
        elif type(child).__name__ == norm_type:
            new_child = QRMSNorm.from_float(child)
            setattr(module, name, new_child)
        else:
            convert_decoder_layer(child, norm_type)


def convert(module, layer_type, norm_type):
    """convert module."""
    for name, child in module.named_children():
        if type(child).__name__ == layer_type:
            convert_decoder_layer(child, norm_type)
        else:
            convert(child, layer_type, norm_type)


def convert_to_qmodules(model):
    """convert to qmodules."""
    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    norm_type = NORM_TYPE_MAP[type(model).__name__]
    convert(model, layer_type, norm_type)
    return
