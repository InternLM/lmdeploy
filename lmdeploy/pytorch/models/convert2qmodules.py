# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from .q_modules import QLayerNorm, QLinear, QRMSNorm


def convert_decoder_layer(module, norm_type):
    """Converts a given module's child layers from regular Linear or RMSNorm to
    their Quantized versions (QLinear, QRMSNorm).

    The conversion is done in place.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_child = QLinear.from_float(child, initialization=False)
            setattr(module, name, new_child)
        elif type(child).__name__ == norm_type:
            if norm_type == 'LayerNorm':
                new_child = QLayerNorm.from_float(child, initialization=False)
                setattr(module, name, new_child)
            else:
                new_child = QRMSNorm.from_float(child, initialization=False)
                setattr(module, name, new_child)
        else:
            convert_decoder_layer(child, norm_type)


def convert(module, layer_type, norm_type):
    """Recursively traverses through given PyTorch module and identifies child
    layers that match the specified layer_type and norm_type for conversion to
    their Quantized counterparts.

    The conversion is done using the `convert_decoder_layer` function.
    """
    for child in module.children():
        if type(child).__name__ == layer_type:
            convert_decoder_layer(child, norm_type)
        else:
            convert(child, layer_type, norm_type)


def convert_to_qmodules(model):
    """Convert all Linear and RMSNorm in the decoder layers of the model into
    their Quantized versions (QLinear, QRMSNorm)."""
    from lmdeploy.lite.apis.calibrate import LAYER_TYPE_MAP, NORM_TYPE_MAP
    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    norm_type = NORM_TYPE_MAP[type(model).__name__]
    convert(model, layer_type, norm_type)
    return
