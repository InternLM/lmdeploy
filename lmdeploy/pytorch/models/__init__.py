# Copyright (c) OpenMMLab. All rights reserved.
from .convert2qmodules import convert_to_qmodules
from .patch import patch
from .q_modules import QLayerNorm, QLinear, QRMSNorm

__all__ = ['patch', 'QLinear', 'QRMSNorm', 'QLayerNorm', 'convert_to_qmodules']
