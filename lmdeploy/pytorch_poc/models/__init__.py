# Copyright (c) OpenMMLab. All rights reserved.
from .patch import patch
from .q_modules import QLinear, QRMSNorm

__all__ = ['patch', 'QLinear', 'QRMSNorm']
