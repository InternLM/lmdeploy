# Copyright (c) OpenMMLab. All rights reserved.
from .device_manager import DefaultContext, DeviceContext, get_device_manager

__all__ = ['DeviceContext', 'DefaultContext', 'get_device_manager', 'is_ascend']


def is_ascend() -> bool:
    """Whether the active engine uses Ascend (without importing torch_npu)."""
    return get_device_manager().current_context().device_type == 'ascend'
