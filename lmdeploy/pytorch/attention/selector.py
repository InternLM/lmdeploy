# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.devices import get_device_manager
from lmdeploy.utils import get_logger


def get_attn_backend():
    """get attention backend."""
    device_mgr = get_device_manager()
    device_ctx = device_mgr.current_context()

    device_type = device_ctx.device_type

    if device_type == 'cuda':
        from .triton import TritonAttentionBackend
        return TritonAttentionBackend
    if device_type == 'ascend':
        from .ascend import AscendAttentionBackend
        return AscendAttentionBackend
    else:
        logger = get_logger('lmdeploy')
        logger.warning(f'Unsupported device type: {device_type}')
        return None
