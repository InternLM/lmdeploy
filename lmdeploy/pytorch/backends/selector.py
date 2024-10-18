# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.devices import get_device_manager


def get_backend():
    """get attention backend."""
    device_mgr = get_device_manager()
    device_ctx = device_mgr.current_context()

    device_type = device_ctx.device_type

    if device_type == 'cuda':
        from .cuda import CudaOpsBackend
        return CudaOpsBackend
    if device_type == 'ascend':
        from .ascend import AscendOpsBackend
        return AscendOpsBackend
    else:
        raise RuntimeError(f'Unsupported device type: {device_type}')
