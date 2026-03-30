# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager


def _get_backend():
    """Get device backend implement."""
    device_mgr = get_device_manager()
    device_ctx = device_mgr.current_context()

    device_type = device_ctx.device_type

    if device_type == 'cuda':
        from .cuda import CudaOpsBackend
        return CudaOpsBackend
    if device_type == 'ascend':
        from .dlinfer.ascend import AscendOpsBackend
        return AscendOpsBackend
    if device_type == 'maca':
        from .dlinfer.maca import MacaOpsBackend
        return MacaOpsBackend
    if device_type == 'camb':
        from .dlinfer.camb import CambOpsBackend
        return CambOpsBackend
    else:
        raise RuntimeError(f'Unsupported device type: {device_type}')


def get_backend(backend_type: str = None):
    """Get device backend."""
    if backend_type is None:
        return _get_backend()
    else:
        device_ctx = DeviceContext(backend_type)
        device_mgr = get_device_manager()
        with device_mgr.context(device_ctx):
            return _get_backend()


def init_backend(backend_type: str):
    """Init device backend."""
    backend = get_backend(backend_type)
    backend.init()
