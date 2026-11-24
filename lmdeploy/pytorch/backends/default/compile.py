# Copyright (c) OpenMMLab. All rights reserved.
"""Call-site compilation policy for shared vision helpers."""
from functools import wraps

import torch

from lmdeploy.pytorch.devices import get_device_manager


def compile_for_cuda(**options):
    """Compile shared helpers only when called in the CUDA device context.

    The decorator supports both CUDA and Ascend callers. The first CUDA call
    creates and caches a ``torch.compile`` wrapper using ``options``. Ascend
    and other device contexts execute the original function eagerly, even if
    the CUDA wrapper has already been created.

    InternVL and InternVL3-HF use this for their pre/post RMSNorm helpers,
    preserving the eager Ascend behavior previously provided by dlinfer
    without replacing the global ``torch.compile`` function.

    This function-level compilation policy is separate from graph capture
    and replay. Ascend graph execution is handled by ``AscendGraphRunner``
    using ``torch.npu.NPUGraph``.
    """
    def decorate(func):
        compiled = None
        @wraps(func)
        def call(*args, **kwargs):
            nonlocal compiled
            if get_device_manager().current_context().device_type != 'cuda':
                return func(*args, **kwargs)
            if compiled is None:
                compiled = torch.compile(func, **options)
            return compiled(*args, **kwargs)
        return call
    return decorate
