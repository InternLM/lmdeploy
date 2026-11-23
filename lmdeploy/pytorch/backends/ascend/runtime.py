# Copyright (c) 2024, DeepLink. All rights reserved.
"""Explicit Ascend initialization; no LMDeploy or Transformers class
patching."""

import os
import pathlib
from functools import lru_cache

import torch


@lru_cache(None)
def register_atb_extensions():
    import torch_npu
    from torch_npu.utils._path_manager import PathManager

    npu_path = pathlib.Path(torch_npu.__file__).parents[0]
    atb_so_path = os.path.join(npu_path, 'lib', 'libop_plugin_atb.so')
    try:
        PathManager.check_directory_path_readable(atb_so_path)
        torch.ops.load_library(atb_so_path)
    except OSError as e:
        nnal_ex = None
        nnal_strerror = ''
        if 'libatb.so' in str(e):
            nnal_strerror = (
                'Please check that the nnal package is installed. '
                "Please run 'source set_env.sh' in the NNAL installation path."
            )
        if 'undefined symbol' in str(e):
            nnal_strerror = (
                'Please check the version of the NNAL package. '
                'An undefined symbol was found, '
                'which may be caused by a version mismatch between NNAL and torch_npu.'
            )
        nnal_ex = OSError(e.errno, nnal_strerror)
        nnal_ex.__traceback__ = e.__traceback__
        raise nnal_ex from e


@lru_cache(None)
def initialize():
    from .version import ensure_ascend_runtime

    ensure_ascend_runtime()
    # The existing engine's CUDA-shaped stream/device API is provided by torch_npu.
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    register_atb_extensions()
    try:
        import triton  # noqa: F401
    except ImportError:
        from lmdeploy.utils import get_logger

        get_logger('lmdeploy').warning('Install triton-ascend for Ascend GDN and Triton rejection sampling.')
    else:
        from lmdeploy.pytorch.kernels.ascend.triton_ops.triton_utils import init_device_properties_triton

        init_device_properties_triton()
