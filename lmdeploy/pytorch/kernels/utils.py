# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
from packaging import version

if version.parse(triton.__version__) <= version.parse('2.2.0'):

    def get_kernel_meta(tensor: torch.Tensor):
        """kernel meta."""
        from triton import get_cuda_stream

        device = tensor.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)
else:

    def get_kernel_meta(tensor: torch.Tensor):
        """kernel meta."""
        return dict()
