# Copyright (c) OpenMMLab. All rights reserved.
import functools

import torch
import triton
from packaging import version

WARPS_PER_SM = {
    (8, 0): 64,
    (8, 6): 48,
    (8, 7): 48,
    (8, 9): 48,
    (9, 0): 64,
    (10, 0): 64,
    (10, 1): 48,
    (11, 0): 48,
    (12, 0): 48,
}

BLOCKS_PER_SM = {
    (8, 0): 32,
    (8, 6): 16,
    (8, 7): 16,
    (8, 9): 24,
    (9, 0): 32,
    (10, 0): 32,
    (10, 1): 24,
    (11, 0): 24,
    (12, 0): 24,
}


@functools.lru_cache
def get_device_props(device=None):
    if device is None:
        device = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device)

    warps_per_sm = WARPS_PER_SM.get((props.major, props.minor), 32)
    blocks_per_sm = BLOCKS_PER_SM.get((props.major, props.minor), warps_per_sm // 2)
    out = dict(
        multi_processor_count=props.multi_processor_count,
        warps_per_sm=warps_per_sm,
        blocks_per_sm=blocks_per_sm,
    )
    return out


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == 'cuda'


@functools.lru_cache
def supports_tma():
    ret = is_cuda() and torch.cuda.get_device_capability()[0] >= 9
    if not ret:
        return False

    TRITON_VERSION = version.parse(triton.__version__)
    VALID_VERSION = version.parse('3.2.0')
    return TRITON_VERSION >= VALID_VERSION


# Copy from:
# https://github.com/triton-lang/triton/blob/main/python/triton/tools/experimental_descriptor.py
class TmaDescKernelParam:
    TMA_DESC_SIZE = 128

    def __init__(self):
        self.desc = torch.empty(self.TMA_DESC_SIZE, dtype=torch.uint8, device='cpu')

    def fill_(self, ptr, dims, block_dims, element_size):
        assert len(dims) == len(block_dims)
        assert 1 <= len(dims) <= 2
        assert self.desc.data_ptr() % 64 == 0

        if len(dims) == 1:
            triton.runtime.driver.active.utils.fill_1d_tma_descriptor(ptr, dims[0], block_dims[0], element_size,
                                                                      self.desc.data_ptr())
        else:
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor(ptr, dims[0], dims[1], block_dims[0],
                                                                      block_dims[1], element_size, self.desc.data_ptr())

    # Return a CUtensorMap* pointer in host memory
    def tma_desc_cpu_ptr(self):
        return self.desc.data_ptr()


# Copy from:
# https://github.com/triton-lang/triton/blob/main/python/triton/tools/experimental_descriptor.py
def create_1d_tma_descriptor_custom(ptr, dim, block_dim, element_size):
    desc = TmaDescKernelParam()
    desc.fill_(ptr, [dim], [block_dim], element_size)
    return desc


# Copy from:
# https://github.com/triton-lang/triton/blob/main/python/triton/tools/experimental_descriptor.py
def create_2d_tma_descriptor_custom(ptr, dim1, dim0, block_dim1, block_dim0, element_size):
    desc = TmaDescKernelParam()
    desc.fill_(ptr, [dim1, dim0], [block_dim1, block_dim0], element_size)
    return desc


try:
    from triton.tools.experimental_descriptor import create_1d_tma_descriptor, create_2d_tma_descriptor  # noqa
except BaseException:
    create_1d_tma_descriptor = create_1d_tma_descriptor_custom
    create_2d_tma_descriptor = create_2d_tma_descriptor_custom


class TmaAutoTuneHelper:

    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:

        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_2d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_2d_tma_descriptor)
        self.descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        self.descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device='cpu', dtype=torch.int8)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size):
        desc_x = self.descriptors[name]
        assert desc_x.data_ptr() % 64 == 0
        self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr())

    def get_tma_descriptor_kernel_param(self, name):
        assert self.descriptors[name] is not None
        return self.KernelParamWrapper(self.descriptors[name])
