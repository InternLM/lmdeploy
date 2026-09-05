# Copyright (c) OpenMMLab. All rights reserved.
from .piecewise import eager_boundary
from .runner import CUDAGraphRunner

__all__ = ['CUDAGraphRunner', 'eager_boundary']
