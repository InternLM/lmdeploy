# Copyright (c) OpenMMLab. All rights reserved.
"""Builder for the DFlash2 grouped-conv weight module."""

import _turbomind as _tm

from ._base import _CPP_TO_TORCH, Builder

GroupedConvWeightConfig = _tm.GroupedConvWeightConfig


class GroupedConvWeightBuilder(Builder):
    """Content-generated grouped conv: kernel_projection (linear) + base_kernel."""

    def add_kernel_projection(self, linear):
        # Conv coefficients depend on the full group layout; replicate for now
        # (TP-aware split is a later refinement).
        self._add_linear('kernel_projection', linear, split_side=None)

    def add_base_kernel(self, tensor):
        # The conv kernel reads base_kernel at the model compute dtype; the
        # checkpoint ships it bf16, so cast it to match (else the kernel
        # reinterprets bf16 bits as fp16 and produces garbage coefficients).
        dtype = _CPP_TO_TORCH.get(self.config.data_type)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        self._add_tensor('base_kernel', tensor, split_side=None)
