# Copyright (c) OpenMMLab. All rights reserved.
"""Builder for the DFlash2 grouped-conv weight module."""

import _turbomind as _tm

from ._base import Builder

GroupedConvWeightConfig = _tm.GroupedConvWeightConfig


class GroupedConvWeightBuilder(Builder):
    """Content-generated grouped conv: kernel_projection (linear) + base_kernel."""

    def add_kernel_projection(self, linear):
        # Conv coefficients depend on the full group layout; replicate for now
        # (TP-aware split is a later refinement).
        self._add_linear('kernel_projection', linear, split_side=None)

    def add_base_kernel(self, tensor):
        self._add_tensor('base_kernel', tensor, split_side=None)
