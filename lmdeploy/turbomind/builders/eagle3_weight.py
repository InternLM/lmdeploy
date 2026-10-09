# Copyright (c) OpenMMLab. All rights reserved.
import _turbomind as _tm

from ._base import Builder

Eagle3WeightConfig = _tm.Eagle3WeightConfig


class Eagle3WeightBuilder(Builder):
    """Builder for the EAGLE3 method weight tree at ModelWeight.spec."""

    def add_target_hidden_proj(self, linear):
        self._add_linear('target_hidden_proj', linear, split_side=None)
