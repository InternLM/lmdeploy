# Copyright (c) OpenMMLab. All rights reserved.
import _turbomind as _tm

from ._base import Builder, SplitSide

Qwen35MtpWeightConfig = _tm.Qwen35MtpWeightConfig


class Qwen35MtpWeightBuilder(Builder):

    def add_fc(self, linear):
        self._add_linear('fc', linear, split_side=SplitSide.OUTPUT)
