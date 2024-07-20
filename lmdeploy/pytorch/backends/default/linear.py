# Copyright (c) OpenMMLab. All rights reserved.
from torch import distributed as dist
from torch import nn

from lmdeploy.pytorch.model_inputs import StepContextManager

from ..linear import LinearBuilder, LinearImpl


class DefaultLinearImpl(LinearImpl):

    def __init__(self, mod: nn.Module, all_reduce: bool = False):
        super().__init__()
        self.mod = mod
        self.all_reduce = all_reduce

    def forward(self, x):
        out = self.mod(x)
        if self.all_reduce:
            dist.all_reduce(out)
        return out


class DefaultLinearBuilder(LinearBuilder):

    @staticmethod
    def build(mod: nn.Module,
              ctx_mgr: StepContextManager = None,
              all_reduce: bool = False):
        return DefaultLinearImpl(mod, all_reduce)
