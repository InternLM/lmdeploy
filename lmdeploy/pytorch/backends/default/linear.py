# Copyright (c) OpenMMLab. All rights reserved.
from torch import distributed as dist
from torch import nn

from lmdeploy.pytorch.model_inputs import StepContextManager

from ..linear import LinearBuilder, LinearImpl


class DefaultLinearImpl(LinearImpl):
    """Linear implementation api."""

    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod

    def forward(self, x, all_reduce: bool = False):
        """forward."""
        out = self.mod(x)
        if all_reduce:
            dist.all_reduce(out)
        return out


class DefaultLinearBuilder(LinearBuilder):
    """linear implementation builder."""

    @staticmethod
    def build(mod: nn.Module, ctx_mgr: StepContextManager = None):
        """build."""
        return DefaultLinearImpl(mod)
