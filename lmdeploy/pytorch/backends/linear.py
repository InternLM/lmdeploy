# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

from torch import nn

from lmdeploy.pytorch.model_inputs import StepContextManager


class LinearImpl(ABC, nn.Module):
    """Linear implementation api."""

    @abstractmethod
    def forward(self, x, all_reduce: bool = False):
        """forward."""
        raise NotImplementedError


class LinearBuilder(ABC):
    """linear implementation builder."""

    @staticmethod
    @abstractmethod
    def build(mod: nn.Module, ctx_mgr: StepContextManager = None):
        """build."""
        raise NotImplementedError
