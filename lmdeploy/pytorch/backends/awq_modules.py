# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

from torch import nn

from lmdeploy.pytorch.model_inputs import StepContextManager


class LinearW4A16Impl(ABC, nn.Module):
    """w4a16 linear implementation."""

    @abstractmethod
    def forward(self, x, all_reduce: bool = False):
        """forward."""
        raise NotImplementedError


class LinearW4A16Builder(ABC):
    """w4a16 linear implementation builder."""

    @staticmethod
    @abstractmethod
    def build(mod: nn.Module, ctx_mgr: StepContextManager = None):
        """build."""
        raise NotImplementedError
