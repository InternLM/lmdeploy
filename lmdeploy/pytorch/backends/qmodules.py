# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch
from torch import nn

from lmdeploy.pytorch.model_inputs import StepContextManager


class RMSNormW8A8Impl(ABC, nn.Module):
    """RMS norm w8a8 implementation api."""

    @abstractmethod
    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        raise NotImplementedError


class RMSNormW8A8Builder(ABC):
    """RMS norm w8a8 implementation builder."""

    @staticmethod
    @abstractmethod
    def build(weight: torch.Tensor, eps: float = 1e-6):
        """build."""
        raise NotImplementedError


class LinearW8A8Impl(ABC, nn.Module):
    """linear w8a8 implementation api."""

    @abstractmethod
    def forward(self, x, all_reduce: bool = False):
        """forward."""
        raise NotImplementedError


class LinearW8A8Builder(ABC):
    """linear w8a8 implementation builder."""

    @staticmethod
    @abstractmethod
    def build(mod: nn.Module, ctx_mgr: StepContextManager = None):
        """build."""
        raise NotImplementedError
