# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch
from torch import nn

from lmdeploy.pytorch.model_inputs import StepContextManager


class RMSNormW8A8Impl(ABC, nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        raise NotImplementedError


class RMSNormW8A8Builder(ABC):

    @staticmethod
    @abstractmethod
    def build(weight: torch.Tensor, eps: float = 1e-6):
        raise NotImplementedError


class LinearW8A8Impl(ABC, nn.Module):

    @abstractmethod
    def forward(self, x, all_reduce: bool = False):
        raise NotImplementedError


class LinearW8A8Builder(ABC):

    @staticmethod
    @abstractmethod
    def build(mod: nn.Module, ctx_mgr: StepContextManager = None):
        raise NotImplementedError
