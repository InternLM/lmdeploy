# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

from torch import nn

from lmdeploy.pytorch.model_inputs import StepContextManager


class LinearW4A16Impl(ABC, nn.Module):

    @abstractmethod
    def forward(self, x, all_reduce: bool = False):
        raise NotImplementedError


class LinearW4A16Builder(ABC):

    @staticmethod
    @abstractmethod
    def build(mod: nn.Module, ctx_mgr: StepContextManager = None):
        raise NotImplementedError
