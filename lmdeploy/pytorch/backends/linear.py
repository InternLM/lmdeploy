# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

from torch import nn

from lmdeploy.pytorch.model_inputs import StepContextManager


class LinearImpl(ABC, nn.Module):

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


class LinearBuilder(ABC):

    @staticmethod
    @abstractmethod
    def build(mod: nn.Module,
              ctx_mgr: StepContextManager = None,
              all_reduce: bool = False):
        raise NotImplementedError
