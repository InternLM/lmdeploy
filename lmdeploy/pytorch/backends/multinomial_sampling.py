# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch


class MultinomialSamplingImpl(ABC):
    """multinomial sampling implementation api."""

    @abstractmethod
    def forward(scores: torch.Tensor,
                seeds: torch.LongTensor,
                offsets: torch.LongTensor,
                indices: torch.Tensor = None):
        """forward."""
        raise NotImplementedError


class MultinomialSamplingBuilder(ABC):
    """multinomial sampling implementation builder."""

    @staticmethod
    @abstractmethod
    def build():
        """build."""
        raise NotImplementedError
