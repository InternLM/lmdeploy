# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch


class MultinomialSamplingImpl(ABC):

    @abstractmethod
    def forward(scores: torch.Tensor,
                seeds: torch.LongTensor,
                offsets: torch.LongTensor,
                indices: torch.Tensor = None):
        raise NotImplementedError


class MultinomialSamplingBuilder(ABC):

    @staticmethod
    @abstractmethod
    def build():
        raise NotImplementedError
