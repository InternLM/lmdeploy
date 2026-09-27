# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .base import BuildSpec


class MultinomialSamplingImpl(ABC):
    """Multinomial sampling implementation api."""

    @abstractmethod
    def forward(scores: torch.Tensor, seeds: torch.LongTensor, offsets: torch.LongTensor, indices: torch.Tensor = None):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class MultinomialSamplingBuildSpec(BuildSpec[MultinomialSamplingImpl]):
    """Request construction of a multinomial sampling operator."""
