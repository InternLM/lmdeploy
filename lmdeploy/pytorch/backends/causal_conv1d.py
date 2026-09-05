# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .base import BuildSpec


class CausalConv1dImpl(ABC):
    """CausalConv1d implementation api."""

    @abstractmethod
    def conv1d_fn(self,
                  x: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor | None = None,
                  seq_idx: torch.Tensor | None = None,
                  initial_states: torch.Tensor | None = None,
                  return_final_states: bool = False,
                  activation: str | None = None):
        """forward."""
        raise NotImplementedError

    @abstractmethod
    def update_fn(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        conv_state_indices: torch.Tensor | None = None,
        cache_seqlens: torch.Tensor | None = None,
    ):
        """Update conv state."""
        raise NotImplementedError


@dataclass(frozen=True)
class CausalConv1dBuildSpec(BuildSpec[CausalConv1dImpl]):
    """Request construction of a causal-convolution operator."""
