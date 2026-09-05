# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch

from .gated_delta_rule import GatedDeltaMeta


class CausalConv1dImpl(ABC):
    """CausalConv1d implementation api."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_state: torch.Tensor,
        gated_delta_meta: GatedDeltaMeta,
        activation: str,
    ) -> torch.Tensor:
        """Run causal convolution and update its state cache."""
        raise NotImplementedError

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


class CausalConv1dBuilder(ABC):
    """CausalConv1d implementation builder."""

    @staticmethod
    @abstractmethod
    def build():
        """build."""
        raise NotImplementedError
