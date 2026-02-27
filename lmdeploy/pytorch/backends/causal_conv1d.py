# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch


class CausalConv1dImpl(ABC):
    """CausalConv1d implementation api."""

    @abstractmethod
    def conv1d_fn(self,
                  x: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor | None = None,
                  seq_idx: torch.Tensor | None = None,
                  return_final_states: bool = False,
                  activation: str | None = None):
        """forward."""
        raise NotImplementedError

    @abstractmethod
    def update_fn(self,
                  x: torch.Tensor,
                  conv_state: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor | None = None,
                  activation: str | None = None,
                  conv_state_indices: torch.Tensor | None = None):
        """Update conv state."""
        raise NotImplementedError


class CausalConv1dBuilder(ABC):
    """CausalConv1d implementation builder."""

    @staticmethod
    @abstractmethod
    def build():
        """build."""
        raise NotImplementedError
