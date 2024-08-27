# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from lmdeploy.pytorch.model_inputs import StepContextManager


@dataclass
class AdapterInfo:
    """Adapter information."""
    in_features: int
    out_features: int
    ranks: torch.Tensor
    scalings: torch.Tensor
    rank_offsets: torch.Tensor
    a_cache: torch.Tensor
    b_cache: torch.Tensor
    base_slice: slice
    max_rank: int


class SLoRAImpl(ABC):
    """slora implementation api."""

    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                base_output: torch.Tensor,
                adapter_info: AdapterInfo,
                ctx_mgr: StepContextManager,
                colwise: bool,
                is_tp: bool = True):
        """forward."""
        raise NotImplementedError


class SLoRABuilder(ABC):
    """slora implementation builder."""

    @staticmethod
    @abstractmethod
    def build():
        """build."""
        raise NotImplementedError
