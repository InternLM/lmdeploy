# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from lmdeploy.pytorch.model_inputs import StepContextManager


@dataclass
class AdapterInfo:
    """Adapter information."""
    in_features: int
    out_features: int
    ranks: torch.Tensor
    scalings: torch.Tensor
    base_slice: slice
    rank_offsets: torch.Tensor = field(init=False)
    max_rank: int = field(init=False)

    def __post_init__(self):
        """post init."""
        ranks = self.ranks
        rank_offsets = ranks.cumsum(0) - ranks
        max_rank = ranks.max().item()
        self.rank_offsets = rank_offsets
        self.max_rank = max_rank


class LoRAImpl(ABC):
    """lora implementation."""

    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                base_output: torch.Tensor,
                lora_A: torch.Tensor,
                lora_B: torch.Tensor,
                adapter_info: AdapterInfo,
                ctx_mgr: StepContextManager,
                colwise: bool,
                is_tp: bool = True):
        """forward."""
        raise NotImplementedError


class LoRABuilder(ABC):
    """lora implementation builder."""

    @staticmethod
    @abstractmethod
    def build():
        """build."""
        raise NotImplementedError
