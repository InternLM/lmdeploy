# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch


class GatedDeltaRuleImpl(ABC):
    """Gated Delta Rule implementation api."""

    @abstractmethod
    def chunk_gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        scale: float | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        output_final_state: bool = False,
        spec_state_offsets: torch.Tensor | None = None,
    ):
        """forward."""
        raise NotImplementedError

    @abstractmethod
    def fused_recurrent_gated_delta_rule(self,
                                         q: torch.Tensor,
                                         k: torch.Tensor,
                                         v: torch.Tensor,
                                         g: torch.Tensor | None = None,
                                         beta: torch.Tensor | None = None,
                                         initial_state: torch.Tensor | None = None,
                                         state_indices: torch.Tensor | None = None,
                                         scale: float | None = None,
                                         use_qk_l2norm_in_kernel: bool = False,
                                         output_final_state: bool = False):
        """forward."""
        raise NotImplementedError


class GatedDeltaRuleBuilder(ABC):
    """Gated Delta Rule implementation builder."""

    @staticmethod
    @abstractmethod
    def build() -> GatedDeltaRuleImpl:
        """build."""
        raise NotImplementedError
