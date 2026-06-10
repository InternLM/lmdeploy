# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class GatedDeltaRuleImpl(ABC):
    """Gated Delta Rule implementation api."""

    def prepare_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log_exp: torch.Tensor,
        kv_ratio: int,
        use_qk_l2norm_in_kernel: bool = False,
        is_decoding: bool = False,
        init_token_mask: torch.Tensor | None = None,
    ):
        """Prepare q/k/g/beta for gated delta rule."""
        if b.dim() == 4:
            beta = b.sigmoid().flatten(-2, -1)
            a = a.float().flatten(-2, -1)
        else:
            beta = b.sigmoid()
            a = a.float()
        g = a_log_exp * F.softplus(a + dt_bias)
        if not is_decoding and init_token_mask is not None:
            g = g.masked_fill(init_token_mask[None, :, None], -1.0e6)
        if kv_ratio > 1:
            q = q.repeat_interleave(kv_ratio, dim=-2)
            k = k.repeat_interleave(kv_ratio, dim=-2)
        return q, k, g, beta, False

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
        transpose_state_layout: bool = False,
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
                                         output_final_state: bool = False,
                                         cache_seqlens: torch.Tensor | None = None,
                                         transpose_state_layout: bool = False):
        """forward."""
        raise NotImplementedError


class GatedDeltaRuleBuilder(ABC):
    """Gated Delta Rule implementation builder."""

    @staticmethod
    @abstractmethod
    def build() -> GatedDeltaRuleImpl:
        """build."""
        raise NotImplementedError
