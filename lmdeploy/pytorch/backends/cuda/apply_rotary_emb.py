# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from lmdeploy.pytorch.kernels.cuda import apply_rotary_pos_emb

from ..apply_rotary_emb import ApplyRotaryEmbBuilder, ApplyRotaryEmbImpl


class TritonApplyRotaryEmbImpl(ApplyRotaryEmbImpl):
    """Apply rotary embedding implementation."""

    def forward(self,
                query: Tensor,
                key: Tensor,
                cos: Tensor,
                sin: Tensor,
                inplace: bool = True):
        """forward."""
        if inplace:
            q_embed = query
            k_embed = key
        else:
            q_embed = torch.empty_like(query)
            k_embed = torch.empty_like(key)
        return apply_rotary_pos_emb(query, key, cos, sin, q_embed, k_embed)


class TritonApplyRotaryEmbBuilder(ApplyRotaryEmbBuilder):
    """Apply rotary embedding implementation builder."""

    @staticmethod
    def build():
        """build implementation."""
        return TritonApplyRotaryEmbImpl()
