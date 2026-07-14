# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from lmdeploy.pytorch.kernels.dlinfer import apply_rotary_pos_emb

from ..apply_rotary_emb import ApplyRotaryEmbBuilder, ApplyRotaryEmbImpl


class DlinferApplyRotaryEmbImpl(ApplyRotaryEmbImpl):
    """Apply rotary embedding implementation."""

    def forward(self, query: Tensor, key: Tensor, cos: Tensor, sin: Tensor, inplace: bool = True):
        """forward."""
        if inplace:
            q_embed = None
            k_embed = None
        else:
            q_embed = query.new_empty(query.shape)
            k_embed = key.new_empty(key.shape)
        return apply_rotary_pos_emb(query, key, cos, sin, q_embed, k_embed)


class DlinferApplyRotaryEmbBuilder(ApplyRotaryEmbBuilder):
    """Apply rotary embedding implementation builder."""

    @staticmethod
    def build():
        """Build implementation."""
        return DlinferApplyRotaryEmbImpl()
