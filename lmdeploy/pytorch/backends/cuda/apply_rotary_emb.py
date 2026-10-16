# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from lmdeploy.pytorch.kernels.cuda import apply_rotary_pos_emb

from ..apply_rotary_emb import ApplyRotaryEmbImpl


class TritonApplyRotaryEmbImpl(ApplyRotaryEmbImpl):
    """Apply rotary embedding implementation."""

    def __init__(self, enable_fp32_compute: bool = False):
        self.enable_fp32_compute = enable_fp32_compute

    def forward(self,
                query: Tensor,
                key: Tensor,
                cos: Tensor,
                sin: Tensor,
                inplace: bool = True,
                complex_mode: bool = False):
        """forward."""
        if inplace:
            q_embed = query
            k_embed = key
        else:
            q_embed = torch.empty_like(query)
            k_embed = torch.empty_like(key)
        return apply_rotary_pos_emb(query, key, cos, sin, q_embed, k_embed,
                                    complex_mode=complex_mode, enable_fp32_compute=self.enable_fp32_compute)
