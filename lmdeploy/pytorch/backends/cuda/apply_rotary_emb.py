# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.kernels.cuda import apply_rotary_pos_emb

from ..apply_rotary_emb import ApplyRotaryEmbBuilder, ApplyRotaryEmbImpl


class TritonApplyRotaryEmbImpl(ApplyRotaryEmbImpl):

    def forward(self, query, key, cos, sin, inplace: bool = True):
        if inplace:
            q_embed = query
            k_embed = key
        else:
            q_embed = torch.empty_like(query)
            k_embed = torch.empty_like(key)
        return apply_rotary_pos_emb(query, key, cos, sin, q_embed, k_embed)


class TritonApplyRotaryEmbBuilder(ApplyRotaryEmbBuilder):

    @staticmethod
    def build():
        return TritonApplyRotaryEmbImpl()
