# Copyright (c) OpenMMLab. All rights reserved.
import deeplink_ext.cpp_extensions as ext
import torch
from torch import Tensor


def fused_rotary_emb(
    query_states: Tensor,
    key_states: Tensor,
    position_ids: torch.LongTensor,
    inv_freq: Tensor,
    scaling_factor: float,
    out_q: Tensor = None,
    out_k: Tensor = None,
):
    _, bs, head, dim = query_states.shape
    _, _, numKeyValueHeads, _ = key_states.shape
    query_states = query_states.view(bs, head * dim)
    key_states = key_states.view(bs, numKeyValueHeads * dim)
    position_ids = position_ids.squeeze(0).unsqueeze(-1)
    pos_freq = position_ids / scaling_factor * inv_freq
    cos = (torch.cos(pos_freq).view(position_ids.shape[0], 1,
                                    -1).repeat(1, 1, 2).to(query_states.dtype))
    sin = (torch.sin(pos_freq).view(position_ids.shape[0], 1,
                                    -1).repeat(1, 1, 2).to(query_states.dtype))
    ext.rotary_embedding_v2(query_states, key_states, cos, sin, dim)
    query_states = query_states.view(1, bs, head, dim)
    key_states = key_states.view(1, bs, numKeyValueHeads, dim)
    return query_states, key_states
