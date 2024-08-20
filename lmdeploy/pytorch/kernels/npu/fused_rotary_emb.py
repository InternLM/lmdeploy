# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch_npu
from torch import Tensor


def fused_rotary_emb(
    query_states: Tensor,
    key_states: Tensor,
    position_ids: torch.LongTensor,
    inv_freq: Tensor,
    scaling_factor: float,
    out_q: Tensor = None,
    out_k: Tensor = None,
    context=None,
):
    _, bs, head, dim = query_states.shape
    _, _, numKeyValueHeads, _ = key_states.shape
    query_states = query_states.view(bs, head * dim)
    key_states = key_states.view(bs, numKeyValueHeads * dim)
    position_ids = position_ids.squeeze(0).unsqueeze(-1)
    pos_freq = position_ids / scaling_factor * inv_freq
    if not (hasattr(context, 'cos') or hasattr(context, 'sin')):
        cos = (torch.cos(pos_freq).view(position_ids.shape[0], 1,
                                        -1).repeat(1, 1,
                                                   2).to(query_states.dtype))
        sin = (torch.sin(pos_freq).view(position_ids.shape[0], 1,
                                        -1).repeat(1, 1,
                                                   2).to(query_states.dtype))
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        setattr(context, 'cos', cos)
        setattr(context, 'sin', sin)

    query_states = query_states.reshape(1, bs, head, dim)
    key_states = key_states.reshape(1, bs, numKeyValueHeads, dim)
    torch_npu.npu_apply_rotary_pos_emb(query_states, key_states, context.cos,
                                       context.sin)

    return query_states.view(1, bs, head,
                             dim), key_states.view(1, bs, numKeyValueHeads,
                                                   dim)
