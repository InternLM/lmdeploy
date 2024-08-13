# Copyright (c) OpenMMLab. All rights reserved.
import deeplink_ext.cpp_extensions as ext
from torch import Tensor


def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor,
    position_ids_1d: Tensor,
    q_embed=None,
    k_embed=None,
    context=None,
):
    bs, head, dim = query_states.shape
    numKeyValueHeads = key_states.shape[1]
    query_states = query_states.reshape(bs, head * dim)
    key_states = key_states.reshape(bs, numKeyValueHeads * dim)
    if not (hasattr(context, 'cos') or hasattr(context, 'sin')):
        if cos.dim() == 3:
            cos = cos[:, position_ids_1d].view(1, bs, 1, -1)
            sin = sin[:, position_ids_1d].view(1, bs, 1, -1)
        elif cos.dim() == 2:
            cos = cos[position_ids_1d].view(1, bs, 1, -1)
            sin = sin[position_ids_1d].view(1, bs, 1, -1)
        else:
            raise RuntimeError(f'Unsupport cos dim: {cos.dim()}')
        setattr(context, 'cos', cos)
        setattr(context, 'sin', sin)
    ext.rotary_embedding_v2(query_states, key_states, context.cos, context.sin,
                            dim)
    return query_states.view(bs, head,
                             dim), key_states.view(bs, numKeyValueHeads, dim)
