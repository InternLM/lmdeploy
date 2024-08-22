# Copyright (c) OpenMMLab. All rights reserved.
import infer_ext.ops as ext_ops
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
    num_kv_heads = key_states.shape[1]
    query_states_reshaped = query_states.reshape(1, bs, head, dim)
    key_states_reshaped = key_states.reshape(1, bs, num_kv_heads, dim)
    if not (hasattr(context, 'cos') or hasattr(context, 'sin')):
        cos = cos[position_ids_1d].view(1, bs, 1, -1)
        sin = sin[position_ids_1d].view(1, bs, 1, -1)
        if context:
            setattr(context, 'cos', cos)
            setattr(context, 'sin', sin)
    cached_cos = context.cos if context else cos
    cached_sin = context.sin if context else sin
    ext_ops.apply_rotary_pos_emb(query_states_reshaped, key_states_reshaped,
                                 cached_cos, cached_sin, None, None)
    if q_embed is None:
        q_embed = query_states
    else:
        q_embed.copy_(query_states)
    if k_embed is None:
        k_embed = key_states
    else:
        k_embed.copy_(key_states)
    return q_embed, k_embed
