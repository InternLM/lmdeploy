# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from torch import Tensor


def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    q_embed: Tensor=None,
    k_embed: Tensor=None,
):
    bs, head, dim = query_states.shape
    num_kv_heads = key_states.shape[1]
    query_states_reshaped = query_states.reshape(1, bs, head, dim)
    key_states_reshaped = key_states.reshape(1, bs, num_kv_heads, dim)
    cos_reshaped = cos.reshape(1, bs, 1, -1)
    sin_reshaped = sin.reshape(1, bs, 1, -1)
    ext_ops.apply_rotary_pos_emb(query_states_reshaped, key_states_reshaped,
                                 cos_reshaped, sin_reshaped, None, None, None)
    if q_embed is None:
        q_embed = query_states
    elif q_embed is not query_states:
        q_embed.copy_(query_states)

    if k_embed is None:
        k_embed = key_states
    elif k_embed is not key_states:
        k_embed.copy_(key_states)

    return q_embed, k_embed
