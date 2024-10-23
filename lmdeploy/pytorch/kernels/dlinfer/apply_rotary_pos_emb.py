# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from torch import Tensor


def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    q_embed: Tensor = None,
    k_embed: Tensor = None,
):
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    bs = query_states.shape[0]
    query_states_reshaped = query_states.unsqueeze(0)
    key_states_reshaped = key_states.unsqueeze(0)
    cos_reshaped = cos.reshape(1, bs, 1, -1)
    sin_reshaped = sin.reshape(1, bs, 1, -1)
    query_states_reshaped, key_states_reshaped = \
        ext_ops.apply_rotary_pos_emb(query_states_reshaped,
                                     key_states_reshaped,
                                     cos_reshaped, sin_reshaped,
                                     None, None)
    if q_embed is None:
        q_embed = query_states_reshaped.view(query_states.shape)
    elif q_embed is not query_states:
        q_embed.copy_(query_states_reshaped.view(query_states.shape))

    if k_embed is None:
        k_embed = key_states_reshaped.view(key_states.shape)
    elif k_embed is not key_states:
        k_embed.copy_(key_states_reshaped.view(key_states.shape))

    return q_embed, k_embed
