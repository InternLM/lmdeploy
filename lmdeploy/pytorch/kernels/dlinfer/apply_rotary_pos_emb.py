# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import dlinfer.ops as ext_ops
from torch import Tensor


def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    q_embed: Optional[Tensor],
    k_embed: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    query_states_embed, key_states_embed = \
        ext_ops.apply_rotary_pos_emb(query_states,
                                     key_states,
                                     cos, sin)
    if q_embed is None:
        q_embed = query_states_embed.view(query_states.shape)
    elif q_embed is not query_states:
        q_embed.copy_(query_states_embed.view(query_states.shape))

    if k_embed is None:
        k_embed = key_states_embed.view(key_states.shape)
    elif k_embed is not key_states:
        k_embed.copy_(key_states_embed.view(key_states.shape))

    return q_embed, k_embed
