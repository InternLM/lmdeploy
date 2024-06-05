# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from .dispatcher import FunctionDispatcher


def _apply_rotary_pos_emb_api(q: Tensor,
                              k: Tensor,
                              cos: Tensor,
                              sin: Tensor,
                              position_ids: Tensor = None,
                              position_ids_1d: Tensor = None,
                              q_embed: Tensor = None,
                              k_embed: Tensor = None):
    """Apply rotary positional embedding on query and key.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state.
        cos (Tensor): cosine matrix (seq_len, dim).
        sin (Tensor): sine matrix (seq_len, dim).
        position_ids (Tensor): Position ids of q and k.
        position_ids_1d (Tensor): 1d Position ids.
        q_embed (Tensor): output q, can be same as q
        k_embed (Tensor): output k, can be same as k

    Returns:
        Tuple[Tensor, Tensor]: Embedded query and key.
    """
    ...


apply_rotary_pos_emb = FunctionDispatcher('apply_rotary_pos_emb').make_caller(
    _apply_rotary_pos_emb_api)
