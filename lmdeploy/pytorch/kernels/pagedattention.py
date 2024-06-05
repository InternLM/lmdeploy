# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from .dispatcher import FunctionDispatcher


def _paged_attention_fwd_api(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int,
    window_size: int = None,
    sm_scale: float = None,
    shared_kv: bool = False,
):
    """Paged Attention forward.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state caches.
        v (Tensor): Value state caches.
        o (Tensor): Output state.
        block_offsets (Tensor): The block offset of key and value.
        q_start_loc (Tensor): Start token location of each data in batch.
        q_seqlens (Tensor): Query length for each data in batch.
        kv_seqlens (Tensor): Key/Value length for each data in batch.
        max_seqlen (int): The max input length.
        BLOCK (int): The kernel block size.
    """
    ...


paged_attention_fwd = FunctionDispatcher('paged_attention_fwd').make_caller(
    _paged_attention_fwd_api)
