# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from .dispatcher import FunctionDispatcher


def _alibi_paged_attention_fwd_api(q: Tensor,
                                   k: Tensor,
                                   v: Tensor,
                                   o: Tensor,
                                   block_offsets: Tensor,
                                   b_start_loc: Tensor,
                                   b_seq_len: Tensor,
                                   b_kv_seq_len: Tensor,
                                   max_input_len: int,
                                   head_offset: int = 0,
                                   num_heads: int = -1,
                                   alibi_scale: float = 1.0):
    """Paged attention forward with alibi bias.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state caches.
        v (Tensor): Value state caches.
        o (Tensor): Output state.
        block_offsets (Tensor): The block offset of key and value.
        b_start_loc (Tensor): Start token location of each data in batch.
        b_seq_len (Tensor): Query length for each data in batch.
        b_kv_seq_len (Tensor): Key/Value length for each data in batch.
        max_input_len (int): The max input length.
        head_offset (int): The offset of the start head. Head might be
            partitioned when tensor parallel inference.
        num_heads (int): The number of heads. Head might be partitioned when
            tensor parallel inference.
        BLOCK (int): The kernel block size.
    """
    ...


alibi_paged_attention_fwd = FunctionDispatcher(
    'alibi_paged_attention_fwd').make_caller(_alibi_paged_attention_fwd_api)
