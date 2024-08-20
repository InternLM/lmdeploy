# Copyright (c) OpenMMLab. All rights reserved.
import torch_npu
from torch import Tensor


def fill_kv_cache(
    key_states: Tensor,
    value_states: Tensor,
    key_caches: Tensor,
    value_caches: Tensor,
    q_start_loc: Tensor,
    q_seq_length: Tensor,
    kv_seq_length: Tensor,
    max_q_seq_length: int,
    block_offsets: Tensor,
    context: None,
):
    """fill key/value state to cache for paged attention."""
    dest_index_copy_kv(key_states, context.kv_start_indices, key_caches)
    dest_index_copy_kv(value_states, context.kv_start_indices, value_caches)


def dest_index_copy_kv(states: Tensor, dest_loc: Tensor, caches: Tensor):
    block_num, block_size, head, dim = caches.size()
    caches_tmp = caches.view(block_num * block_size, head * dim)
    states = states.reshape(states.shape[0], -1)
    dest_loc = dest_loc.view(-1, 1)
    torch_npu.npu_scatter_nd_update_(caches_tmp, dest_loc, states)
