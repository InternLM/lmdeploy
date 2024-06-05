# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from .dispatcher import FunctionDispatcher


def _fill_kv_cache_api(k_states: Tensor, v_states: Tensor, k_caches: Tensor,
                       v_caches: Tensor, q_start_loc: Tensor,
                       q_seq_length: Tensor, kv_seq_length: Tensor,
                       max_q_seq_length: int, block_offsets: Tensor):
    """fill key/value state to cache for paged attention."""
    ...


fill_kv_cache = FunctionDispatcher('fill_kv_cache').make_caller(
    _fill_kv_cache_api)
