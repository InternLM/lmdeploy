# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

import dlinfer.ops as ext_ops
from torch import Tensor


def fill_kv_cache(
    key_states: Tensor,
    value_states: Tensor,
    key_caches: Tensor,
    value_caches: Tensor,
    kv_start_indices: Tensor,
    k_scales_zeros: list = [],
    v_scales_zeros: list = [],
    quant_bits: Literal[0, 4, 8] = 0,
):
    """fill key/value state to cache for paged attention."""
    return ext_ops.fill_kv_cache(key_states,
                                 value_states,
                                 key_caches,
                                 value_caches,
                                 kv_start_indices,
                                 k_scales_zeros=k_scales_zeros,
                                 v_scales_zeros=v_scales_zeros,
                                 quant_bits=quant_bits)
