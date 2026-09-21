# Copyright (c) OpenMMLab. All rights reserved.
import functools

import torch

from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func
from flash_attn_interface import flash_attn_with_kvcache as _flash_attn_with_kvcache


@functools.wraps(_flash_attn_varlen_func)
def flash_attn_varlen_func(*args, return_lse: bool = False, **kwargs):
    """Normalize FA3 returns with optional natural-log LSE.

    LSE is FP32 [tokens, heads]. Callers requesting LSE provide cu_seqlens_q and cu_seqlens_k as keyword arguments.
    Empty request partitions have -inf LSE.
    """
    if return_lse:
        kwargs['return_attn_probs'] = True
    output = _flash_attn_varlen_func(*args, **kwargs)
    if return_lse:
        from lmdeploy.pytorch.kernels.cuda.dcp import sanitize_dcp_lse

        output, lse = output
        cu_q, cu_k = kwargs['cu_seqlens_q'], kwargs['cu_seqlens_k']
        valid_counts = torch.repeat_interleave(cu_k[1:] - cu_k[:-1], cu_q[1:] - cu_q[:-1],
                                               output_size=output.size(0))
        lse = sanitize_dcp_lse(lse.transpose(0, 1), valid_counts)
        return output, lse
    if isinstance(output, tuple):
        # for old api
        return output[0]
    return output


@functools.wraps(_flash_attn_with_kvcache)
def flash_attn_with_kvcache(*args, **kwargs):
    output = _flash_attn_with_kvcache(*args, **kwargs)
    return output
