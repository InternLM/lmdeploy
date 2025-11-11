# Copyright (c) OpenMMLab. All rights reserved.
import functools

from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func
from flash_attn_interface import flash_attn_with_kvcache as _flash_attn_with_kvcache


@functools.wraps(_flash_attn_varlen_func)
def flash_attn_varlen_func(*args, **kwargs):
    output = _flash_attn_varlen_func(*args, **kwargs)
    if isinstance(output, tuple):
        # for old api
        return output[0]
    return output


@functools.wraps(_flash_attn_with_kvcache)
def flash_attn_with_kvcache(*args, **kwargs):
    output = _flash_attn_with_kvcache(*args, **kwargs)
    return output
