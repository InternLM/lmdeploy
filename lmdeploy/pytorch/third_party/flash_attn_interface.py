# Copyright (c) OpenMMLab. All rights reserved.
import functools

from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func


@functools.wraps(_flash_attn_varlen_func)
def flash_attn_varlen_func(*args, **kwargs):
    output = _flash_attn_varlen_func(*args, **kwargs)
    if isinstance(output, tuple):
        # for old api
        return output[0]
    return output
