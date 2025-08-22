# Copyright (c) OpenMMLab. All rights reserved.

from .w8a8_triton_kernels import (matmul_kernel_dynamic_quant, per_channel_quant, per_token_quant_int8,
                                  rms_norm_dynamic_quant)

__all__ = [
    'matmul_kernel_dynamic_quant',
    'per_channel_quant',
    'per_token_quant_int8',
    'rms_norm_dynamic_quant',
]
