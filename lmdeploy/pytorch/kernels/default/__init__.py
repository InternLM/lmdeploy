# Copyright (c) OpenMMLab. All rights reserved.
from .multinomial_sampling import multinomial_sampling
from .w8a8_kernels import per_channel_quant

__all__ = [
    'multinomial_sampling',
    'per_channel_quant',
]
