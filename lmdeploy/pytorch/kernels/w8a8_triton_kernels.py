# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

per_channel_quant = FunctionDispatcher('per_channel_quant').make_caller()

matmul_kernel_dynamic_quant = FunctionDispatcher(
    'matmul_kernel_dynamic_quant').make_caller()

per_token_quant_int8 = FunctionDispatcher('per_token_quant_int8').make_caller()

rms_norm_dynamic_quant = FunctionDispatcher(
    'rms_norm_dynamic_quant').make_caller()
