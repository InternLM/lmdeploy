# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import dlinfer.ops as ext_ops
import torch
from torch import Tensor


def per_token_quant_int8(x):
    """Function to perform per-token quantization on an input tensor `x`.

    It converts the tensor values into signed 8-bit integers and returns the quantized tensor along with the scaling
    factor used for quantization.
    """
    input_quant, input_scale = ext_ops.per_token_quant_int8(x)
    return input_quant, input_scale


def smooth_quant_matmul(
    a: Tensor,
    a_scale: Optional[torch.Tensor],
    b: Tensor,
    b_scale: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    bias: Tensor = None,
    all_reduce: bool = False,
):
    """This function performs matrix multiplication with dynamic quantization.

    It takes two input tensors `a` and `b`, scales them with `rms_scale` and `linear_scale`, and optionally adds a
    `bias`. The output is returned in the specified `output_dtype`.
    """
    return ext_ops.smooth_quant_matmul(a, a_scale, b, b_scale, out_dtype, bias, all_reduce)
