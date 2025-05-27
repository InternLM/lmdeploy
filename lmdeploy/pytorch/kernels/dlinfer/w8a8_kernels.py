# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
import torch
from torch import Tensor


def dynamic_quant(x: Tensor, quant_dtype: torch.dtype, quant_granularity: str = 'PER_TOKEN'):
    input_quant, input_scale = ext_ops.dynamic_quant(x, quant_dtype, quant_granularity)
    return input_quant, input_scale


def linear_w8a8(
    a: Tensor,
    b: Tensor,
    rms_scale: float,
    linear_scale: float,
    out_dtype: torch.dtype,
    quant_dtype: torch.dtype,
    bias=None,
):
    """This function performs matrix multiplication with dynamic quantization.

    It takes two input tensors `a` and `b`, scales them with `rms_scale` and `linear_scale`, and optionally adds a
    `bias`. The output is returned in the specified `output_dtype`.
    """
    return ext_ops.linear_w8a8(a, b, rms_scale, linear_scale, out_dtype, quant_dtype, bias)


def rms_norm_w8a8(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype = torch.int8,
    residual: Tensor = None,
):
    """Rms norm kernel."""
    if residual is None:
        return ext_ops.rms_norm_w8a8(hidden_states, weight, epsilon, quant_dtype)
    else:
        return ext_ops.add_rms_norm_w8a8(hidden_states, residual, weight, epsilon, quant_dtype)
