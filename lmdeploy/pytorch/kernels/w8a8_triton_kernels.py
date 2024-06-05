# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .dispatcher import FunctionDispatcher


def _per_channel_quant_api(x, n_bits, dtype):
    """Quantize the input tensor 'x' channel-wise using the given number of
    bits.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be a
            2-dimensional tensor.
        n_bits (int): The number of bits to use for quantization.
        dtype (torch.dtype): The data type to which the quantized tensor should
            be converted.

    Returns:
        tuple: A tuple containing two items -- the quantized tensor and
            the scale used for quantization.
    """
    ...


def _matmul_kernel_dynamic_quant_api(a,
                                     b,
                                     rms_scale,
                                     linear_scale,
                                     residual=None,
                                     bias=None,
                                     output_dtype=torch.float16):
    """This function performs matrix multiplication with dynamic quantization.

    It takes two input tensors `a` and `b`, scales them with `rms_scale` and
    `linear_scale`, and optionally adds a `residual` tensor and a `bias`. The
    output is returned in the specified `output_dtype`.
    """
    ...


def _per_token_quant_int8_api(x, eps):
    """Function to perform per-token quantization on an input tensor `x`.

    It converts the tensor values into signed 8-bit integers and returns the
    quantized tensor along with the scaling factor used for quantization.
    """
    ...


def _rms_norm_dynamic_quant_api(x, w, eps):
    """Performs RMS normalization with dynamic quantization.

    The function reshapes the input tensor `x`, creates an empty tensor `y`
    with the same shape as `x`, and calculates RMS normalization on the
    reshaped `x` using a Triton kernel `_rms_norm_fwd_fused_dynamic_symmetric`.
    """
    ...


per_channel_quant = FunctionDispatcher('per_channel_quant').make_caller(
    _per_channel_quant_api)

matmul_kernel_dynamic_quant = FunctionDispatcher(
    'matmul_kernel_dynamic_quant').make_caller(
        _matmul_kernel_dynamic_quant_api, globals=globals())

per_token_quant_int8 = FunctionDispatcher('per_token_quant_int8').make_caller(
    _per_token_quant_int8_api)

rms_norm_dynamic_quant = FunctionDispatcher(
    'rms_norm_dynamic_quant').make_caller(_rms_norm_dynamic_quant_api)
