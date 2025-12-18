# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch


def _aligned_size(a, b):
    return (a + b - 1) // b * b


def fast_log2_ceil_torch(x: torch.Tensor) -> torch.Tensor:
    bits_x = x.view(torch.int32)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    result = (exp_x - 127).to(torch.int32)
    result = result + torch.where(man_bits != 0, 1, 0)

    return result.to(torch.int32)


def fast_pow2_torch(x: torch.Tensor) -> torch.Tensor:
    bits_x = (x + 127) << 23
    return bits_x.view(torch.float32)


def fast_round_scale_torch(amax: torch.Tensor, fp8_max: torch.Tensor) -> torch.Tensor:
    return fast_pow2_torch(fast_log2_ceil_torch(amax / fp8_max))


def _get_quant_scaling(weight: torch.Tensor,
                       fp8_dtype: torch.dtype,
                       dim: Union[int, Sequence[int]],
                       scale_fmt: Optional[str] = None):
    """Get the scaling factor for FP8 quantization."""
    finfo = torch.finfo(fp8_dtype)
    fmax = finfo.max
    amax = weight.abs().amax(dim, keepdim=True).clamp_min(1e-6).float()

    if scale_fmt == 'ue8m0':
        return fast_round_scale_torch(amax, fmax)
    else:
        # default
        scaling = amax / fmax
    return scaling


def quant_blocked_fp8(weight: torch.Tensor,
                      fp8_dtype: torch.dtype,
                      block_size: int = 128,
                      scale_fmt: Optional[str] = None):
    """Quantize the weight tensor to blocked FP8 format."""
    assert scale_fmt in (None, 'ue8m0'), f'Unsupported scale_fmt: {scale_fmt}'

    weight_shape = weight.shape
    K, N = weight_shape[-2:]
    aligned_k = _aligned_size(K, block_size)
    aligned_n = _aligned_size(N, block_size)

    # fill the weight tensor with zeros if it is not aligned
    if aligned_k != K or aligned_n != N:
        new_weight = weight.new_zeros(weight_shape[:-2] + (aligned_k, aligned_n))
        new_weight[..., :K, :N] = weight
        weight = new_weight
    aligned_shape = weight.shape

    # reverse pixel shuffle
    weight = weight.unflatten(-2, (-1, block_size)).unflatten(-1, (-1, block_size))
    weight = weight.to(torch.float32)

    # get scaling
    scaling = _get_quant_scaling(weight, fp8_dtype, dim=(-3, -1), scale_fmt=scale_fmt)

    # get quantized weight
    quantized_weight = weight / scaling
    quantized_weight = quantized_weight.to(fp8_dtype)
    quantized_weight = quantized_weight.view(aligned_shape)
    quantized_weight = quantized_weight[..., :K, :N]

    # reshape scaling
    scaling = scaling.squeeze(-3, -1)

    return quantized_weight, scaling
