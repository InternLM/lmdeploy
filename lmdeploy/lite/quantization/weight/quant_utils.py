# Copyright (c) OpenMMLab. All rights reserved.
import torch


def _aligned_size(a, b):
    return (a + b - 1) // b * b


def quant_blocked_fp8(weight: torch.Tensor, fp8_dtype: torch.dtype, block_size: int = 128):
    """Quantize the weight tensor to blocked FP8 format."""
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
    finfo = torch.finfo(fp8_dtype)
    fmax = finfo.max
    scaling = weight.abs().amax((-3, -1), keepdim=True) / fmax

    # get quantized weight
    quantized_weight = weight / scaling
    quantized_weight = quantized_weight.to(fp8_dtype)
    quantized_weight = quantized_weight.view(aligned_shape)
    quantized_weight = quantized_weight[..., :K, :N]

    # reshape scaling
    scaling = scaling.squeeze(-3, -1)

    return quantized_weight, scaling
