# Copyright (c) OpenMMLab. All rights reserved.
import torch


def per_channel_quant(x: torch.Tensor, dtype: torch.dtype):
    """Quantize the input tensor 'x' channel-wise using the given number of
    bits.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be a
            2-dimensional tensor.
        dtype (torch.dtype): The data type to which the quantized tensor should
            be converted.

    Returns:
        tuple: A tuple containing two items -- the quantized tensor and
            the scale used for quantization.
    """
    assert x.ndim == 2
    x = x.to(torch.float32)
    x_absmax = x.view(x.shape[0], -1).abs().max(dim=1, keepdim=True)[0]
    qtype_info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    q_max = qtype_info.max
    q_min = qtype_info.min
    scale = x_absmax / q_max
    x_q = x / scale
    if not dtype.is_floating_point:
        x_q = torch.round(x_q)
    x_q = x_q.clamp(q_min, q_max).to(dtype)
    return x_q, scale
