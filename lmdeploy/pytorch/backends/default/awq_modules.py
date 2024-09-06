# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from torch import distributed as dist

from ..awq_modules import LinearW4A16Builder, LinearW4A16Impl

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    # unpacking columnwise
    iweights = torch.bitwise_right_shift(
        qweight[:, :, None],
        shifts[None, None, :]).to(torch.int8  # smallest dtype available
                                  )
    iweights = iweights.view(iweights.shape[0], -1)

    # unpacking columnwise
    izeros = torch.bitwise_right_shift(
        qzeros[:, :, None],
        shifts[None, None, :]).to(torch.int8  # smallest dtype available
                                  )
    izeros = izeros.view(izeros.shape[0], -1)

    return iweights, izeros


def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(
        izeros.shape[-1],
        dtype=torch.int32,
        device=izeros.device,
    )
    # (-1, 8)
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    # (-1, 2, 4)
    reverse_order_tensor = reverse_order_tensor.unflatten(-1, (2, 4))
    # (-1, 4, 2)
    reverse_order_tensor = reverse_order_tensor.transpose(-1, -2)
    reverse_order_tensor = reverse_order_tensor.flatten()

    izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]

    return iweights, izeros


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    # Reverse the order of the iweight and izeros tensors
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    # fp16 weights
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    iweight = (iweight - izeros) * scales

    return iweight


class DefaultLinearW4A16Impl(LinearW4A16Impl):
    """w4a16 linear implementation."""

    def __init__(self, in_features: int, out_features: int, w_bit: int,
                 group_size: int):
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size

    def forward(self,
                x,
                qweight: torch.Tensor,
                scales: torch.Tensor,
                qzeros: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False):
        """forward."""
        out_shape = x.shape[:-1] + (self.out_features, )
        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()
        out = dequantize_gemm(qweight, qzeros, scales, self.w_bit,
                              self.group_size)
        out = torch.matmul(x, out)

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)
        if all_reduce:
            dist.all_reduce(out)
        return out


class DefaultLinearW4A16Builder(LinearW4A16Builder):
    """w4a16 linear implementation builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              w_bit: int,
              group_size: int,
              bias: bool = False,
              dtype: torch.dtype = None):
        """build."""
        return DefaultLinearW4A16Impl(in_features, out_features, w_bit,
                                      group_size)
