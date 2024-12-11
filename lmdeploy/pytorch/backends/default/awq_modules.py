# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache
from typing import Optional

import torch
from torch import distributed as dist

from ..awq_modules import LinearW4A16Builder, LinearW4A16Impl


@lru_cache
def get_shifts(bits: int, device: torch.device):
    """get awq shifts."""
    shifts = torch.arange(0, 32, bits, device=device)
    shifts = shifts.view(2, 4).t().flatten()
    return shifts


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = get_shifts(bits, qzeros.device)

    # unpacking columnwise
    iweights = torch.bitwise_right_shift(qweight[:, :, None],
                                         shifts[None, None, :]).to(torch.int8)
    iweights = iweights.view(iweights.shape[0], -1)

    # unpacking columnwise
    izeros = torch.bitwise_right_shift(qzeros[:, :, None],
                                       shifts[None, None, :]).to(torch.int8)
    izeros = izeros.view(izeros.shape[0], -1)

    # overflow checks
    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    return iweights, izeros


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)

    # fp16 weights
    iweight = iweight.unflatten(0, (-1, group_size))
    iweight = (iweight - izeros[:, None]) * scales[:, None]
    iweight = iweight.flatten(0, 1)

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
