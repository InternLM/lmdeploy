# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import blocked_gemm_fp8, deep_gemm_fp8, quant_fp8, quant_fp8_tma
from lmdeploy.utils import get_logger

from ..blockedf8_modules import LinearBlockedF8Builder, LinearBlockedF8Impl

logger = get_logger('lmdeploy')


def _reduce_scatter_input(out: torch.Tensor, rank: int, tp_sizes: List[int]):
    """reduce scatter."""
    outs = out.split(tp_sizes, -2)
    out = outs[rank]
    outs = list(outs)
    dist.reduce_scatter(out, outs)
    return out


class TritonLinearBlockedF8Impl(LinearBlockedF8Impl):
    """triton linear blocked f8 implementation."""

    def __init__(self, in_features: int, out_features: int, block_size: int, out_dtype: torch.dtype = torch.float16):
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.block_size = block_size

    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                rank: int = 0,
                scatter_size: List[int] = None):
        """forward."""
        x_shape = x.shape
        x = x.flatten(0, -2)
        input_quant, input_scale = quant_fp8(x, self.block_size, dtype=weight.dtype)

        out = blocked_gemm_fp8(input_quant, input_scale, weight.t(), scale.t(), out_dtype=x.dtype)
        if bias is not None:
            out += bias

        out = out.unflatten(0, x_shape[:-1])

        if all_reduce:
            if scatter_size is not None:
                out = _reduce_scatter_input(out, rank, scatter_size)
            else:
                dist.all_reduce(out)
        return out


class TritonLinearBlockedF8Builder(LinearBlockedF8Builder):
    """triton linear blocked f8 implementation builder."""

    @staticmethod
    def build(in_features: int, out_features: int, block_size: int = 128, bias: bool = True, dtype: torch.dtype = None):
        """build."""
        try:
            import deep_gemm  # noqa
            logger.debug('build with DeepGemmLinearBlockedF8Impl')
            return DeepGemmLinearBlockedF8Impl(in_features, out_features, block_size, dtype)
        except:  # noqa
            return TritonLinearBlockedF8Impl(in_features, out_features, block_size, dtype)


class DeepGemmLinearBlockedF8Impl(LinearBlockedF8Impl):
    """Deep gemm blocked f8 implementation."""

    def __init__(self, in_features: int, out_features: int, block_size: int, out_dtype: torch.dtype = torch.float16):
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.block_size = block_size

    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False):
        """forward."""
        x_shape = x.shape
        x = x.flatten(0, -2)
        input_quant, input_scale = quant_fp8_tma(x, self.block_size, dtype=weight.dtype)

        out = deep_gemm_fp8(input_quant, input_scale, weight, scale, out_dtype=x.dtype)
        if bias is not None:
            out += bias

        if all_reduce:
            dist.all_reduce(out)

        out = out.unflatten(0, x_shape[:-1])
        return out
