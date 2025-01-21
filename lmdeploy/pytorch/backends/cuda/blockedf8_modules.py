# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional

import torch
import torch.distributed as dist

from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import blocked_gemm_fp8, quant_fp8

from ..blockedf8_modules import LinearBlockedF8Builder, LinearBlockedF8Impl


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
                all_reduce: bool = False):
        """forward."""
        x_shape = x.shape
        x = x.flatten(0, -2)
        input_quant, input_scale = quant_fp8(x, self.block_size, dtype=weight.dtype)

        out = blocked_gemm_fp8(input_quant, input_scale, weight.t(), scale.t(), out_dtype=x.dtype)
        if bias is not None:
            out += bias

        if all_reduce:
            dist.all_reduce(out)

        out = out.unflatten(0, x_shape[:-1])
        return out


class TritonLinearBlockedF8Builder(LinearBlockedF8Builder):
    """triton linear blocked f8 implementation builder."""

    @staticmethod
    def build(in_features: int, out_features: int, block_size: int = 128, bias: bool = True, dtype: torch.dtype = None):
        """build."""
        return TritonLinearBlockedF8Impl(in_features, out_features, block_size, dtype)
