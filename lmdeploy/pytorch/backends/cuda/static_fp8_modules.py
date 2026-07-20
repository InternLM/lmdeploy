# Copyright (c) OpenMMLab. All rights reserved.

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import (
    matmul_kernel_static_quant,
)

from ..static_fp8_modules import (
    LinearStaticF8Builder,
    LinearStaticF8Impl,
)


class TritonLinearStaticF8Impl(LinearStaticF8Impl):
    """Triton static per-tensor FP8 linear implementation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_dtype: torch.dtype = torch.float16,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        all_reduce: bool = False,
        group: dist.ProcessGroup | None = None,
        rank: int = 0,
        scatter_size: list[int] | None = None,
    ):
        """Run static FP8 linear."""
        output_dtype = self.out_dtype or x.dtype

        output = matmul_kernel_static_quant(
            x,
            weight,
            input_scale,
            weight_scale,
            bias=bias,
            output_dtype=output_dtype,
        )

        if all_reduce:
            if scatter_size is not None:
                output = dist.reduce_scatter_by_tp_sizes(
                    output,
                    rank,
                    scatter_size,
                    group=group,
                )
            else:
                dist.all_reduce(output, group=group)

        return output


class TritonLinearStaticF8Builder(LinearStaticF8Builder):
    """Triton static per-tensor FP8 linear builder."""

    @staticmethod
    def build(
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype | None = None,
    ):
        """Build static FP8 linear implementation."""
        return TritonLinearStaticF8Impl(
            in_features,
            out_features,
            out_dtype=dtype,
        )
