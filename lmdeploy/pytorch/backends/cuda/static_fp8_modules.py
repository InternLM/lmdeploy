# Copyright (c) OpenMMLab. All rights reserved.

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import (
    matmul_kernel_static_quant,
    per_tensor_quant_fp8,
)

from ..static_fp8_modules import (
    LinearStaticF8Builder,
    LinearStaticF8Impl,
)


@torch.compile(
    backend='inductor',
    fullgraph=True,
    dynamic=True,
    mode='default',
)
def _per_tensor_quant_fp8_e4m3fn_inductor(
    x: torch.Tensor,
    scale: torch.Tensor,
):
    """Fuse static E4M3 quantization for fixed token counts."""
    dtype_info = torch.finfo(torch.float8_e4m3fn)
    return torch.clamp(
        x.float() / scale.float(),
        min=dtype_info.min,
        max=dtype_info.max,
    ).to(torch.float8_e4m3fn)


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
        self.use_scaled_mm = _envs.static_fp8_use_scaled_mm
        self.use_compiled_quant = (
            _envs.static_fp8_use_compiled_quant
        )
        self.compiled_quant_token_counts = set(
            _envs.static_fp8_compiled_quant_token_counts
        )

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

        if self.use_scaled_mm:
            num_tokens = x.numel() // x.shape[-1]
            use_compiled_quant = (
                self.use_compiled_quant
                and num_tokens in self.compiled_quant_token_counts
                and weight.dtype == torch.float8_e4m3fn
            )
            if use_compiled_quant:
                input_quant = (
                    _per_tensor_quant_fp8_e4m3fn_inductor(
                        x,
                        input_scale,
                    )
                )
            else:
                input_quant = per_tensor_quant_fp8(
                    x,
                    input_scale,
                    quant_dtype=weight.dtype,
                )

            in_features = input_quant.shape[-1]
            out_features = weight.shape[0]

            input_quant_2d = input_quant.reshape(
                num_tokens,
                in_features,
            )
            if input_scale.numel() == 1 and weight_scale.numel() == 1:
                input_scale_mm = input_scale.float()
                weight_scale_mm = weight_scale.float()
            else:
                input_scale_mm = (
                    input_scale.float()
                    .reshape(1, 1)
                    .expand(num_tokens, 1)
                    .contiguous()
                )
                assert weight_scale.numel() == out_features
                weight_scale_mm = (
                    weight_scale.float()
                    .reshape(1, out_features)
                    .contiguous()
                )

            output = torch._scaled_mm(
                input_quant_2d,
                weight.t(),
                input_scale_mm,
                weight_scale_mm,
                out_dtype=output_dtype,
                use_fast_accum=True,
            )
            output = output.reshape(
                *x.shape[:-1],
                out_features,
            )

            if bias is not None:
                output = output + bias
        else:
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
