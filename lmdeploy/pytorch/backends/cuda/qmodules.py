# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import (matmul_kernel_dynamic_quant, per_token_quant_int8,
                                                               rms_norm_dynamic_quant)
from lmdeploy.pytorch.models.q_modules import QTensor

from ..qmodules import LinearW8A8Builder, LinearW8A8Impl, RMSNormW8A8Builder, RMSNormW8A8Impl


class TritonRMSNormW8A8Impl(RMSNormW8A8Impl):
    """Triton RMS norm w8a8 implementation api."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, quant_dtype: torch.dtype = torch.int8):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.quant_dtype = quant_dtype

    def forward(self, x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        if residual is None:
            (x, rms_scale) = rms_norm_dynamic_quant(x, weight, self.eps, quant_dtype=self.quant_dtype)
            x = QTensor(x, rms_scale)
            return x
        else:
            (x, rms_scale, residual) = rms_norm_dynamic_quant(x,
                                                              weight,
                                                              self.eps,
                                                              residual=residual,
                                                              quant_dtype=self.quant_dtype)
            x = QTensor(x, rms_scale)
            return x, residual


class TritonRMSNormBuilder(RMSNormW8A8Builder):
    """Triton RMS norm w8a8 implementation builder."""

    @staticmethod
    def build(hidden_size: int, eps: float = 1e-6, quant_dtype: torch.dtype = torch.int8):
        """build."""
        return TritonRMSNormW8A8Impl(hidden_size, eps, quant_dtype)


class TritonLinearW8A8Impl(LinearW8A8Impl):
    """Triton linear w8a8 implementation."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 out_dtype: torch.dtype = torch.float16,
                 quant_dtype: torch.dtype = torch.int8):
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.quant_dtype = quant_dtype

    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                group: Optional[torch.distributed.ProcessGroup] = None):
        """forward."""
        if isinstance(x, torch.Tensor):
            input_quant, input_scale = per_token_quant_int8(x, 1e-7, quant_dtype=self.quant_dtype)
        else:
            assert isinstance(x, QTensor)
            input_quant, input_scale = x.tensor, x.scale

        out = matmul_kernel_dynamic_quant(input_quant,
                                          weight,
                                          input_scale,
                                          scale,
                                          output_dtype=self.out_dtype,
                                          bias=bias)

        if all_reduce:
            dist.all_reduce(out, group=group)
        return out


class TritonLinearW8A8Builder(LinearW8A8Builder):
    """Triton linear w8a8 implementation builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              bias: bool = True,
              dtype: torch.dtype = None,
              quant_dtype: torch.dtype = torch.int8):
        """build."""
        return TritonLinearW8A8Impl(in_features, out_features, dtype, quant_dtype=quant_dtype)
