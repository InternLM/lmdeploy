# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.distributed as dist

from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import (
    matmul_kernel_dynamic_quant, per_token_quant_int8, rms_norm_dynamic_quant)
from lmdeploy.pytorch.models.q_modules import QTensor

from ..qmodules import (LinearW8A8Builder, LinearW8A8Impl, RMSNormW8A8Builder,
                        RMSNormW8A8Impl)


class TritonRMSNormW8A8Impl(RMSNormW8A8Impl):
    """triton RMS norm w8a8 implementation api."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor,
                residual: torch.Tensor = None):
        """forward."""
        if residual is not None:
            x = x + residual
            residual = x
        hidden_states_quant, rms_scale = rms_norm_dynamic_quant(
            x, weight, self.eps)
        x = QTensor(hidden_states_quant, rms_scale)
        if residual is None:
            return x
        return x, residual


class TritonRMSNormBuilder(RMSNormW8A8Builder):
    """triton RMS norm w8a8 implementation builder."""

    @staticmethod
    def build(hidden_size: int, eps: float = 1e-6):
        """build."""
        return TritonRMSNormW8A8Impl(hidden_size, eps)


class TritonLinearW8A8Impl(LinearW8A8Impl):
    """triton linear w8a8 implementation."""

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False):
        """forward."""
        if isinstance(x, torch.Tensor):
            x = x.contiguous()
            input_quant, input_scale = per_token_quant_int8(x, 1e-7)
        else:
            assert isinstance(x, QTensor)
            input_quant, input_scale = x.tensor, x.scale

        out = matmul_kernel_dynamic_quant(input_quant,
                                          weight,
                                          input_scale,
                                          scale,
                                          output_dtype=torch.float16,
                                          bias=bias)

        if all_reduce:
            dist.all_reduce(out)
        return out


class TritonLinearW8A8Builder(LinearW8A8Builder):
    """triton linear w8a8 implementation builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              bias: bool = True,
              dtype: torch.dtype = None):
        """build."""
        return TritonLinearW8A8Impl(in_features, out_features)
