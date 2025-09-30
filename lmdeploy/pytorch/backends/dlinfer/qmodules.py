# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional

import torch
import torch.distributed as dist

from lmdeploy.pytorch.kernels.dlinfer.w8a8_kernels import dynamic_quant, linear_w8a8, rms_norm_w8a8
from lmdeploy.pytorch.models.q_modules import QTensor

from ..qmodules import LinearW8A8Builder, LinearW8A8Impl, RMSNormW8A8Builder, RMSNormW8A8Impl


class DlinferLinearW8A8Impl(LinearW8A8Impl):
    """Dlinfer linear w8a8 implementation."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 out_dtype: torch.dtype = torch.float16,
                 quant_dtype: torch.dtype = torch.int8):
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.quant_dtype = quant_dtype

    def update_weights(self, weight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Update weights."""
        if os.getenv('DLINFER_LINEAR_USE_NN_LAYOUT', '0') == '1':
            weight = weight.data.t().contiguous()
            scale = scale.data.t().contiguous()
        return weight, scale, bias

    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                group: Optional[torch.distributed.ProcessGroup] = None):
        """forward."""
        if isinstance(x, torch.Tensor):
            input_quant, input_scale = dynamic_quant(x, self.quant_dtype)
        else:
            assert isinstance(x, QTensor)
            input_quant, input_scale = x.tensor, x.scale

        out = linear_w8a8(input_quant, weight, input_scale, scale, self.out_dtype, self.quant_dtype, bias)
        if all_reduce:
            dist.all_reduce(out, group=group)
        return out


class DlinferLinearW8A8Builder(LinearW8A8Builder):
    """Dlinfer linear w8a8 implementation builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              bias: bool = True,
              dtype: torch.dtype = None,
              quant_dtype: torch.dtype = torch.int8):
        """build."""
        return DlinferLinearW8A8Impl(in_features, out_features, dtype, quant_dtype)


class DlinferRMSNormW8A8Impl(RMSNormW8A8Impl):
    """Dlinfer RMS norm w8a8 implementation api."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, quant_dtype: torch.dtype = torch.int8):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.quant_dtype = quant_dtype

    def forward(self, x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        if residual is None:
            (x, rms_scale) = rms_norm_w8a8(x, weight, self.eps, self.quant_dtype)
            x = QTensor(x, rms_scale)
            return x
        else:
            (x, rms_scale, residual) = rms_norm_w8a8(x, weight, self.eps, self.quant_dtype, residual)
            x = QTensor(x, rms_scale)
            return x, residual


class DlinferRMSNormW8A8Builder(RMSNormW8A8Builder):
    """Dlinfer RMS norm w8a8 implementation builder."""

    @staticmethod
    def build(hidden_size: int, eps: float = 1e-6, quant_dtype: torch.dtype = torch.int8):
        """build."""
        return DlinferRMSNormW8A8Impl(hidden_size, eps, quant_dtype)
