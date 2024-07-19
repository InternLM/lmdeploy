# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..norm import RMSNormBuilder, RMSNormImpl


class DefaultRMSNormImpl(RMSNormImpl, nn.Module):

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        input_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x.to(input_dtype)
        if residual is None:
            return x
        return x, residual


class DefaultRMSNormBuilder(RMSNormBuilder):

    @staticmethod
    def build(weight: torch.Tensor, eps: float = 1e-6):
        return DefaultRMSNormImpl(weight, eps)
