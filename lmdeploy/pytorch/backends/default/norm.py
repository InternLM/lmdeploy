# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..norm import RMSNormBuilder, RMSNormImpl


class DefaultRMSNormImpl(RMSNormImpl):
    """RMS norm implementation api."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor,
                residual: torch.Tensor = None):
        """forward."""
        input_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = weight * x.to(input_dtype)
        if residual is None:
            return x
        return x, residual


class DefaultRMSNormBuilder(RMSNormBuilder):
    """RMS norm implementation builder."""

    @staticmethod
    def build(hidden_size: int, eps: float = 1e-6, inplace: bool = False):
        """build."""
        return DefaultRMSNormImpl(hidden_size, eps)
