# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.kernels.cuda import rms_norm

from ..norm import RMSNormBuilder, RMSNormImpl


class TritonRMSNormImpl(RMSNormImpl, nn.Module):
    """triton RMS norm implementation."""

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        if residual is None:
            x = rms_norm(x, self.weight, self.eps)
            return x
        else:
            x, residual = rms_norm(x, self.weight, self.eps, residual=residual)
            return x, residual


class TritonRMSNormBuilder(RMSNormBuilder):
    """triton RMS norm implementation builder."""

    @staticmethod
    def build(weight: torch.Tensor, eps: float = 1e-6):
        """build."""
        return TritonRMSNormImpl(weight, eps)
