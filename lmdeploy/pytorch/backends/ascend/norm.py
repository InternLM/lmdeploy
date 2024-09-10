# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.kernels.ascend import rms_norm

from ..norm import RMSNormBuilder, RMSNormImpl


class AscendRMSNormImpl(RMSNormImpl, nn.Module):
    """ascend RMS norm implementation."""

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


class AscendRMSNormBuilder(RMSNormBuilder):
    """ascend RMS norm implementation builder."""

    @staticmethod
    def build(weight: torch.Tensor, eps: float = 1e-6):
        """build."""
        return AscendRMSNormImpl(weight, eps)
