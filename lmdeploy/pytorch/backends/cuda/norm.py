# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.kernels.cuda import rms_norm

from ..norm import RMSNormBuilder, RMSNormImpl


class TritonRMSNormImpl(RMSNormImpl):
    """triton RMS norm implementation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor,
                residual: torch.Tensor = None):
        """forward."""
        if residual is None:
            x = rms_norm(x, weight, self.eps)
            return x
        else:
            x, residual = rms_norm(x, weight, self.eps, residual=residual)
            return x, residual


class TritonRMSNormBuilder(RMSNormBuilder):
    """triton RMS norm implementation builder."""

    @staticmethod
    def build(weight: torch.Tensor, eps: float = 1e-6):
        """build."""
        return TritonRMSNormImpl(weight, eps)
