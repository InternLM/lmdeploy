# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.kernels.cuda import rms_norm

from ..norm import RMSNormBuilder, RMSNormImpl


class TritonRMSNormImpl(RMSNormImpl, nn.Module):

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        if residual is not None:
            x = x + residual
            residual = x
        x = rms_norm(x, self.weight, self.eps)
        if residual is None:
            return x
        return x, residual


class TritonRMSNormBuilder(RMSNormBuilder):

    @staticmethod
    def build(weight: torch.Tensor, eps: float = 1e-6):
        return TritonRMSNormImpl(weight, eps)
