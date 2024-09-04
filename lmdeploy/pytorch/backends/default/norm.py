# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..norm import LayerNormBuilder, LayerNormImpl, RMSNormBuilder, RMSNormImpl


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
    def build(hidden_size: int, eps: float = 1e-6):
        """build."""
        return DefaultRMSNormImpl(hidden_size, eps)


class DefaultLayerNormImpl(LayerNormImpl):
    """RMS norm implementation api."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor = None,
                bias: torch.Tensor = None,
                residual: torch.Tensor = None):
        """forward."""
        if residual is not None:
            x = x + residual
            residual = x
        x = torch.nn.functional.layer_norm(x,
                                           self.normalized_shape,
                                           weight=weight,
                                           bias=bias,
                                           eps=self.eps)
        if residual is None:
            return x
        return x, residual


class DefaultLayerNormBuilder(LayerNormBuilder):
    """RMS norm implementation builder."""

    @staticmethod
    def build(normalized_shape: int, eps: float = 1e-6):
        """build."""
        return DefaultLayerNormImpl(normalized_shape, eps)
