# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..norm import LayerNormImpl, RMSNormImpl


class DefaultRMSNormImpl(RMSNormImpl):
    """RMS norm implementation api."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        input_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x
        x = x.to(torch.float32)
        variance = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = weight.to(torch.float32) * x
        x = x.to(input_dtype)
        if residual is None:
            return x
        return x, residual


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
        x = torch.nn.functional.layer_norm(x, self.normalized_shape, weight=weight, bias=bias, eps=self.eps)
        if residual is None:
            return x
        return x, residual
