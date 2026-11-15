# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.kernels.ascend.rms_norm import rms_norm

from ..norm import RMSNormGatedImpl, RMSNormImpl


class AscendRMSNormImpl(RMSNormImpl):
    """Ascend RMS norm implementation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        if residual is None:
            x = rms_norm(x, weight, self.eps)
            return x
        else:
            x, residual = rms_norm(x, weight, self.eps, residual=residual)
            return x, residual


class AscendRMSNormGatedImpl(RMSNormGatedImpl):
    """Ascend gated RMS normalization without owning model parameters."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = 'silu'):
        self.hidden_size = hidden_size
        self.eps = eps
        self.activation = activation

    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor | None,
                z: torch.Tensor,
                residual: torch.Tensor | None = None,
                prenorm: bool = False,
                residual_in_fp32: bool = False):
        """Keep the existing fused kernel for the ordinary no-residual path."""
        from lmdeploy.pytorch.kernels.ascend.triton_ops.rms_norm_gated import rmsnorm_fn

        norm_input = x if residual is None else x.float() + residual.float()
        output = rmsnorm_fn(norm_input,
                            weight,
                            None,
                            z=z,
                            eps=self.eps,
                            norm_before_gate=True,
                            activation=self.activation)
        if residual is not None:
            output = output.to(x.dtype)
        if not prenorm:
            return output
        residual_dtype = residual.dtype if residual is not None else (torch.float32 if residual_in_fp32 else x.dtype)
        return output, norm_input.to(residual_dtype)
