# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.kernels.ascend.rms_norm import rms_norm

from ..default.norm import DefaultRMSNormImpl
from ..norm import RMSNormGatedImpl


class AscendRMSNormImpl(DefaultRMSNormImpl):
    """Ascend RMS norm implementation."""

    def forward(self, x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        # Preserve the FP32 scale when activations use a lower precision.
        if weight.dtype == torch.float32 and x.dtype in (torch.float16, torch.bfloat16):
            return super().forward(x, weight, residual)
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
