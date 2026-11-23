# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.utils import get_logger

from ..norm import LayerNormImpl, RMSNormGatedImpl, RMSNormImpl


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


class DefaultRMSNormGatedImpl(RMSNormGatedImpl):
    """FLA adapter shared with backends that previously used the FLA module.

    Keep vendor imports lazy and retain the existing autotuning workaround. The adapter owns no parameters; FLA receives
    the public layer's weight.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = 'silu'):
        self.hidden_size = hidden_size
        self.eps = eps
        self.activation = activation
        self._fla_forward = None

    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor | None,
                z: torch.Tensor,
                residual: torch.Tensor | None = None,
                prenorm: bool = False,
                residual_in_fp32: bool = False):
        """Call the same functional entry point as FusedRMSNormGated."""
        if self._fla_forward is None:
            from fla.modules.fused_norm_gate import rms_norm_gated
            try:
                # Preserve the old factory's avoidance of NB specialization.
                from fla.modules.fused_norm_gate import layer_norm_gated_fwd_kernel
                keys = layer_norm_gated_fwd_kernel.fn.keys
                if 'NB' in keys:
                    keys.remove('NB')
            except Exception:
                get_logger('lmdeploy').debug('patch layer_norm_gated_fwd_kernel autotuning failed.')
            self._fla_forward = rms_norm_gated
        return self._fla_forward(x,
                                 z,
                                 weight,
                                 None,
                                 self.activation,
                                 residual=residual,
                                 eps=self.eps,
                                 prenorm=prenorm,
                                 residual_in_fp32=residual_in_fp32)
