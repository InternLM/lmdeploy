# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..kernels.w8a8_triton_kernels import (layer_norm_dynamic_quant,
                                           matmul_kernel_dynamic_quant,
                                           per_channel_quant,
                                           per_token_quant_int8,
                                           rms_norm_dynamic_quant)


@dataclass
class QTensor:
    """A data class representing a Quantized Tensor.

    This class wraps around a regular Pytorch tensor and adds quantization-
    specific parameters.
    """
    tensor: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor = None

    def to(self, *args, **kwargs):
        """Move the tensor to target dtype or device."""
        tensor = self.tensor.to(*args, **kwargs)
        scale = self.scale.to(*args, **kwargs)
        zero_point = None
        if self.zero_point is not None:
            zero_point = self.zero_point.to(*args, **kwargs)
        return QTensor(tensor, scale, zero_point)

    def __getattr__(self, name: str):
        """Allows attribute access to be forwarded to the wrapped tensor when
        the attribute doesn't exist in QTensor."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.tensor, name)


class QRMSNorm(nn.Module):
    """It performs traditional RMS normalization and then quantizes the output
    to 8-bit integers."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @classmethod
    def from_float(cls, mod: nn.Module, initialization: bool = True):
        """Class method to create a QRMSNorm instance from a floating-point
        module.

        `initialization = True` for real init.
        `initialization = False` for dummy init.
        """
        hidden_size = mod.weight.shape[0]
        eps = getattr(mod, 'variance_epsilon', getattr(mod, 'eps', None))
        assert eps is not None
        q_mod = cls(hidden_size, eps)
        if initialization:
            q_mod.weight = nn.Parameter(mod.weight.detach())
        return q_mod

    def forward(self, hidden_states):
        """Defines the computation performed at every call.

        Performs RMS normalization followed by dynamic quantization on
        hidden_states. Returns a QTensor which wraps the quantized tensor along
        with its scale factor.
        """
        hidden_states_quant, rms_scale = rms_norm_dynamic_quant(
            hidden_states, self.weight, self.variance_epsilon)
        return QTensor(hidden_states_quant, rms_scale)


class QLayerNorm(nn.Module):
    """It performs traditional RMS normalization and then quantizes the output
    to 8-bit integers."""

    def __init__(self, normalized_shape, bias: bool = True, eps=1e-6):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        self.variance_epsilon = eps
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))

    @classmethod
    def from_float(cls, mod: nn.Module, initialization: bool = True):
        """Class method to create a QRMSNorm instance from a floating-point
        module.

        `initialization = True` for real init.
        `initialization = False` for dummy init.
        """
        normalized_shape = mod.normalized_shape
        eps = mod.eps
        bias = mod.bias is not None
        q_mod = cls(normalized_shape, bias, eps)
        if initialization:
            q_mod.weight = nn.Parameter(mod.weight.detach())
            if bias:
                q_mod.bias = nn.Parameter(mod.bias.detach())
        return q_mod

    def forward(self, hidden_states):
        out, scale = layer_norm_dynamic_quant(hidden_states, self.weight,
                                              self.bias, self.variance_epsilon)
        return QTensor(out, scale)


class QLinear(nn.Module):
    """A Linear layer that operates on quantized inputs and weights.

    It performs matrix multiplication in 8-bit precision and dequantize the
    results back to float.
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            'weight',
            torch.empty((out_features, in_features),
                        device=device,
                        dtype=torch.int8))
        self.register_buffer(
            'scale',
            torch.empty((out_features, 1), device=device, dtype=torch.float32))
        if bias:
            self.register_buffer('bias',
                                 torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_float(cls, mod: nn.Module, initialization: bool = True):
        """Class method to create a QLinear instance from a floating-point
        module.

        `initialization = True` for real init.
        `initialization = False` for dummy init.
        """
        q_mod = cls(mod.in_features,
                    mod.out_features,
                    mod.bias is not None,
                    device=mod.weight.device,
                    dtype=mod.weight.dtype)

        if initialization:
            weight_quant, scale = per_channel_quant(mod.weight.detach(), 8,
                                                    torch.int8)
            q_mod.weight.data = weight_quant
            q_mod.scale = scale

        if mod.bias is not None:
            q_mod.bias.data = mod.bias.detach()
        return q_mod

    def forward(self, input):
        """Defines the computation performed at every call.

        Performs quantization if the input is a tensor, otherwise it assumes
        the input is already quantized (instance of QTensor). Then, it performs
        linear transformation using dynamic quantization method, resulting in
        an 8-bit integer output. Finally, it dequantizes the result back to a
        floating point tensor.
        """

        if isinstance(input, torch.Tensor):
            input_quant, input_scale = per_token_quant_int8(input, 1e-7)
        else:
            assert isinstance(input, QTensor)
            input_quant, input_scale = input.tensor, input.scale

        out = matmul_kernel_dynamic_quant(input_quant,
                                          self.weight,
                                          input_scale,
                                          self.scale,
                                          output_dtype=torch.float16,
                                          bias=self.bias)
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
