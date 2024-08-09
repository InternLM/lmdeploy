# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.distributed as dist
from torch import nn

from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import (
    matmul_kernel_dynamic_quant, per_token_quant_int8, rms_norm_dynamic_quant)
from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.models.q_modules import QTensor

from ..qmodules import (LinearW8A8Builder, LinearW8A8Impl, RMSNormW8A8Builder,
                        RMSNormW8A8Impl)


class TritonRMSNormW8A8Impl(RMSNormW8A8Impl, nn.Module):
    """triton RMS norm w8a8 implementation api."""

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        if residual is not None:
            x = x + residual
            residual = x
        hidden_states_quant, rms_scale = rms_norm_dynamic_quant(
            x, self.weight, self.eps)
        x = QTensor(hidden_states_quant, rms_scale)
        if residual is None:
            return x
        return x, residual


class TritonRMSNormBuilder(RMSNormW8A8Builder):
    """triton RMS norm w8a8 implementation builder."""

    @staticmethod
    def build(weight: torch.Tensor, eps: float = 1e-6):
        """build."""
        return TritonRMSNormW8A8Impl(weight, eps)


class TritonLinearW8A8Impl(LinearW8A8Impl, nn.Module):
    """triton linear w8a8 implementation."""

    def __init__(self, mod: nn.Module):
        super().__init__()
        self.weight = mod.weight
        self.scale = mod.scale
        self.bias = mod.bias

    def forward(self, x, all_reduce: bool = False):
        """forward."""
        if isinstance(x, torch.Tensor):
            x = x.contiguous()
            input_quant, input_scale = per_token_quant_int8(x, 1e-7)
        else:
            assert isinstance(x, QTensor)
            input_quant, input_scale = x.tensor, x.scale

        out = matmul_kernel_dynamic_quant(input_quant,
                                          self.weight,
                                          input_scale,
                                          self.scale,
                                          output_dtype=torch.float16,
                                          bias=self.bias)

        if all_reduce:
            dist.all_reduce(out)
        return out


class TritonLinearW8A8Builder(LinearW8A8Builder):
    """triton linear w8a8 implementation builder."""

    @staticmethod
    def build(mod: nn.Module, ctx_mgr: StepContextManager = None):
        """build."""
        return TritonLinearW8A8Impl(mod)
