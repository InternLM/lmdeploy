# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import distributed as dist
from torch import nn

from lmdeploy.pytorch.model_inputs import StepContextManager

from ..awq_modules import LinearW4A16Builder, LinearW4A16Impl


class DefaultLinearW4A16Impl(LinearW4A16Impl):
    """w4a16 linear implementation."""

    def __init__(self, mod: nn.Module):
        super().__init__()
        self.qweight = mod.qweight
        self.qzeros = mod.qzeros
        self.scales = mod.scales
        self.w_bit = mod.w_bit
        self.group_size = mod.group_size
        self.bias = mod.bias
        self.in_features = mod.in_features
        self.out_features = mod.out_features

    def forward(self, x, all_reduce: bool = False):
        """forward."""
        from awq.utils.packing_utils import dequantize_gemm
        out_shape = x.shape[:-1] + (self.out_features, )
        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        out = dequantize_gemm(self.qweight, self.qzeros, self.scales,
                              self.w_bit, self.group_size)
        out = torch.matmul(x, out)

        out = out + self.bias if self.bias is not None else out
        out = out.reshape(out_shape)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)
        if all_reduce:
            dist.all_reduce(out)
        return out


class DefaultLinearW4A16Builder(LinearW4A16Builder):
    """w4a16 linear implementation builder."""

    @staticmethod
    def build(mod: nn.Module, ctx_mgr: StepContextManager = None):
        """build."""
        return DefaultLinearW4A16Impl(mod)
