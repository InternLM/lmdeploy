# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from torch import distributed as dist

from ..awq_modules import LinearW4A16Builder, LinearW4A16Impl


def wq_gemm_forward(
    x,
    qweight,
    qzeros,
    scales,
    w_bit=4,
    group_size=128,
    bias=None,
    out_features=0,
):
    """wq gemm forward."""
    from awq.modules.linear.gemm import awq_ext

    from lmdeploy.pytorch.kernels.cuda.awq_kernels import awq_linear
    out_shape = x.shape[:-1] + (out_features, )
    input_dtype = x.dtype
    if input_dtype != torch.float16:
        x = x.half()

    FP16_MATMUL_HEURISTIC_CONDITION = x.size(0) * x.size(1) >= 64

    x = x.flatten(0, -2)
    if FP16_MATMUL_HEURISTIC_CONDITION:
        out = awq_linear(x, qweight, scales, qzeros)
    else:
        if not x.is_contiguous():
            x = x.contiguous()
        out = awq_ext.gemm_forward_cuda(x, qweight, scales, qzeros, 8)

    out = out + bias if bias is not None else out
    out = out.reshape(out_shape)

    # always want 3D tensor if tensor is 2D
    if len(out.shape) == 2:
        out = out.unsqueeze(0)

    if input_dtype != torch.float16:
        out = out.to(dtype=input_dtype)
    return out


class AwqLinearW4A16Impl(LinearW4A16Impl):
    """awq kernel linear."""

    def __init__(self, in_features: int, out_features: int, w_bit: int,
                 group_size: int):
        from awq.modules.linear.gemm import AWQ_INSTALLED
        assert AWQ_INSTALLED
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size

    def forward(self,
                x,
                qweight: torch.Tensor,
                scales: torch.Tensor,
                qzeros: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False):
        """forward."""
        out_features = scales.size(1)
        out = wq_gemm_forward(x, qweight, qzeros, scales, self.w_bit,
                              self.group_size, bias, out_features)
        if all_reduce:
            dist.all_reduce(out)
        return out


class AwqLinearW4A16Builder(LinearW4A16Builder):
    """awq linear builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              w_bit: int,
              group_size: int,
              bias: bool = False,
              dtype: torch.dtype = None):
        """build."""
        return AwqLinearW4A16Impl(in_features, out_features, w_bit, group_size)
