# Copyright (c) Shanghai AI Laboratory. All rights reserved.

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch import envs as _envs
from lmdeploy.utils import get_logger

from ..awq_modules import LinearW4A16BuildSpec, LinearW4A16Impl
from .tm_probe import turbomind_support_reason

logger = get_logger('lmdeploy')


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
    """Wq gemm forward."""
    from lmdeploy.pytorch.kernels.cuda.awq_kernels import awq_linear
    out_shape = x.shape[:-1] + (out_features, )
    input_dtype = x.dtype
    if input_dtype != torch.float16:
        x = x.half()

    x = x.flatten(0, -2)
    out = awq_linear(x, qweight, scales, qzeros)

    out = out + bias if bias is not None else out
    out = out.reshape(out_shape)

    # always want 3D tensor if tensor is 2D
    if len(out.shape) == 2:
        out = out.unsqueeze(0)

    if input_dtype != torch.float16:
        out = out.to(dtype=input_dtype)
    return out


class AwqLinearW4A16Impl(LinearW4A16Impl):
    """Awq kernel linear."""

    def __init__(self, in_features: int, out_features: int, w_bit: int, group_size: int):
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size

    def forward(self,
                x,
                qweight: torch.Tensor,
                scales: torch.Tensor,
                qzeros: torch.Tensor,
                bias: torch.Tensor | None = None,
                all_reduce: bool = False,
                group: torch.distributed.ProcessGroup | None = None):
        """forward."""
        out_features = scales.size(1)
        out = wq_gemm_forward(x, qweight, qzeros, scales, self.w_bit, self.group_size, bias, out_features)
        if all_reduce:
            dist.all_reduce(out, group=group)
        return out


def _build_linear_w4a16(spec: LinearW4A16BuildSpec) -> LinearW4A16Impl:
    """Build the selected CUDA W4A16 linear implementation."""
    state = _envs.tm_kernel_state('w4a16')

    if state in ('auto', 'on'):
        reason = turbomind_support_reason(spec.in_features, spec.out_features,
                                          spec.w_bit, spec.group_size, spec.output_dtype)
        if reason is None:
            from .tm_gemm_modules import TurbomindGemmW4A16Impl
            logger.debug('Build LinearW4A16 with TurbomindGemmW4A16Impl.')
            return TurbomindGemmW4A16Impl(spec.in_features, spec.out_features, spec.group_size)
        elif state == 'on':
            from lmdeploy import turbomind

            raise RuntimeError(
                f'TurboMind W4A16 linear was requested but is unavailable or incompatible: {reason}.'
            ) from turbomind._import_error

    logger.debug('Build LinearW4A16 with AwqLinearW4A16Impl.')
    return AwqLinearW4A16Impl(spec.in_features, spec.out_features, spec.w_bit, spec.group_size)
