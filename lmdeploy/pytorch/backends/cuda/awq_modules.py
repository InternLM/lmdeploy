# Copyright (c) OpenMMLab. All rights reserved.

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch import envs as _envs
from lmdeploy.utils import get_logger

from ..awq_modules import LinearW4A16Builder, LinearW4A16Impl

logger = get_logger('lmdeploy')


def _is_turbomind_gemm_capability_supported(capability: tuple[int, int]):
    """Limit the prototype to the SM90 architecture validated end to end."""
    return capability == (9, 0)


def _turbomind_support_reason(in_features: int,
                              out_features: int,
                              w_bit: int,
                              group_size: int,
                              dtype: torch.dtype = None) -> str | None:
    """Return why the bundled TurboMind provider cannot serve this layer."""
    if w_bit != 4:
        return f'w_bit must be 4, but got {w_bit}'
    if group_size != 128:
        return f'group_size must be 128, but got {group_size}'
    if in_features % 128 != 0:
        return f'K must be divisible by 128, but got {in_features}'
    if out_features % 32 != 0:
        return f'N must be divisible by 32, but got {out_features}'
    if dtype not in (None, torch.float16):
        return f'dtype must be float16, but got {dtype}'
    if not torch.cuda.is_available():
        return 'CUDA is unavailable'

    capability = torch.cuda.get_device_capability()
    if not _is_turbomind_gemm_capability_supported(capability):
        return f'CUDA capability {capability} is unsupported'

    from .turbomind_awq_modules import _load_turbomind
    try:
        _load_turbomind()
    except RuntimeError as error:
        return str(error)
    return None


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


class AwqLinearW4A16Builder(LinearW4A16Builder):
    """Awq linear builder."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              w_bit: int,
              group_size: int,
              bias: bool = False,
              dtype: torch.dtype = None):
        """build."""
        provider = _envs.w4a16_gemm_backend

        if provider == 'auto':
            reason = _turbomind_support_reason(in_features, out_features,
                                               w_bit, group_size, dtype)
            if reason is None:
                from .turbomind_awq_modules import TurbomindAwqLinearW4A16Impl
                impl_cls = TurbomindAwqLinearW4A16Impl
            else:
                impl_cls = AwqLinearW4A16Impl
        elif provider == 'turbomind':
            reason = _turbomind_support_reason(in_features, out_features,
                                               w_bit, group_size, dtype)
            if reason is not None:
                raise RuntimeError(
                    'TurboMind W4A16 linear was requested but is unavailable '
                    f'or incompatible: {reason}.')
            from .turbomind_awq_modules import TurbomindAwqLinearW4A16Impl
            impl_cls = TurbomindAwqLinearW4A16Impl
        else:
            impl_cls = AwqLinearW4A16Impl

        logger.debug('Build LinearW4A16 with %s.', impl_cls.__name__)
        return impl_cls(in_features, out_features, w_bit, group_size)
