# Copyright (c) OpenMMLab. All rights reserved.

import functools
from abc import abstractmethod

import torch
from packaging import version

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.envs import blocked_fp8_gemm_backend
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import blocked_gemm_fp8, deep_gemm_fp8, quant_fp8, quant_fp8_tma
from lmdeploy.utils import get_logger

from ..blockedf8_modules import LinearBlockedF8Builder, LinearBlockedF8Impl
from .warmup_manager import WarmupMeta, get_warmup_manager

logger = get_logger('lmdeploy')

_GLUON_TRITON_MIN_VERSION = version.parse('3.6.0')
_GLUON_TRITON_MAX_EXCLUSIVE = version.parse('3.8.0')


def _is_supported_gluon_triton_version(triton_version: str) -> bool:
    """Whether the installed Triton has the validated experimental API."""
    try:
        parsed_version = version.parse(triton_version)
    except version.InvalidVersion:
        return False
    return _GLUON_TRITON_MIN_VERSION <= parsed_version < _GLUON_TRITON_MAX_EXCLUSIVE


class CudaLinearBlockedF8Impl(LinearBlockedF8Impl):
    """Common CUDA blocked-FP8 linear lifecycle around a provider GEMM."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 block_size: int,
                 out_dtype: torch.dtype = torch.float16,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.fp8_dtype = fp8_dtype
        self.block_size = block_size

    @abstractmethod
    def _gemm(self, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
        raise NotImplementedError

    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: torch.Tensor | None = None,
                all_reduce: bool = False,
                group: dist.ProcessGroup | None = None,
                rank: int = 0,
                scatter_size: list[int] = None):
        """Quantize, run the selected GEMM, and apply linear postprocessing."""
        x_shape = x.shape
        x = x.flatten(0, -2)
        out = self._gemm(x, weight, scale)

        if bias is not None:
            out += bias
        out = out.unflatten(0, x_shape[:-1])

        if all_reduce:
            if scatter_size is not None:
                out = dist.reduce_scatter_by_tp_sizes(out, rank, scatter_size, group=group)
            else:
                dist.all_reduce(out, group=group)
        return out


class TritonLinearBlockedF8Impl(CudaLinearBlockedF8Impl):
    """Portable Triton blocked-FP8 linear implementation."""

    def _gemm(self, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
        input_quant, input_scale = quant_fp8(x,
                                             self.block_size,
                                             dtype=weight.dtype,
                                             trans_scale=True,
                                             scale_fmt=self.scale_fmt)
        return blocked_gemm_fp8(input_quant, input_scale, weight.t(), scale.t(), out_dtype=x.dtype)


class GluonLinearBlockedF8Impl(CudaLinearBlockedF8Impl):
    """Hopper Gluon blocked-FP8 linear implementation."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 block_size: int,
                 out_dtype: torch.dtype = torch.bfloat16,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn):
        super().__init__(in_features, out_features, block_size, out_dtype, fp8_dtype)
        from lmdeploy.pytorch.kernels.cuda.blocked_fp8_gemm_gluon import fp8_gemm_nt

        self._fp8_gemm_nt = fp8_gemm_nt

        warmup_mgr = get_warmup_manager()
        key = f'gluon_blockedfp8_gemm_{in_features}_{out_features}_{block_size}_{out_dtype}_{fp8_dtype}'
        if key not in warmup_mgr:
            warmup_mgr[key] = self.warmup

    def _gemm(self, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
        input_quant, input_scale = quant_fp8(x,
                                             self.block_size,
                                             dtype=weight.dtype,
                                             trans_scale=True,
                                             scale_fmt=self.scale_fmt)
        out = x.new_empty((x.size(0), weight.size(0)))
        self._fp8_gemm_nt((input_quant, input_scale), (weight, scale), out, None)
        return out

    def warmup(self, warmup_meta: WarmupMeta):
        """Compile one representative from every reachable M schedule."""
        max_num_tokens = warmup_meta.max_num_tokens
        if max_num_tokens <= 0:
            return

        device = 'cuda'
        k, n = self.in_features, self.out_features
        block_size = self.block_size
        weight = torch.empty(n, k, dtype=self.fp8_dtype, device=device)
        scale = torch.empty(
            ((n + block_size - 1) // block_size, (k + block_size - 1) // block_size),
            dtype=torch.float32,
            device=device,
        )

        # M is dynamic inside each schedule. These values compile only the
        # Python-selected tiny, small, mid, and persistent configurations.
        candidates = (1, 16, 128, 256, 257, max_num_tokens)
        for m in sorted({m for m in candidates if m <= max_num_tokens}):
            inputs = torch.empty(m, k, dtype=self.out_dtype, device=device)
            self._gemm(inputs, weight, scale)


class DeepGemmLinearBlockedF8Impl(CudaLinearBlockedF8Impl):
    """DeepGEMM blocked-FP8 linear implementation."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 block_size: int,
                 out_dtype: torch.dtype = torch.bfloat16,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn):
        super().__init__(in_features, out_features, block_size, out_dtype, fp8_dtype)
        from lmdeploy.pytorch.third_party.deep_gemm import PDL_ENABLED

        self.pdl_enabled = PDL_ENABLED

        warmup_mgr = get_warmup_manager()
        key = f'deepgemm_blockedfp8_gemm_{in_features}_{out_features}_{block_size}_{out_dtype}_{fp8_dtype}'
        if key not in warmup_mgr:
            warmup_mgr[key] = self.warmup

    def warmup(self, warmup_meta: WarmupMeta):
        """Warm up DeepGEMM specializations up to the configured token cap."""
        import random

        from lmdeploy.pytorch.third_party.deep_gemm import get_m_alignment_for_contiguous_layout

        device = 'cuda'
        max_num_tokens = warmup_meta.max_num_tokens
        alignment = get_m_alignment_for_contiguous_layout()
        range_end = max_num_tokens + alignment - 1
        k, n = self.in_features, self.out_features
        block_size = self.block_size
        weight = torch.empty(n, k, dtype=self.fp8_dtype, device=device)
        scale = torch.empty(
            ((n + block_size - 1) // block_size, (k + block_size - 1) // block_size),
            dtype=torch.float32,
            device=device,
        )
        # Shuffle ranges so ranks might compile different kernels concurrently.
        ranges = list(range(alignment, range_end, alignment))
        random.shuffle(ranges)
        for m in ranges:
            inputs = torch.empty(m, k, dtype=self.out_dtype, device=device)
            self._gemm(inputs, weight, scale)

    def _gemm(self, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
        input_quant, input_scale = quant_fp8_tma(x,
                                                 self.block_size,
                                                 dtype=weight.dtype,
                                                 scale_fmt=self.scale_fmt,
                                                 launch_pdl=self.pdl_enabled)
        out = deep_gemm_fp8(input_quant, input_scale, weight, scale, out_dtype=x.dtype)
        return out[:x.size(0)]


@functools.lru_cache
def _has_deep_gemm() -> bool:
    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


@functools.lru_cache
def _has_gluon() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return False
    try:
        import triton

        if not _is_supported_gluon_triton_version(triton.__version__):
            return False
        from lmdeploy.pytorch.kernels.cuda.blocked_fp8_gemm_gluon import fp8_gemm_nt  # noqa: F401
    except (AttributeError, ImportError):
        return False
    return True


def _supports_deep_gemm(block_size: int, dtype: torch.dtype, fp8_dtype: torch.dtype) -> bool:
    return (_has_deep_gemm() and block_size == 128 and dtype == torch.bfloat16
            and fp8_dtype == torch.float8_e4m3fn)


def _supports_gluon(in_features: int, out_features: int, block_size: int, dtype: torch.dtype,
                    fp8_dtype: torch.dtype) -> bool:
    return (_has_gluon() and block_size == 128 and dtype == torch.bfloat16
            and fp8_dtype == torch.float8_e4m3fn and in_features % block_size == 0 and out_features % 8 == 0)


class CudaLinearBlockedF8Builder(LinearBlockedF8Builder):
    """Select one blocked-FP8 GEMM provider when constructing the layer."""

    @staticmethod
    def build(in_features: int,
              out_features: int,
              block_size: int = 128,
              bias: bool = True,
              dtype: torch.dtype = None,
              fp8_dtype: torch.dtype = torch.float8_e4m3fn):
        """Build the requested provider or the best compatible provider."""
        provider = blocked_fp8_gemm_backend

        if provider == 'auto':
            if _supports_deep_gemm(block_size, dtype, fp8_dtype):
                impl_cls = DeepGemmLinearBlockedF8Impl
            elif _supports_gluon(in_features, out_features, block_size, dtype, fp8_dtype):
                impl_cls = GluonLinearBlockedF8Impl
            else:
                impl_cls = TritonLinearBlockedF8Impl
        elif provider == 'deepgemm':
            if not _supports_deep_gemm(block_size, dtype, fp8_dtype):
                raise RuntimeError('DeepGEMM blocked-FP8 linear was requested but is unavailable or incompatible.')
            impl_cls = DeepGemmLinearBlockedF8Impl
        elif provider == 'gluon':
            if not _supports_gluon(in_features, out_features, block_size, dtype, fp8_dtype):
                try:
                    import triton
                    triton_version = triton.__version__
                except ImportError:
                    triton_version = 'not installed'
                raise RuntimeError('Gluon blocked-FP8 linear requires Hopper, BF16 output, FP8 E4M3, block size 128, '
                                   'K divisible by 128, N divisible by 8, and Triton '
                                   f'>={_GLUON_TRITON_MIN_VERSION},<{_GLUON_TRITON_MAX_EXCLUSIVE} '
                                   f'(found {triton_version}).')
            impl_cls = GluonLinearBlockedF8Impl
        else:
            impl_cls = TritonLinearBlockedF8Impl

        logger.debug(f'Build LinearBlockedF8 with {impl_cls.__name__}.')
        return impl_cls(in_features, out_features, block_size, dtype, fp8_dtype)
