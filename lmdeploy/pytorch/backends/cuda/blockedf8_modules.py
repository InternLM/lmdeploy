# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import blocked_gemm_fp8, deep_gemm_fp8, quant_fp8, quant_fp8_tma
from lmdeploy.utils import get_logger

from ..blockedf8_modules import LinearBlockedF8Builder, LinearBlockedF8Impl
from .warmup_manager import WarmupMeta, get_warmup_manager

logger = get_logger('lmdeploy')


class TritonLinearBlockedF8Impl(LinearBlockedF8Impl):
    """Triton linear blocked f8 implementation."""

    def __init__(self, in_features: int, out_features: int, block_size: int, out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.block_size = block_size

    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                group: Optional[dist.ProcessGroup] = None,
                rank: int = 0,
                scatter_size: List[int] = None):
        """forward."""
        x_shape = x.shape
        x = x.flatten(0, -2)
        input_quant, input_scale = quant_fp8(x,
                                             self.block_size,
                                             dtype=weight.dtype,
                                             trans_scale=True,
                                             scale_fmt=self.scale_fmt)

        out = blocked_gemm_fp8(input_quant,
                               input_scale,
                               weight.t(),
                               scale.t(),
                               out_dtype=x.dtype,
                               scale_fmt=self.scale_fmt)
        if bias is not None:
            out += bias

        out = out.unflatten(0, x_shape[:-1])

        if all_reduce:
            if scatter_size is not None:
                out = dist.reduce_scatter_by_tp_sizes(out, rank, scatter_size, group=group)
            else:
                dist.all_reduce(out)
        return out


class TritonLinearBlockedF8Builder(LinearBlockedF8Builder):
    """Triton linear blocked f8 implementation builder."""

    @staticmethod
    def build(in_features: int, out_features: int, block_size: int = 128, bias: bool = True, dtype: torch.dtype = None):
        """build."""
        try:
            import deep_gemm  # noqa
            logger.debug('build with DeepGemmLinearBlockedF8Impl')
            return DeepGemmLinearBlockedF8Impl(in_features, out_features, block_size, dtype)
        except:  # noqa
            logger.warning('Failed to import deep_gemm, LinearBlockedF8 fallback to triton implementation.')
            return TritonLinearBlockedF8Impl(in_features, out_features, block_size, dtype)


class DeepGemmLinearBlockedF8Impl(LinearBlockedF8Impl):
    """Deep gemm blocked f8 implementation."""

    def __init__(self, in_features: int, out_features: int, block_size: int, out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.block_size = block_size

        warmup_mgr = get_warmup_manager()
        key = ('deepgemm_blockedfp8_gemm_'
               f'{in_features}_{out_features}_{block_size}_{out_dtype}')
        if key not in warmup_mgr:
            warmup_mgr[key] = self.warmup

    def warmup(self, warmup_meta: WarmupMeta):
        """warmup."""
        import random

        from lmdeploy.pytorch.third_party.deep_gemm import get_m_alignment_for_contiguous_layout
        device = 'cuda'
        max_num_tokens = warmup_meta.max_num_tokens
        alignment = get_m_alignment_for_contiguous_layout()
        range_end = max_num_tokens + alignment - 1
        k, n = self.in_features, self.out_features
        block_size = self.block_size
        weight = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=device)
        scale = torch.empty(((n + block_size - 1) // block_size, (k + block_size - 1) // block_size),
                            dtype=torch.float32,
                            device=device)
        # shuffle ranges so ranks might compile different kernels concurrently.
        ranges = list(range(alignment, range_end, alignment))
        random.shuffle(ranges)
        for m in ranges:
            inputs = torch.empty(m, k, dtype=self.out_dtype, device=device)
            input_quant, input_scale = quant_fp8_tma(inputs,
                                                     self.block_size,
                                                     dtype=weight.dtype,
                                                     scale_fmt=self.scale_fmt)
            deep_gemm_fp8(input_quant, input_scale, weight, scale, out_dtype=inputs.dtype)

    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                group: Optional[dist.ProcessGroup] = None,
                rank: int = 0,
                scatter_size: List[int] = None):
        """forward."""
        x_shape = x.shape
        x = x.flatten(0, -2)
        input_quant, input_scale = quant_fp8_tma(x, self.block_size, dtype=weight.dtype, scale_fmt=self.scale_fmt)

        out = deep_gemm_fp8(input_quant, input_scale, weight, scale, out_dtype=x.dtype)
        out = out[:x.size(0)]
        if bias is not None:
            out += bias
        out = out.unflatten(0, x_shape[:-1])

        if all_reduce:
            if scatter_size is not None:
                out = dist.reduce_scatter_by_tp_sizes(out, rank, scatter_size, group=group)
            else:
                dist.all_reduce(out, group=group)
        return out
