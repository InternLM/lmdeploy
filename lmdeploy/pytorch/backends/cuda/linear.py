# Copyright (c) OpenMMLab. All rights reserved.
"""Opt-in small-token BF16 projection schedules for SM90."""

import torch
import torch.distributed as dist

from lmdeploy.pytorch.kernels.cuda.skinny_gemm import skinny_gemm, warmup_skinny_gemm
from lmdeploy.utils import get_logger

from ..default.linear import DefaultLinearImpl
from .warmup_manager import get_warmup_manager

logger = get_logger('lmdeploy')


class SkinnyLinearImpl(DefaultLinearImpl):
    """Preserve default linear communication around a measured local GEMM."""

    def __init__(self, n: int, k: int, plans):
        self.n = n
        self.k = k
        self.plans = plans
        manager = get_warmup_manager()
        key = f'bf16_skinny_{n}_{k}'
        if key not in manager:
            manager[key] = self.warmup
            logger.info('BF16 skinny GEMM enabled for (%d, %d), token rows %s', n, k, sorted(plans))

    def warmup(self, meta):
        for m, config in self.plans.items():
            warmup_skinny_gemm(self.n, self.k, m, config)

    def forward(self, x, weight: torch.Tensor, bias=None, all_reduce=False, group=None, rank=0, scatter_size=None):
        m = x.numel() // self.k
        config = self.plans.get(m)
        eligible = (config is not None and bias is None and x.is_cuda and x.device == weight.device
                    and x.dtype == weight.dtype == torch.bfloat16 and x.is_contiguous() and weight.is_contiguous()
                    and weight.shape == (self.n, self.k))
        if not eligible:
            return super().forward(x, weight, bias, all_reduce, group, rank, scatter_size)
        out = skinny_gemm(x.view(m, self.k), weight, config).view(*x.shape[:-1], self.n)
        if all_reduce:
            if scatter_size is not None:
                from lmdeploy.pytorch.distributed import reduce_scatter_by_tp_sizes
                out = reduce_scatter_by_tp_sizes(out, rank, scatter_size, group=group)
            else:
                dist.all_reduce(out, group=group)
        return out
