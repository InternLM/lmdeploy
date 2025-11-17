# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
from torch import nn

from lmdeploy.pytorch.distributed import get_tp_world_rank

from ..backends import OpType, get_backend
from .utils import chunk_aligned, get_distribute_size


def _is_w8a8(quant_config: Any):
    """Is w8a8."""
    quant_dtype = None
    w8a8_flag = False
    if quant_config is not None:
        quant_method = quant_config['quant_method']
        if quant_method == 'smooth_quant':
            w8a8_flag = True
            quant_dtype = quant_config.get('quant_dtype', 'int8')
            quant_dtype = eval(f'torch.{quant_dtype}')
    return w8a8_flag, quant_dtype


class RMSNorm(nn.Module):
    """RMS Norm with add residual."""

    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-6,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 quant_config: Any = None,
                 tp: bool = False,
                 align: int = 1):
        super().__init__()
        backend = get_backend()

        w8a8_flag, quant_dtype = _is_w8a8(quant_config)

        if w8a8_flag:
            builder = backend.get_layer_impl_builder(OpType.RMSNormW8A8)
        else:
            builder = backend.get_layer_impl_builder(OpType.RMSNorm)

        if tp:
            world_size, rank = get_tp_world_rank('attn')
            hidden_size = get_distribute_size(hidden_size, world_size, rank, align=align)

        self.register_parameter('weight', self.create_weight(hidden_size, dtype, device))
        if w8a8_flag:
            self.impl = builder.build(hidden_size, eps, quant_dtype=quant_dtype)
        else:
            self.impl = builder.build(hidden_size, eps)

        if tp:
            self.weight.weight_loader = self.weight_loader
        self.align = align

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Weight loader."""
        world_size, rank = get_tp_world_rank('attn')
        loaded_weight = chunk_aligned(loaded_weight, world_size, 0, self.align)[rank]
        param.copy_(loaded_weight)

    @staticmethod
    def create_weight(hidden_size: int, dtype: torch.dtype = None, device: torch.device = None):
        """Create weight."""
        if dtype is None:
            dtype = torch.float16
        if device is None:
            device = 'cuda'
        weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device), requires_grad=False)
        return weight

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        return self.impl.forward(x, self.weight, residual)


class LayerNorm(nn.Module):
    """Layer Norm with add residual."""

    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-6,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.LayerNorm)
        weight, bias = self.create_weight(hidden_size, bias, dtype, device)
        self.register_parameter('weight', weight)
        self.register_parameter('bias', bias)
        self.impl = builder.build(hidden_size, eps)

    @staticmethod
    def create_weight(hidden_size: int, bias: bool = True, dtype: torch.dtype = None, device: torch.device = None):
        """Create weight."""
        if dtype is None:
            dtype = torch.float16
        if device is None:
            device = 'cuda'
        weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device), requires_grad=False)
        if bias:
            bias = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device), requires_grad=False)
        else:
            bias = None

        return weight, bias

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        return self.impl.forward(x, self.weight, self.bias, residual)
