# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Callable, Optional

import torch

from .base import MoeType, SoftmaxTopK  # noqa: F401


def build_fused_moe(
    hidden_dim: int,
    ffn_dim: int,
    num_experts: int,
    top_k: int,
    bias: bool = False,
    renormalize: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    all_reduce: bool = True,
    enable_ep: bool = False,
    quant_config: Any = None,
    layer_idx: int = 0,
    act_func: Callable = None,
):
    """Fused moe builder."""

    if quant_config is None:
        from .default import FusedMoE
        return FusedMoE(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            bias=bias,
            renormalize=renormalize,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
            layer_idx=layer_idx,
            act_func=act_func,
        )

    quant_method = quant_config['quant_method']
    if quant_method == 'smooth_quant':
        assert not bias, 'Quant model does not support bias for now.'
        assert act_func is None, ('Quant model does not support activation function for now.')
        quant_dtype = eval('torch.' + quant_config.get('quant_dtype', 'int8'))
        from .w8a8 import FusedMoEW8A8
        return FusedMoEW8A8(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            renormalize=renormalize,
            dtype=dtype,
            quant_dtype=quant_dtype,
            device=device,
            all_reduce=all_reduce,
        )
    elif quant_method == 'fp8':
        fmt = quant_config.get('fmt', 'e4m3')
        if fmt == 'e4m3':
            fp8_dtype = torch.float8_e4m3fn
        elif fmt == 'e5m2':
            fp8_dtype = torch.float8_e5m2
        else:
            raise TypeError(f'Unsupported fp8 fmt: {fmt}')
        scale_fmt = quant_config.get('scale_fmt', None)
        from .blocked_fp8 import FusedMoEBlockedF8
        return FusedMoEBlockedF8(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            bias=bias,
            renormalize=renormalize,
            fp8_dtype=fp8_dtype,
            scale_fmt=scale_fmt,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
            layer_idx=layer_idx,
            act_func=act_func,
        )
    else:
        raise RuntimeError(f'Unsupported quant method: {quant_method}')
