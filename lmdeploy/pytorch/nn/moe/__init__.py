# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, Optional

import torch

from lmdeploy.pytorch.models.patch import get_build_model_context

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
    quant_config: Dict = None,
    layer_idx: int = 0,
    act_func: Callable = None,
    prefix: str = '',
):
    """Fused moe builder."""
    quant_method = None
    if quant_config is not None:
        quant_config = get_build_model_context().quant_config
        quant_method = quant_config.get_quant_method(prefix)

    if quant_method is None:
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

    if quant_method == 'smooth_quant':
        assert not bias, 'Quant model does not support bias for now.'
        assert act_func is None, ('Quant model does not support activation function for now.')
        from .w8a8 import FusedMoEW8A8
        return FusedMoEW8A8(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            renormalize=renormalize,
            dtype=dtype,
            quant_dtype=quant_config.quant_dtype,
            device=device,
            all_reduce=all_reduce,
        )
    elif quant_method == 'fp8':
        from .blocked_fp8 import FusedMoEBlockedF8
        return FusedMoEBlockedF8(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            bias=bias,
            renormalize=renormalize,
            fp8_dtype=quant_config.quant_dtype,
            scale_fmt=quant_config.scale_fmt,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
            layer_idx=layer_idx,
            act_func=act_func,
        )
    else:
        raise RuntimeError(f'Unsupported quant method: {quant_method}')
