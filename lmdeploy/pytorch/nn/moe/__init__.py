# Copyright (c) Shanghai AI Laboratory. All rights reserved.
from collections.abc import Callable

import torch

from lmdeploy.pytorch.models.patch import get_build_model_context

from .base import MoeType, SoftmaxTopK  # noqa: F401
from .v4_fp4 import FusedMoEV4FP4  # noqa: F401


def build_fused_moe(
    hidden_dim: int,
    ffn_dim: int,
    num_experts: int,
    top_k: int,
    bias: bool = False,
    renormalize: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    all_reduce: bool = True,
    enable_ep: bool = False,
    quant_config: dict = None,
    layer_idx: int = 0,
    act_func: Callable = None,
    prefix: str = '',
    *,
    output_scale: float = 1.0,
    fp32_acc: bool = False,
):
    """Fused moe builder."""
    quant_method = None
    if quant_config is not None:
        quant_config = get_build_model_context().quant_config
        quant_method = quant_config.get_quant_method(prefix, module_kind='moe')

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
            output_scale=output_scale,
            fp32_acc=fp32_acc,
        )

    if quant_method == 'smooth_quant':
        if fp32_acc or output_scale != 1.0:
            raise NotImplementedError('W8A8 MoE does not support fp32_acc or output_scale.')
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
        is_static_per_tensor = (
            quant_config.activation_scheme == 'static'
            and quant_config.weight_block_size is None
        )

        if is_static_per_tensor:
            if fp32_acc or output_scale != 1.0:
                raise NotImplementedError('Static FP8 MoE does not support fp32_acc or output_scale.')
            assert not bias, (
                'Static FP8 MoE does not support bias.'
            )
            assert act_func is None, (
                'Static FP8 MoE currently uses '
                'the built-in SiLU activation.'
            )

            from .static_fp8 import FusedMoEStaticF8

            return FusedMoEStaticF8(
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
            output_scale=output_scale,
            fp32_acc=fp32_acc,
        )
    elif quant_method == 'compressed-tensors':
        # W4A16 already requests FP32 expert reduction for every invocation.
        if output_scale != 1.0:
            raise NotImplementedError('W4A16 MoE does not support output_scale.')
        if bias:
            raise RuntimeError('Compressed-tensors W4A16 routed experts do not support bias.')
        if act_func is not None:
            raise RuntimeError('Compressed-tensors W4A16 only supports the built-in SiLU activation.')
        # The TurboMind grouped route serves the group-128 ct checkpoints the
        # packed-kernel route rejects. The load-time ct -> AWQ conversion has
        # already normalized the tensors, so the TM route consumes the same
        # canonical AWQ parameters as awq-sourced checkpoints. Group-32
        # checkpoints keep the packed-kernel route below.
        if quant_config.group_size == 128:
            from lmdeploy.pytorch.backends.cuda.tm_moe_modules import tm_moe_route

            reason = tm_moe_route(
                group_size=quant_config.group_size,
                w_bit=quant_config.bits,
                dtype=dtype,
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                prefix=prefix,
            )
            if reason is None:
                from .tm_w4a16 import FusedMoETmW4A16
                return FusedMoETmW4A16(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    renormalize=renormalize,
                    dtype=dtype,
                    device=device,
                    all_reduce=all_reduce,
                    num_bits=quant_config.bits,
                    group_size=quant_config.group_size,
                    layer_idx=layer_idx,
                )
        from .compressed_tensors import FusedMoEW4A16
        return FusedMoEW4A16(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            renormalize=renormalize,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
            num_bits=quant_config.bits,
            group_size=quant_config.group_size,
            layer_idx=layer_idx,
        )
    elif quant_method == 'awq':
        # The AWQ + MoE combination routes to the TurboMind grouped
        # W4A16 channel. No triton fallback exists for the AWQ expert layout,
        # so a declined route (probe rejection, or the backend env set off
        # the TurboMind values) keeps the historical unsupported error, now
        # with the probe reason attached.
        if output_scale != 1.0:
            raise NotImplementedError('W4A16 MoE does not support output_scale.')
        if bias:
            raise RuntimeError('AWQ W4A16 routed experts do not support bias.')
        if act_func is not None:
            raise RuntimeError('AWQ W4A16 only supports the built-in SiLU activation.')
        from lmdeploy.pytorch.backends.cuda.tm_moe_modules import tm_moe_route

        reason = tm_moe_route(
            group_size=quant_config.group_size,
            w_bit=quant_config.bits,
            dtype=dtype,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            prefix=prefix,
        )
        if reason is not None:
            raise RuntimeError(
                'AWQ W4A16 routed experts require the TurboMind grouped-GEMM route, which is unavailable: '
                f'{reason}.')
        from .tm_w4a16 import FusedMoETmW4A16
        return FusedMoETmW4A16(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            renormalize=renormalize,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
            num_bits=quant_config.bits,
            group_size=quant_config.group_size,
            layer_idx=layer_idx,
        )
    else:
        raise RuntimeError(f'Unsupported quant method: {quant_method}')
