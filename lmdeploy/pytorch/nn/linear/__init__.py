# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional

import torch
from torch import nn

from lmdeploy.pytorch.distributed import get_dist_manager, get_dp_world_rank, get_tp_world_rank

from .awq import AwqLinear, MergedAwqLinear, QKVAwqLinear
from .blocked_fp8 import BlockedF8Linear, MergedBlockedF8Linear, QKVBlockedF8Linear
from .default import BaseLinear, MergedBaseLinear, QKVBaseLinear
from .lora import LoRA  # noqa: F401
from .w8a8 import MergedW8A8Linear, QKVW8A8Linear, W8A8Linear


def _is_dp_enabled():
    """Is dp."""
    return get_dp_world_rank()[0] > 1


def _get_dp_gather(is_tp: bool):
    """Get dp gather."""
    dp_gather = True
    if not _is_dp_enabled():
        # disable if not dp
        dp_gather = False
    if not is_tp:
        dp_gather = False
    return dp_gather


def _get_dp_tp_meta(all_reduce: bool = True):
    """Get tp meta."""
    dist_ctx = get_dist_manager().current_context()
    dist_attn_cfg = dist_ctx.dist_config.attn_config
    tp = dist_attn_cfg.tp
    is_tp = tp > 1
    all_reduce = all_reduce if is_tp else False
    return is_tp, all_reduce


def build_linear(in_features: int,
                 out_features: int,
                 bias: bool,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 colwise: bool = True,
                 is_tp: bool = False,
                 quant_config: Any = None,
                 all_reduce: bool = True,
                 tp_align_size: int = 1,
                 dp_gather: bool = False,
                 dp_scatter: bool = False) -> nn.Module:
    """Build linear."""
    if is_tp:
        is_tp = get_tp_world_rank()[0] > 1
    if not is_tp:
        all_reduce = False

    if (dp_scatter or dp_gather) and quant_config is not None:
        quant_method = quant_config['quant_method']
        assert quant_method in ['fp8'], (f'Do not support dp_gather with quant_method={quant_method}')

    if quant_config is None:
        return BaseLinear(
            in_features,
            out_features,
            bias=bias,
            dtype=dtype,
            device=device,
            colwise=colwise,
            is_tp=is_tp,
            all_reduce=all_reduce,
            tp_align_size=tp_align_size,
            dp_gather=dp_gather,
            dp_scatter=dp_scatter,
        )

    quant_method = quant_config['quant_method']
    quant_dtype = torch.int8
    if 'quant_dtype' in quant_config:
        quant_dtype = eval('torch.' + quant_config['quant_dtype'])

    if quant_method == 'awq':
        w_bit = quant_config.get('bits', 4)
        group_size = quant_config.get('group_size', 128)
        return AwqLinear(
            in_features,
            out_features,
            w_bit=w_bit,
            group_size=group_size,
            bias=bias,
            device=device,
            colwise=colwise,
            is_tp=is_tp,
            all_reduce=all_reduce,
        )
    if quant_method == 'smooth_quant':
        return W8A8Linear(in_features,
                          out_features,
                          bias=bias,
                          dtype=dtype,
                          device=device,
                          colwise=colwise,
                          is_tp=is_tp,
                          all_reduce=all_reduce,
                          quant_dtype=quant_dtype)
    elif quant_method == 'fp8':
        fmt = quant_config.get('fmt', 'e4m3')
        if fmt == 'e4m3':
            fp8_dtype = torch.float8_e4m3fn
        elif fmt == 'e5m2':
            fp8_dtype = torch.float8_e5m2
        else:
            raise TypeError(f'Unsupported fp8 fmt: {fmt}')
        return BlockedF8Linear(
            in_features,
            out_features,
            bias=bias,
            fp8_dtype=fp8_dtype,
            dtype=dtype,
            device=device,
            colwise=colwise,
            is_tp=is_tp,
            all_reduce=all_reduce,
            dp_gather=dp_gather,
            dp_scatter=dp_scatter,
        )
    else:
        raise RuntimeError(f'Unsupported quant method: {quant_method}')


def build_colwise_linear(in_features: int,
                         out_features: int,
                         bias: bool,
                         dtype: Optional[torch.dtype] = None,
                         device: Optional[torch.device] = None,
                         is_tp: bool = False,
                         tp_align_size: int = 1,
                         quant_config: Any = None,
                         dp_disable_tp: bool = False,
                         dp_gather: bool = False) -> nn.Module:
    """Build columnwise parallel linear layer."""
    if dp_disable_tp and is_tp:
        is_tp, _ = _get_dp_tp_meta()
    elif is_tp:
        is_tp = get_tp_world_rank()[0] > 1

    if dp_gather:
        assert not dp_disable_tp
        dp_gather = _get_dp_gather(is_tp)

    return build_linear(in_features=in_features,
                        out_features=out_features,
                        bias=bias,
                        dtype=dtype,
                        device=device,
                        colwise=True,
                        is_tp=is_tp,
                        quant_config=quant_config,
                        all_reduce=False,
                        tp_align_size=tp_align_size,
                        dp_gather=dp_gather)


def build_rowwise_linear(in_features: int,
                         out_features: int,
                         bias: bool,
                         dtype: Optional[torch.dtype] = None,
                         device: Optional[torch.device] = None,
                         is_tp: bool = False,
                         tp_align_size: int = 1,
                         quant_config: Any = None,
                         all_reduce: bool = True,
                         dp_disable_tp: bool = False,
                         dp_scatter: bool = False) -> nn.Module:
    """Build rowwise parallel linear layer."""
    if dp_disable_tp and is_tp:
        is_tp, all_reduce = _get_dp_tp_meta(all_reduce)
    return build_linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        dtype=dtype,
        device=device,
        colwise=False,
        is_tp=is_tp,
        quant_config=quant_config,
        all_reduce=all_reduce,
        tp_align_size=tp_align_size,
        dp_scatter=dp_scatter,
    )


def build_merged_colwise_linear(
    in_features: int,
    all_out_features: List[int],
    bias: bool,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    quant_config: Any = None,
    is_tp: bool = True,
    out_names: List[Any] = None,
    dp_gather: bool = False,
):
    """Merge linear."""
    if is_tp:
        is_tp = get_tp_world_rank()[0] > 1

    if dp_gather and quant_config is not None:
        quant_method = quant_config['quant_method']
        assert quant_method in ['fp8'], (f'Do not support dp_gather with quant_method={quant_method}')

    if quant_config is None:
        return MergedBaseLinear(in_features=in_features,
                                all_out_features=all_out_features,
                                bias=bias,
                                dtype=dtype,
                                device=device,
                                is_tp=is_tp,
                                out_names=out_names,
                                dp_gather=dp_gather)

    quant_method = quant_config['quant_method']
    quant_dtype = torch.int8
    if 'quant_dtype' in quant_config:
        quant_dtype = eval('torch.' + quant_config['quant_dtype'])

    if quant_method == 'awq':
        w_bit = quant_config.get('bits', 4)
        group_size = quant_config.get('group_size', 128)
        return MergedAwqLinear(
            in_features,
            all_out_features=all_out_features,
            w_bit=w_bit,
            group_size=group_size,
            bias=bias,
            device=device,
            is_tp=is_tp,
        )
    if quant_method == 'smooth_quant':
        return MergedW8A8Linear(in_features=in_features,
                                all_out_features=all_out_features,
                                bias=bias,
                                dtype=dtype,
                                device=device,
                                is_tp=is_tp,
                                out_names=out_names,
                                quant_dtype=quant_dtype)
    elif quant_method == 'fp8':
        fmt = quant_config.get('fmt', 'e4m3')
        if fmt == 'e4m3':
            fp8_dtype = torch.float8_e4m3fn
        elif fmt == 'e5m2':
            fp8_dtype = torch.float8_e5m2
        else:
            raise TypeError(f'Unsupported fp8 fmt: {fmt}')
        return MergedBlockedF8Linear(
            in_features=in_features,
            all_out_features=all_out_features,
            bias=bias,
            fp8_dtype=fp8_dtype,
            dtype=dtype,
            device=device,
            is_tp=is_tp,
            out_names=out_names,
            dp_gather=dp_gather,
        )
    else:
        raise RuntimeError(f'Unsupported quant method: {quant_method}')


def build_qkv_proj(in_features: int,
                   num_q_heads: int,
                   num_kv_heads: int,
                   head_size: int,
                   head_size_v: int = None,
                   bias: bool = False,
                   quant_config: Any = None,
                   dtype: Optional[torch.dtype] = None,
                   device: Optional[torch.device] = None,
                   is_tp: bool = True,
                   num_replicate_kv_heads: int = 1,
                   dp_disable_tp: bool = True,
                   all_reduce: bool = False,
                   dp_gather: bool = False):
    """Build qkv proj."""
    if dp_disable_tp and is_tp:
        is_tp, _ = _get_dp_tp_meta(all_reduce)
    elif is_tp:
        is_tp = get_tp_world_rank()[0] > 1

    if dp_gather:
        assert not dp_disable_tp
        dp_gather = _get_dp_gather(is_tp)

    if head_size_v is None:
        head_size_v = head_size

    if quant_config is None:
        return QKVBaseLinear(in_features=in_features,
                             num_q_heads=num_q_heads,
                             num_kv_heads=num_kv_heads,
                             head_size=head_size,
                             head_size_v=head_size_v,
                             bias=bias,
                             dtype=dtype,
                             device=device,
                             is_tp=is_tp,
                             num_replicate_kv_heads=num_replicate_kv_heads)

    quant_method = quant_config['quant_method']
    quant_dtype = torch.int8
    if 'quant_dtype' in quant_config:
        quant_dtype = eval('torch.' + quant_config['quant_dtype'])

    if quant_method == 'awq':
        w_bit = quant_config.get('bits', 4)
        group_size = quant_config.get('group_size', 128)
        return QKVAwqLinear(in_features=in_features,
                            num_q_heads=num_q_heads,
                            num_kv_heads=num_kv_heads,
                            head_size=head_size,
                            head_size_v=head_size_v,
                            w_bit=w_bit,
                            group_size=group_size,
                            bias=bias,
                            device=device,
                            is_tp=is_tp,
                            num_replicate_kv_heads=num_replicate_kv_heads)
    if quant_method == 'smooth_quant':
        return QKVW8A8Linear(in_features=in_features,
                             num_q_heads=num_q_heads,
                             num_kv_heads=num_kv_heads,
                             head_size=head_size,
                             head_size_v=head_size_v,
                             bias=bias,
                             dtype=dtype,
                             device=device,
                             is_tp=is_tp,
                             num_replicate_kv_heads=num_replicate_kv_heads,
                             quant_dtype=quant_dtype)
    if quant_method == 'fp8':
        fmt = quant_config.get('fmt', 'e4m3')
        if fmt == 'e4m3':
            fp8_dtype = torch.float8_e4m3fn
        elif fmt == 'e5m2':
            fp8_dtype = torch.float8_e5m2
        else:
            raise TypeError(f'Unsupported fp8 fmt: {fmt}')
        return QKVBlockedF8Linear(in_features=in_features,
                                  num_q_heads=num_q_heads,
                                  num_kv_heads=num_kv_heads,
                                  head_size=head_size,
                                  head_size_v=head_size_v,
                                  bias=bias,
                                  fp8_dtype=fp8_dtype,
                                  dtype=dtype,
                                  device=device,
                                  is_tp=is_tp,
                                  dp_gather=dp_gather,
                                  num_replicate_kv_heads=num_replicate_kv_heads)
    else:
        raise RuntimeError(f'Unsupported quant method: {quant_method}')


def build_o_proj(in_features: int,
                 out_features: int,
                 bias: bool,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = False,
                 tp_align_size: int = 1,
                 quant_config: Any = None,
                 all_reduce: bool = True) -> nn.Module:
    """Build down linear."""
    if is_tp:
        is_tp, all_reduce = _get_dp_tp_meta(all_reduce)

    return build_rowwise_linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        dtype=dtype,
        device=device,
        is_tp=is_tp,
        tp_align_size=tp_align_size,
        quant_config=quant_config,
        all_reduce=all_reduce,
    )


def build_gateup_linear(in_features: int,
                        all_out_features: List[int],
                        bias: bool,
                        dtype: Optional[torch.dtype] = None,
                        device: Optional[torch.device] = None,
                        quant_config: Any = None,
                        is_tp: bool = True,
                        out_names: List[Any] = None,
                        dp_gather: bool = True):
    """Build gate up linear."""
    if dp_gather:
        if is_tp:
            is_tp = get_tp_world_rank()[0] > 1
        dp_gather = _get_dp_gather(is_tp)
    elif is_tp:
        is_tp, _ = _get_dp_tp_meta()

    return build_merged_colwise_linear(
        in_features=in_features,
        all_out_features=all_out_features,
        bias=bias,
        dtype=dtype,
        device=device,
        quant_config=quant_config,
        is_tp=is_tp,
        out_names=out_names,
        dp_gather=dp_gather,
    )


def build_down_linear(in_features: int,
                      out_features: int,
                      bias: bool,
                      dtype: Optional[torch.dtype] = None,
                      device: Optional[torch.device] = None,
                      is_tp: bool = False,
                      tp_align_size: int = 1,
                      quant_config: Any = None,
                      all_reduce: bool = True,
                      dp_scatter: bool = True) -> nn.Module:
    """Build down linear."""
    if dp_scatter:
        if is_tp:
            is_tp = get_tp_world_rank()[0] > 1
        if not _is_dp_enabled():
            # disable if not dp
            dp_scatter = False
        if not is_tp:
            dp_scatter = False
    elif is_tp:
        is_tp, all_reduce = _get_dp_tp_meta(all_reduce)

    return build_rowwise_linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        dtype=dtype,
        device=device,
        is_tp=is_tp,
        tp_align_size=tp_align_size,
        quant_config=quant_config,
        all_reduce=all_reduce,
        dp_scatter=dp_scatter,
    )
