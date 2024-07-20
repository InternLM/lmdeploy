# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Any, List

import torch
import torch.distributed as dist
from torch import nn

from lmdeploy.pytorch.models.q_modules import QLinear
from lmdeploy.utils import get_logger

from ..backends import LayerType, get_backend

logger = get_logger('lmdeploy')

try:
    from peft.tuners.lora import Linear as LoRALinear
except ImportError:
    logger.debug('load peft.tuners.lora.Linear failed.')

    class LoRALinear:
        pass


try:
    from peft.tuners.lora.awq import AwqLoraLinear
except ImportError:
    logger.debug('load peft.tuners.lora.awq.AwqLoraLinear failed.')

    class AwqLoraLinear:
        pass


try:
    from awq.modules.linear.gemm import WQLinear_GEMM
except ImportError:
    logger.debug('load awq.modules.linear.gemm.WQLinearGEMM failed.')

    class WQLinear_GEMM:
        pass


def _get_world_rank():
    """get current world size and rank."""
    world_size = 1
    rank = 0

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    return world_size, rank


@dataclass
class AdapterInfo:
    r: dict
    lora_A: nn.ModuleDict
    lora_B: nn.ModuleDict
    scaling: dict
    base_slice: slice

    @staticmethod
    def from_lora_linear(mod: nn.Module, base_slice: slice = None):
        return AdapterInfo(
            r=mod.r,
            lora_A=mod.lora_A,
            lora_B=mod.lora_B,
            scaling=mod.scaling,
            base_slice=base_slice,
        )


class SLoRA(nn.Module):

    def __init__(self, adapter_infos: List[AdapterInfo]):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class AwqLinear(nn.Module):

    def __init__(self,
                 mod: nn.Module,
                 adapter_infos: List[AdapterInfo] = None):
        super().__init__()


class W8A8Linear(nn.Module):

    def __init__(self,
                 mod: nn.Module,
                 adapter_infos: List[AdapterInfo] = None):
        super().__init__()


class BaseLinear(nn.Module):

    def __init__(self,
                 mod: nn.Module,
                 adapter_infos: List[AdapterInfo] = None,
                 ctx_mgr: Any = None,
                 all_reduce: bool = False):
        super().__init__()
        layer_backend = get_backend()

        if isinstance(mod, nn.Linear):
            impl_builder = layer_backend.get_layer_impl_builder(
                LayerType.Linear)
        else:
            raise NotImplementedError(f'Unsupported linear type: {type(mod)}')
        self.impl = impl_builder.build(mod, ctx_mgr, all_reduce)

        adapter_infos = adapter_infos if adapter_infos is not None else []
        self.adapter = None
        if len(adapter_infos) > 0:
            self.adapter = SLoRA(adapter_infos)

    def forward(self, x):
        out = self.impl.forward(x)

        if self.adapter is not None:
            out = self.adapter(out)
        return out


def _merge_base_linear(*linears: List[nn.Module]):
    """merge naive linear."""
    weights = [mod.weight for mod in linears]
    bias = [mod.bias for mod in linears]

    in_features = weights[0].size(1)
    dtype = weights[0].dtype
    device = weights[0].device
    for w in weights:
        assert w.size(1) == in_features
        assert w.dtype == dtype
        assert w.device == device
    out_features = sum(w.size(0) for w in weights)

    new_weight = torch.cat(weights, dim=0)
    new_bias = None
    if bias[0] is not None:
        assert all(b is not None for b in bias)
        new_bias = torch.cat(bias)
    has_bias = new_bias is not None
    merged_linear = nn.Linear(in_features,
                              out_features,
                              bias=has_bias,
                              dtype=dtype,
                              device=device)
    state_dict = dict(weight=new_weight)
    if has_bias:
        state_dict['bias'] = new_bias
    merged_linear.load_state_dict(state_dict)
    return merged_linear


def _merge_qlinear(*linears: List[nn.Module]):
    """merge qlinear."""
    weights = [mod.weight for mod in linears]
    scalings = [mod.scaling for mod in linears]
    bias = [mod.bias for mod in linears]

    in_features = weights[0].size(1)
    dtype = weights[0].dtype
    device = weights[0].device
    for w in weights:
        assert w.size(1) == in_features
        assert w.dtype == dtype
        assert w.device == device
    out_features = sum(w.size(0) for w in weights)

    new_weight = torch.cat(weights, dim=0)
    new_scaling = torch.cat(scalings, dim=0)
    new_bias = None
    if bias[0] is not None:
        assert all(b is not None for b in bias)
        new_bias = torch.cat(bias)
    has_bias = new_bias is not None
    merged_linear = QLinear(in_features,
                            out_features,
                            bias=has_bias,
                            dtype=dtype,
                            device=device)
    state_dict = dict(
        weight=new_weight,
        scaling=new_scaling,
    )
    if has_bias:
        state_dict['bias'] = new_bias
    merged_linear.load_state_dict(state_dict)
    return merged_linear


def _merge_awqlinear(*linears: List[nn.Module]):
    """merge awqlinear."""
    qweights = [mod.qweight for mod in linears]
    scales = [mod.scales for mod in linears]
    qzeros = [mod.qzeros for mod in linears]
    bias = [mod.bias for mod in linears]
    w_bits = [mod.w_bit for mod in linears]
    group_sizes = [mod.group_size for mod in linears]

    w_bit = w_bits[0]
    group_size = group_sizes[0]
    assert all(wb == w_bit for wb in w_bits)
    assert all(gs == group_size for gs in group_sizes)
    in_features = qweights[0].size(1)
    device = qweights[0].device
    for w in qweights:
        assert w.size(1) == in_features
        assert w.device == device
    out_features = sum(w.size(0) for w in qweights)

    new_qweight = torch.cat(qweights, dim=0)
    new_scales = torch.cat(scales, dim=0)
    new_qzeros = torch.cat(qzeros, dim=0)
    new_bias = None
    if bias[0] is not None:
        assert all(b is not None for b in bias)
        new_bias = torch.cat(bias)
    has_bias = new_bias is not None
    merged_linear = WQLinear_GEMM(
        w_bit,
        group_size,
        in_features,
        out_features,
        bias=has_bias,
        dev=device,
    )
    state_dict = dict(
        qweight=new_qweight,
        scales=new_scales,
        qzeros=new_qzeros,
    )
    if has_bias:
        state_dict['bias'] = new_bias
    merged_linear.load_state_dict(state_dict)
    return merged_linear


def build_merged_linear(*linears: List[nn.Module],
                        ctx_mgr: Any = None,
                        all_reduce: bool = False,
                        free_origin=False):
    """merge linear."""
    base_layers = []
    out_features = []
    adapter_infos = []
    cum_out_feature = 0
    for mod in linears:
        # get base layers
        base_layer = getattr(mod, 'base_layer', mod)
        base_layers.append(base_layer)

        # get out_feature
        if hasattr(base_layer, 'weight'):
            weight = base_layer.weight
            out_feature = weight.size(0)
        else:
            scales = base_layer.scales
            out_feature = scales.size(1)
        slice_start = cum_out_feature
        cum_out_feature += out_feature

        # get adapter info
        adapter_info = None
        if isinstance(mod, (LoRALinear, AwqLoraLinear)):
            adapter_slice = slice(slice_start, cum_out_feature)
            adapter_info = AdapterInfo.from_lora_linear(mod, adapter_slice)
        out_features.append(out_feature)
        if adapter_info is not None:
            adapter_infos.append(adapter_info)

    # check base layer type
    base_type = type(base_layers[0])
    assert all(isinstance(layer, base_type) for layer in base_layers)

    # merge base layer
    if base_type == nn.Linear:
        base_layer = _merge_base_linear(*base_layers)
    elif base_type == WQLinear_GEMM:
        base_layer = _merge_awqlinear(*base_layers)
    elif base_type == QLinear:
        base_layer = _merge_qlinear(*base_layers)
    else:
        raise NotImplementedError(f'Unknown linear type: {type(mod)}')
    ret = build_linear(base_layer,
                       adapter_infos,
                       ctx_mgr=ctx_mgr,
                       all_reduce=all_reduce)
    if free_origin:
        for mod in linears:
            mod.to('meta')
    return ret


def build_linear(mod: nn.Module,
                 adapter_infos: List[AdapterInfo] = None,
                 ctx_mgr: Any = None,
                 all_reduce: bool = False) -> nn.Module:
    """build linear."""
    if all_reduce:
        world_size, rank = _get_world_rank()
        all_reduce = world_size > 1

    if isinstance(mod, nn.Linear):
        return BaseLinear(mod, adapter_infos, ctx_mgr, all_reduce)
    elif isinstance(mod, WQLinear_GEMM):
        return AwqLinear(mod, adapter_infos)
    elif isinstance(mod, QLinear):
        return W8A8Linear(mod, adapter_infos)
    elif isinstance(mod, LoRALinear):
        base_layer = mod.base_layer
        adapter_info = AdapterInfo.from_lora_linear(mod)
        return build_linear(base_layer, [adapter_info])
    elif isinstance(mod, AwqLoraLinear):
        base_layer = mod.base_layer
        adapter_info = AdapterInfo.from_lora_linear(mod)
        return build_linear(base_layer, [adapter_info])
    else:
        raise NotImplementedError(f'Unknown linear type: {type(mod)}')
