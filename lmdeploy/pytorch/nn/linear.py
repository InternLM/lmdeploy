# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional

import torch
import torch.distributed as dist
from torch import nn

from lmdeploy.pytorch.weight_loader.model_weight_loader import \
    default_weight_loader
from lmdeploy.utils import get_logger

from ..backends import OpType, get_backend
from ..backends.lora import AdapterInfo
from .utils import div_up, get_distribute_size, get_world_rank

logger = get_logger('lmdeploy')

QKV_SPLIT_LAYOUTS = ['default', 'hgd']


def _check_qkv_split_layout(layout: str):
    if layout not in QKV_SPLIT_LAYOUTS:
        raise RuntimeError(f'Expect qkv split layout in {QKV_SPLIT_LAYOUTS}, '
                           f'but get: {layout}')


def _chunk_align(weight: torch.Tensor, chunks: int, dim: int, align: int):
    """chunk aligned."""
    if align == 1:
        return weight.chunk(chunks, dim=dim)
    size = weight.size(dim)
    assert size % align == 0
    aligned_size = size // align
    align_per_chunk = div_up(aligned_size, chunks)
    sections = [align_per_chunk] * (chunks - 1)
    sections += [aligned_size - align_per_chunk * (chunks - 1)]
    sections = [sec * align for sec in sections]
    return weight.split(sections, dim=dim)


class QKVMixin:
    """qkv mixin."""

    def _get_qkv_out_features(self, num_q_heads: int, num_kv_heads: int,
                              head_size: int, head_size_v: int):
        """get io features."""
        all_out_features = (num_q_heads * head_size, num_kv_heads * head_size,
                            num_kv_heads * head_size_v)
        return all_out_features

    def _update_num_heads(self, num_q_heads: int, num_kv_heads: int,
                          replicate_kv: bool):
        """update num heads."""
        world_size, rank = get_world_rank()
        num_q_heads = get_distribute_size(num_q_heads, world_size, rank)
        if not replicate_kv:
            num_kv_heads = get_distribute_size(num_kv_heads, world_size, rank)

        return num_q_heads, num_kv_heads

    def split_qkv(self, x: torch.Tensor):
        """split query, key and value."""
        num_q_heads = self.num_q_heads
        num_kv_heads = self.num_kv_heads
        head_size = self.head_size
        head_size_v = self.head_size_v

        sections = self.all_out_features
        q, k, v = x.split(sections, dim=-1)
        q = q.unflatten(-1, (num_q_heads, head_size))
        k = k.unflatten(-1, (num_kv_heads, head_size))
        v = v.unflatten(-1, (num_kv_heads, head_size_v))
        return q, k, v


class LoRA(nn.Module):
    """LoRA layer."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 ranks: torch.Tensor,
                 scalings: torch.Tensor,
                 lora_a: torch.Tensor,
                 lora_b: torch.Tensor,
                 base_slice: slice,
                 ctx_mgr: Any = None,
                 colwise: bool = True,
                 is_tp: bool = True,
                 lora_b_spliter: Any = None):
        super().__init__()
        self.adapter_info = AdapterInfo(
            in_features=in_features,
            out_features=out_features,
            ranks=ranks,
            scalings=scalings,
            base_slice=base_slice,
        )
        impl_builder = get_backend().get_layer_impl_builder(OpType.LoRA)
        self.impl = impl_builder.build()

        lora_A = nn.Parameter(lora_a, requires_grad=False)
        lora_B = nn.Parameter(lora_b, requires_grad=False)
        self.register_parameter('lora_A', lora_A)
        self.register_parameter('lora_B', lora_B)
        lora_A.weight_loader = self.weight_loader_A
        lora_B.weight_loader = self.weight_loader_B
        self.is_tp = is_tp
        self.ctx_mgr = ctx_mgr
        self.colwise = colwise
        self.lora_b_spliter = lora_b_spliter

    def forward(self, x, base_output=None):
        """forward of loraA@loraB."""
        return self.impl.forward(x,
                                 self.lora_A,
                                 self.lora_B,
                                 base_output,
                                 self.adapter_info,
                                 ctx_mgr=self.ctx_mgr,
                                 colwise=self.colwise,
                                 is_tp=self.is_tp)

    def weight_loader_A(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                        adapter_id: int):
        """weight loader."""
        rank = self.adapter_info.ranks[adapter_id].item()
        r_start = self.adapter_info.rank_offsets[adapter_id].item()
        r_end = r_start + rank
        param_r = param.data[r_start:r_end]

        if self.is_tp and not self.colwise:
            world_size, rank = get_world_rank()
            loaded_weight = loaded_weight.chunk(world_size, dim=1)[rank]

        param_r.copy_(loaded_weight)

    def weight_loader_B(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                        adapter_id: int):
        """weight loader."""
        rank = self.adapter_info.ranks[adapter_id].item()
        r_start = self.adapter_info.rank_offsets[adapter_id].item()
        r_end = r_start + rank
        param_r = param.data[r_start:r_end]

        if self.is_tp and self.colwise:
            world_size, rank = get_world_rank()
            if self.lora_b_spliter is not None:
                loaded_weights = self.lora_b_spliter(loaded_weight)
                new_weights = []
                for w in loaded_weights:
                    w = w.chunk(world_size, dim=0)[rank]
                    new_weights.append(w)
                loaded_weight = torch.cat(new_weights, dim=0)
            else:
                loaded_weight = loaded_weight.chunk(world_size, dim=0)[rank]

        param_r.copy_(loaded_weight.t())


class AwqLinear(nn.Module):
    """w4a16 linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_bit: int,
        group_size: int,
        bias: bool,
        device: Optional[torch.device] = None,
        colwise: bool = True,
        is_tp: bool = False,
        all_reduce: bool = True,
    ):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        dtype = torch.float16
        if is_tp:
            in_features, out_features = self._get_io_features(
                in_features, out_features, w_bit, group_size, colwise)
        qweight, scales, qzeros, bias = self.create_weights(
            in_features, out_features, w_bit, group_size, bias, dtype, device)
        impl_builder = get_backend().get_layer_impl_builder(OpType.LinearW4A16)
        self.impl = impl_builder.build(in_features,
                                       out_features,
                                       w_bit,
                                       group_size,
                                       bias is not None,
                                       dtype=scales.dtype)
        qweight = torch.nn.Parameter(qweight, requires_grad=False)
        qweight.weight_loader = self.weight_loader
        qweight._weight_type = 'qweight'
        scales = torch.nn.Parameter(scales, requires_grad=False)
        scales.weight_loader = self.weight_loader
        scales._weight_type = 'scales'
        qzeros = torch.nn.Parameter(qzeros, requires_grad=False)
        qzeros.weight_loader = self.weight_loader
        qzeros._weight_type = 'qzeros'
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
            bias.weight_loader = self.weight_loader
            bias._weight_type = 'bias'
        self.register_parameter('qweight', qweight)
        self.register_parameter('scales', scales)
        self.register_parameter('qzeros', qzeros)
        self.register_parameter('bias', bias)

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size
        self.elem_per_int = 32 // self.w_bit
        self.lora_adapters = nn.ModuleDict()
        self.is_tp = is_tp
        self.colwise = colwise
        self.all_reduce = all_reduce

    def _get_io_features(self, in_features: int, out_features: int, w_bit: int,
                         group_size: int, colwise: bool):
        """get io features."""
        align = max(32 // w_bit, group_size)
        world_size, rank = get_world_rank()
        if colwise:
            out_features = get_distribute_size(out_features,
                                               world_size,
                                               rank,
                                               align=align)
        else:
            in_features = get_distribute_size(in_features,
                                              world_size,
                                              rank,
                                              align=align)
        return in_features, out_features

    def _weight_loader_tp_colwise(self, param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for colwise linear."""
        if loaded_weight.dim() == 1:
            # bias
            align = max(self.elem_per_int, self.group_size)
            weight = _chunk_align(loaded_weight, world_size, 0, align)[rank]
            return default_weight_loader(param, weight)

        if loaded_weight.size(1) == self.out_features:
            # scaling
            align = max(self.elem_per_int, self.group_size)
            weight = _chunk_align(loaded_weight, world_size, 1, align)[rank]
            return default_weight_loader(param, weight)

        align = max(self.elem_per_int, self.group_size) // self.elem_per_int
        weight = _chunk_align(loaded_weight, world_size, 1, align)[rank]
        return default_weight_loader(param, weight)

    def _weight_loader_tp_rowwise(self, param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for rowwise linear."""
        if loaded_weight.dim() == 1:
            # bias
            if rank == 0:
                loaded_weight = torch.zeros_like(loaded_weight)
            return default_weight_loader(param, loaded_weight)

        if loaded_weight.size(0) == self.in_features:
            # qweight
            align = max(self.elem_per_int, self.group_size)
            weight = _chunk_align(loaded_weight, world_size, 0, align)[rank]
            return default_weight_loader(param, weight)

        align = max(self.elem_per_int, self.group_size) // self.group_size
        weight = _chunk_align(loaded_weight, world_size, 0, align)[rank]
        return default_weight_loader(param, weight)

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor):
        """weight loader."""
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = get_world_rank()
        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank,
                                                  world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank,
                                                  world_size)

    def create_weights(self, in_features: int, out_features: int, w_bit: int,
                       group_size: int, bias: bool, dtype: torch.dtype,
                       device: torch.device):
        """create weights."""
        assert in_features % group_size == 0
        elem_per_int = 32 // w_bit
        assert out_features % elem_per_int == 0

        grouped_in_feats = in_features // group_size
        quant_out_feats = out_features // elem_per_int
        qweight = torch.empty((in_features, quant_out_feats),
                              dtype=torch.int32,
                              device=device)
        scales = torch.empty((grouped_in_feats, out_features),
                             dtype=dtype,
                             device=device)
        qzeros = torch.empty((grouped_in_feats, quant_out_feats),
                             dtype=torch.int32,
                             device=device)
        if bias:
            bias = torch.empty((out_features, ), dtype=dtype, device=device)
        else:
            bias = None
        return qweight, scales, qzeros, bias

    def update_weights(self):
        """update weights."""
        qweight, scales, qzeros, bias = self.impl.update_weights(
            self.qweight, self.scales, self.qzeros, self.bias)
        qweight = torch.nn.Parameter(qweight, requires_grad=False)
        qweight.weight_loader = self.weight_loader
        qweight._weight_type = 'qweight'
        scales = torch.nn.Parameter(scales, requires_grad=False)
        scales.weight_loader = self.weight_loader
        scales._weight_type = 'scales'
        qzeros = torch.nn.Parameter(qzeros, requires_grad=False)
        qzeros.weight_loader = self.weight_loader
        qzeros._weight_type = 'qzeros'
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
            bias.weight_loader = self.weight_loader
            bias._weight_type = 'bias'
        self.register_parameter('qweight', qweight)
        self.register_parameter('scales', scales)
        self.register_parameter('qzeros', qzeros)
        self.register_parameter('bias', bias)

    def forward(self, x):
        """w4a16 forward."""
        all_reduce = False if self.colwise else self.is_tp
        all_reduce = all_reduce and self.all_reduce
        if self.lora_adapters is None:
            return self.impl.forward(x, self.qweight, self.scales, self.qzeros,
                                     self.bias, all_reduce)

        out = self.impl.forward(x, self.qweight, self.scales, self.qzeros,
                                self.bias, False)
        if self.lora_adapters is not None:
            for lora_adapter in self.lora_adapters.values():
                out = lora_adapter(x, out)
        if all_reduce:
            dist.all_reduce(out)
        return out


class MergedAwqLinear(AwqLinear):
    """merged awq linear."""

    def __init__(self,
                 in_features: int,
                 all_out_features: List[int],
                 w_bit: int,
                 group_size: int,
                 bias: bool,
                 replicate: Optional[List[bool]] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 out_names: Optional[List[int]] = None):
        if replicate is None:
            replicate = tuple(False for _ in all_out_features)

        self.split_section_s = all_out_features
        elem_per_int = 32 // w_bit
        self.split_section_wz = [
            size // elem_per_int for size in all_out_features
        ]

        all_out_features = self._update_all_out_features(
            all_out_features, w_bit, group_size, replicate)
        self.all_out_features = all_out_features
        self.replicate = replicate
        if out_names is None:
            out_names = torch.arange(len(self.all_out_features)).tolist()
        assert len(out_names) == len(self.all_out_features)
        self.out_names_map = dict(
            (name, idx) for idx, name in enumerate(out_names))
        out_features = sum(all_out_features)
        super().__init__(in_features,
                         out_features,
                         w_bit,
                         group_size,
                         bias,
                         device,
                         colwise=True,
                         is_tp=is_tp)
        self.qweight.weight_loader = self.weight_loader
        self.qweight.weight_spliter = self.weight_spliter_wz
        self.qweight._weight_type = 'qweight'
        self.scales.weight_loader = self.weight_loader
        self.scales.weight_spliter = self.weight_spliter_s
        self.scales._weight_type = 'scales'
        self.qzeros.weight_loader = self.weight_loader
        self.qzeros.weight_spliter = self.weight_spliter_wz
        self.qzeros._weight_type = 'qzeros'
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
            self.bias.weight_spliter = self.weight_spliter_s
            self.bias._weight_type = 'bias'

    def _get_io_features(self, in_features: int, out_features: int, w_bit: int,
                         group_size: int, colwise: bool):
        """get io features."""
        return in_features, out_features

    def _update_all_out_features(self, all_out_features: List[int], w_bit: int,
                                 group_size: int,
                                 replicate: Optional[List[bool]]):
        """update all out features."""
        world_size, rank = get_world_rank()
        new_all_out_features = []
        align = max(32 // w_bit, group_size)
        for out_feat, rep in zip(all_out_features, replicate):
            if rep:
                new_all_out_features.append(out_feat)
            new_out_feat = get_distribute_size(out_feat, world_size, rank,
                                               align)
            new_all_out_features.append(new_out_feat)
        return new_all_out_features

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, shard_id: Any):
        """weight loader."""
        world_size, rank = get_world_rank()
        shard_idx = self.out_names_map[shard_id]

        if loaded_weight.dim() == 1:
            # bias
            align = max(self.elem_per_int, self.group_size)
            param_w = param.data.split(self.all_out_features, 0)[shard_idx]
            if not self.replicate[shard_idx]:
                weight = _chunk_align(loaded_weight, world_size, 0,
                                      align)[rank]
            param_w.copy_(weight)

        if param._weight_type in ['scales', 'bias']:
            # scales
            align = max(self.elem_per_int, self.group_size)
            param_w = param.data.split(self.all_out_features, -1)[shard_idx]
        else:
            # qweight or qzeros
            align = max(self.elem_per_int,
                        self.group_size) // self.elem_per_int
            quanted_out_feats = [
                feat // self.elem_per_int for feat in self.all_out_features
            ]
            param_w = param.data.split(quanted_out_feats, 1)[shard_idx]

        if not self.replicate[shard_idx]:
            weight = _chunk_align(loaded_weight, world_size, -1, align)[rank]
        param_w.copy_(weight)

    def weight_spliter_wz(self, loaded_weight: torch.Tensor):
        """weight spliter."""
        return loaded_weight.split(self.split_section_wz, dim=1)

    def weight_spliter_s(self, loaded_weight: torch.Tensor):
        """weight spliter."""
        return loaded_weight.split(self.split_section_s, dim=-1)


class QKVAwqLinear(MergedAwqLinear, QKVMixin):
    """qkv awq linear."""

    def __init__(self,
                 in_features: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 head_size: int,
                 head_size_v: int,
                 w_bit: int,
                 group_size: int,
                 replicate_kv: bool = False,
                 bias: bool = False,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True):

        self.qkv_split_section_s = self._get_qkv_out_features(
            num_q_heads, num_kv_heads, head_size, head_size_v)
        elem_per_int = 32 // w_bit
        self.qkv_split_section_wz = [
            size // elem_per_int for size in self.qkv_split_section_s
        ]

        num_q_heads, num_kv_heads = self._update_num_heads(
            num_q_heads, num_kv_heads, replicate_kv)
        all_out_features = self._get_qkv_out_features(num_q_heads,
                                                      num_kv_heads, head_size,
                                                      head_size_v)
        replicate = (False, replicate_kv, replicate_kv)
        out_names = ('q', 'k', 'v')
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size_v
        super().__init__(in_features,
                         all_out_features,
                         w_bit=w_bit,
                         group_size=group_size,
                         bias=bias,
                         replicate=replicate,
                         device=device,
                         is_tp=is_tp,
                         out_names=out_names)

    def _update_all_out_features(self, all_out_features: List[int], w_bit: int,
                                 group_size: int,
                                 replicate: Optional[List[bool]]):
        """update all out features."""
        return all_out_features

    def weight_spliter_wz(self,
                          loaded_weight: torch.Tensor,
                          layout: str = 'default'):
        """weight spliter."""
        _check_qkv_split_layout(layout)
        if layout == 'default':
            return loaded_weight.split(self.qkv_split_section_wz, dim=1)
        elif layout == 'hgd':
            assert self.head_size == self.head_size_v
            heads = [sec // self.head_size for sec in self.qkv_split_section_s]
            kv_heads = heads[-1]
            loaded_weight = loaded_weight.unflatten(
                1, (kv_heads, -1, self.head_size // self.elem_per_int))
            q = loaded_weight[:, :, :-2].flatten(1, 3)
            k = loaded_weight[:, :, -2].flatten(1, 2)
            v = loaded_weight[:, :, -1].flatten(1, 2)
            return q, k, v
        else:
            raise RuntimeError(f'Unsupported layout: {layout}')

    def weight_spliter_s(self,
                         loaded_weight: torch.Tensor,
                         layout: str = 'default'):
        """weight spliter."""
        _check_qkv_split_layout(layout)
        if layout == 'default':
            return loaded_weight.split(self.qkv_split_section_s, dim=-1)
        elif layout == 'hgd':
            assert self.head_size == self.head_size_v
            heads = [sec // self.head_size for sec in self.qkv_split_section_s]
            kv_heads = heads[-1]
            loaded_weight = loaded_weight.unflatten(
                1, (kv_heads, -1, self.head_size))
            q = loaded_weight[:, :, :-2].flatten(1, 3)
            k = loaded_weight[:, :, -2].flatten(1, 2)
            v = loaded_weight[:, :, -1].flatten(1, 2)
            return q, k, v
        else:
            raise RuntimeError(f'Unsupported layout: {layout}')

    def weight_spliter_lora_b(self, loaded_weight: torch.Tensor):
        return loaded_weight.split(self.qkv_split_section_s, dim=0)


class W8A8Linear(nn.Module):
    """w8a8 linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        colwise: bool = True,
        is_tp: bool = False,
        all_reduce: bool = True,
    ):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16
        if is_tp:
            in_features, out_features = self._get_io_features(
                in_features, out_features, colwise)
        impl_builder = get_backend().get_layer_impl_builder(OpType.LinearW8A8)
        self.impl = impl_builder.build(in_features,
                                       out_features,
                                       bias is not None,
                                       dtype=dtype)
        weight, scale, bias = self.create_weights(in_features, out_features,
                                                  bias, dtype, device)
        weight = torch.nn.Parameter(weight, requires_grad=False)
        weight.weight_loader = self.weight_loader
        scale = torch.nn.Parameter(scale, requires_grad=False)
        scale.weight_loader = self.weight_loader
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
            bias.weight_loader = self.weight_loader
        self.register_parameter('weight', weight)
        self.register_parameter('scale', scale)
        self.register_parameter('bias', bias)

        self.in_features = in_features
        self.out_features = out_features
        self.lora_adapters = nn.ModuleDict()
        self.is_tp = is_tp
        self.colwise = colwise
        self.all_reduce = all_reduce

    def _get_io_features(self, in_features: int, out_features: int,
                         colwise: bool):
        """get io features."""
        world_size, rank = get_world_rank()
        if colwise:
            out_features = get_distribute_size(out_features, world_size, rank)
        else:
            in_features = get_distribute_size(in_features, world_size, rank)
        return in_features, out_features

    def _weight_loader_tp_colwise(self, param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for colwise linear."""
        weight = loaded_weight.chunk(world_size, 0)[rank]
        return default_weight_loader(param, weight)

    def _weight_loader_tp_rowwise(self, param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for rowwise linear."""
        if loaded_weight.dim() == 2 and param.dtype == torch.int8:
            weight = loaded_weight.chunk(world_size, 1)[rank]
            return default_weight_loader(param, weight)
        elif loaded_weight.dim() == 2 and loaded_weight.size(1) == 1:
            # scaling
            return default_weight_loader(param, loaded_weight)
        else:
            # bias
            if rank != 0:
                loaded_weight = torch.zeros_like(loaded_weight)
            return default_weight_loader(param, loaded_weight)

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor):
        """weight loader."""
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = get_world_rank()
        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank,
                                                  world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank,
                                                  world_size)

    def create_weights(self, in_features: int, out_features: int, bias: bool,
                       dtype: torch.dtype, device: torch.device):
        """create weights."""
        weight = torch.empty((out_features, in_features),
                             dtype=torch.int8,
                             device=device)
        scale = torch.empty((out_features, 1),
                            dtype=torch.float32,
                            device=device)
        if bias:
            bias = torch.empty((out_features, ), dtype=dtype, device=device)
        else:
            bias = None
        return weight, scale, bias

    def update_weights(self):
        """update weights."""
        weight, scale, bias = self.impl.update_weights(self.weight, self.scale,
                                                       self.bias)
        weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight.weight_loader = self.weight_loader
        scale = torch.nn.Parameter(scale, requires_grad=False)
        self.scale.weight_loader = self.weight_loader
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
            self.bias.weight_loader = self.weight_loader
        self.register_parameter('weight', weight)
        self.register_parameter('scale', scale)
        self.register_parameter('bias', bias)

    def forward(self, x):
        """forward of w8a8 linear."""
        all_reduce = False if self.colwise else self.is_tp
        all_reduce = all_reduce and self.all_reduce
        if len(self.lora_adapters) == 0:
            return self.impl.forward(x, self.weight, self.scale, self.bias,
                                     all_reduce)

        out = self.impl.forward(x, self.weight, self.scale, self.bias, False)
        for lora_adapter in self.lora_adapters.values():
            out = lora_adapter(x, out)
        if all_reduce:
            dist.all_reduce(out)
        return out


class MergedW8A8Linear(W8A8Linear):
    """merged w8a8 linear."""

    def __init__(self,
                 in_features: int,
                 all_out_features: List[int],
                 bias: bool,
                 replicate: Optional[List[bool]] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 out_names: Optional[List[int]] = None):
        if replicate is None:
            replicate = tuple(False for _ in all_out_features)
        self.split_section = all_out_features
        all_out_features = self._update_all_out_features(
            all_out_features, replicate)
        self.all_out_features = all_out_features
        self.replicate = replicate
        if out_names is None:
            out_names = torch.arange(len(self.all_out_features)).tolist()
        assert len(out_names) == len(self.all_out_features)
        self.out_names_map = dict(
            (name, idx) for idx, name in enumerate(out_names))
        out_features = sum(all_out_features)
        super().__init__(in_features,
                         out_features,
                         bias,
                         dtype,
                         device,
                         colwise=True,
                         is_tp=is_tp)
        self.weight.weight_loader = self.weight_loader
        self.scale.weight_loader = self.weight_loader
        self.weight.weight_spliter = self.weight_spliter
        self.scale.weight_spliter = self.weight_spliter
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
            self.bias.weight_spliter = self.weight_spliter

    def _get_io_features(self, in_features: int, out_features: int,
                         colwise: bool):
        """get io features."""
        return in_features, out_features

    def _update_all_out_features(self, all_out_features: List[int],
                                 replicate: Optional[List[bool]]):
        """update all out features."""
        world_size, rank = get_world_rank()
        new_all_out_features = []
        for out_feat, rep in zip(all_out_features, replicate):
            if rep:
                new_all_out_features.append(out_feat)
            new_out_feat = get_distribute_size(out_feat, world_size, rank)
            new_all_out_features.append(new_out_feat)
        return new_all_out_features

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, shard_id: Any):
        """weight loader."""
        world_size, rank = get_world_rank()
        shard_idx = self.out_names_map[shard_id]
        param_w = param.data.split(self.all_out_features, 0)[shard_idx]
        if not self.replicate[shard_idx]:
            loaded_weight = loaded_weight.chunk(world_size, 0)[rank]
        param_w.copy_(loaded_weight)

    def weight_spliter(self, loaded_weight: torch.Tensor):
        """weight spliter."""
        return loaded_weight.split(self.split_section, dim=0)

    def weight_spliter_lora_b(self, loaded_weight: torch.Tensor):
        return loaded_weight.split(self.split_section, dim=0)


class QKVW8A8Linear(MergedW8A8Linear, QKVMixin):
    """qkv w8a8 linear."""

    def __init__(self,
                 in_features: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 head_size: int,
                 head_size_v: int,
                 replicate_kv: bool = False,
                 bias: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True):
        self.qkv_split_section = self._get_qkv_out_features(
            num_q_heads, num_kv_heads, head_size, head_size_v)
        num_q_heads, num_kv_heads = self._update_num_heads(
            num_q_heads, num_kv_heads, replicate_kv)
        all_out_features = self._get_qkv_out_features(num_q_heads,
                                                      num_kv_heads, head_size,
                                                      head_size_v)
        replicate = (False, replicate_kv, replicate_kv)
        out_names = ('q', 'k', 'v')
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size_v
        super().__init__(in_features,
                         all_out_features,
                         bias=bias,
                         replicate=replicate,
                         dtype=dtype,
                         device=device,
                         is_tp=is_tp,
                         out_names=out_names)

    def _update_all_out_features(self, all_out_features: List[int],
                                 replicate: Optional[List[bool]]):
        """update all out features."""
        return all_out_features

    def weight_spliter(self,
                       loaded_weight: torch.Tensor,
                       layout: str = 'default'):
        """weight spliter."""
        _check_qkv_split_layout(layout)
        if layout == 'default':
            return loaded_weight.split(self.qkv_split_section, dim=0)
        elif layout == 'hgd':
            assert self.head_size == self.head_size_v
            heads = [sec // self.head_size for sec in self.qkv_split_section]
            kv_heads = heads[-1]
            loaded_weight = loaded_weight.unflatten(
                0, (kv_heads, -1, self.head_size))
            q = loaded_weight[:, :-2].flatten(0, 2)
            k = loaded_weight[:, -2].flatten(0, 1)
            v = loaded_weight[:, -1].flatten(0, 1)
            return q, k, v
        else:
            raise RuntimeError(f'Unsupported layout: {layout}')

    def weight_spliter_lora_b(self, loaded_weight: torch.Tensor):
        return loaded_weight.split(self.qkv_split_section, dim=0)


class BaseLinear(nn.Module):
    """linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        colwise: bool = True,
        is_tp: bool = False,
        all_reduce: bool = True,
        tp_align_size: int = 1,
    ):
        super().__init__()
        self.tp_align_size = tp_align_size
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16
        if is_tp:
            in_features, out_features = self._get_io_features(
                in_features, out_features, colwise)
        impl_builder = get_backend().get_layer_impl_builder(OpType.Linear)
        self.impl = impl_builder.build(in_features,
                                       out_features,
                                       bias is not None,
                                       dtype=dtype)
        weight, bias = self.create_weights(in_features, out_features, bias,
                                           dtype, device)
        weight = torch.nn.Parameter(weight, requires_grad=False)
        weight.weight_loader = self.weight_loader
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
            bias.weight_loader = self.weight_loader
        self.register_parameter('weight', weight)
        self.register_parameter('bias', bias)

        self.in_features = in_features
        self.out_features = out_features
        self.lora_adapters = nn.ModuleDict()
        self.is_tp = is_tp
        self.colwise = colwise
        self.all_reduce = all_reduce

    def _get_io_features(self, in_features: int, out_features: int,
                         colwise: bool):
        """get io features."""
        world_size, rank = get_world_rank()
        if colwise:
            out_features = get_distribute_size(out_features,
                                               world_size,
                                               rank,
                                               align=self.tp_align_size)
        else:
            in_features = get_distribute_size(in_features,
                                              world_size,
                                              rank,
                                              align=self.tp_align_size)
        return in_features, out_features

    def _weight_loader_tp_colwise(self, param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for colwise linear."""
        weight = _chunk_align(loaded_weight, world_size, 0,
                              self.tp_align_size)[rank]
        return default_weight_loader(param, weight)

    def _weight_loader_tp_rowwise(self, param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for rowwise linear."""
        if loaded_weight.dim() == 2:
            weight = _chunk_align(loaded_weight, world_size, 1,
                                  self.tp_align_size)[rank]
            return default_weight_loader(param, weight)
        else:
            # bias
            if rank != 0:
                loaded_weight = torch.zeros_like(loaded_weight)
            return default_weight_loader(param, loaded_weight)

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor):
        """weight loader."""
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = get_world_rank()
        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank,
                                                  world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank,
                                                  world_size)

    def create_weights(self, in_features: int, out_features: int, bias: bool,
                       dtype: torch.dtype, device: torch.device):
        """create weights."""
        weight = torch.empty((out_features, in_features),
                             dtype=dtype,
                             device=device)
        if bias:
            bias = torch.empty((out_features, ), dtype=dtype, device=device)
        else:
            bias = None
        return weight, bias

    def update_weights(self):
        """update weights."""
        weight, bias = self.impl.update_weights(self.weight, self.bias)
        weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight.weight_loader = self.weight_loader
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
            self.bias.weight_loader = self.weight_loader
        self.register_parameter('weight', weight)
        self.register_parameter('bias', bias)

    def forward(self, x):
        """forward of linear layer."""
        all_reduce = False if self.colwise else self.is_tp
        all_reduce = all_reduce and self.all_reduce
        if len(self.lora_adapters) == 0:
            return self.impl.forward(x, self.weight, self.bias, all_reduce)

        out = self.impl.forward(x, self.weight, self.bias, False)
        for lora_adapter in self.lora_adapters.values():
            out = lora_adapter(x, out)
        if all_reduce:
            dist.all_reduce(out)
        return out


class MergedBaseLinear(BaseLinear):
    """merged base linear."""

    def __init__(self,
                 in_features: int,
                 all_out_features: List[int],
                 bias: bool,
                 replicate: Optional[List[bool]] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 out_names: Optional[List[int]] = None):
        if replicate is None:
            replicate = tuple(False for _ in all_out_features)
        self.split_section = all_out_features
        all_out_features = self._update_all_out_features(
            all_out_features, replicate)
        self.all_out_features = all_out_features
        self.replicate = replicate
        if out_names is None:
            out_names = torch.arange(len(self.all_out_features)).tolist()
        assert len(out_names) == len(self.all_out_features)
        self.out_names_map = dict(
            (name, idx) for idx, name in enumerate(out_names))
        out_features = sum(all_out_features)
        super().__init__(in_features,
                         out_features,
                         bias,
                         dtype,
                         device,
                         colwise=True,
                         is_tp=is_tp)
        self.weight.weight_loader = self.weight_loader
        self.weight.weight_spliter = self.weight_spliter
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
            self.bias.weight_spliter = self.weight_spliter

    def _get_io_features(self, in_features: int, out_features: int,
                         colwise: bool):
        """get io features."""
        return in_features, out_features

    def _update_all_out_features(self, all_out_features: List[int],
                                 replicate: Optional[List[bool]]):
        """update all out features."""
        world_size, rank = get_world_rank()
        new_all_out_features = []
        for out_feat, rep in zip(all_out_features, replicate):
            if rep:
                new_all_out_features.append(out_feat)
            new_out_feat = get_distribute_size(out_feat, world_size, rank)
            new_all_out_features.append(new_out_feat)
        return new_all_out_features

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, shard_id: Any):
        """weight loader."""
        world_size, rank = get_world_rank()
        shard_idx = self.out_names_map[shard_id]
        param_w = param.data.split(self.all_out_features, 0)[shard_idx]
        if not self.replicate[shard_idx]:
            loaded_weight = loaded_weight.chunk(world_size, 0)[rank]
        param_w.copy_(loaded_weight)

    def weight_spliter(self, loaded_weight: torch.Tensor):
        """weight spliter."""
        return loaded_weight.split(self.split_section, dim=0)

    def weight_spliter_lora_b(self, loaded_weight: torch.Tensor):
        return loaded_weight.split(self.split_section, dim=0)


class QKVBaseLinear(MergedBaseLinear, QKVMixin):
    """qkv base linear."""

    def __init__(self,
                 in_features: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 head_size: int,
                 head_size_v: int,
                 replicate_kv: bool = False,
                 bias: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True):
        self.qkv_split_section = self._get_qkv_out_features(
            num_q_heads, num_kv_heads, head_size, head_size_v)
        num_q_heads, num_kv_heads = self._update_num_heads(
            num_q_heads, num_kv_heads, replicate_kv)
        all_out_features = self._get_qkv_out_features(num_q_heads,
                                                      num_kv_heads, head_size,
                                                      head_size_v)
        replicate = (False, replicate_kv, replicate_kv)
        out_names = ('q', 'k', 'v')
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size_v
        super().__init__(in_features,
                         all_out_features,
                         bias=bias,
                         replicate=replicate,
                         dtype=dtype,
                         device=device,
                         is_tp=is_tp,
                         out_names=out_names)

    def _update_all_out_features(self, all_out_features: List[int],
                                 replicate: Optional[List[bool]]):
        """update all out features."""
        return all_out_features

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, shard_id: Any):
        """weight loader."""
        world_size, rank = get_world_rank()
        shard_idx = self.out_names_map[shard_id]
        param_w = param.data.split(self.all_out_features, 0)[shard_idx]
        if not self.replicate[shard_idx]:
            if shard_idx in [0, 1]:
                loaded_weight = _chunk_align(loaded_weight, world_size, 0,
                                             self.head_size)[rank]
            if shard_idx == 2:
                loaded_weight = _chunk_align(loaded_weight, world_size, 0,
                                             self.head_size_v)[rank]
        param_w.copy_(loaded_weight)

    def weight_spliter(self,
                       loaded_weight: torch.Tensor,
                       layout: str = 'default'):
        """weight spliter."""
        _check_qkv_split_layout(layout)
        if layout == 'default':
            return loaded_weight.split(self.qkv_split_section, dim=0)
        elif layout == 'hgd':
            assert self.head_size == self.head_size_v
            heads = [sec // self.head_size for sec in self.qkv_split_section]
            kv_heads = heads[-1]
            loaded_weight = loaded_weight.unflatten(
                0, (kv_heads, -1, self.head_size))
            q = loaded_weight[:, :-2].flatten(0, 2)
            k = loaded_weight[:, -2].flatten(0, 1)
            v = loaded_weight[:, -1].flatten(0, 1)
            return q, k, v
        else:
            raise RuntimeError(f'Unsupported layout: {layout}')

    def weight_spliter_lora_b(self, loaded_weight: torch.Tensor):
        return loaded_weight.split(self.qkv_split_section, dim=0)


def build_linear(in_features: int,
                 out_features: int,
                 bias: bool,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 colwise: bool = True,
                 is_tp: bool = False,
                 quant_config: Any = None,
                 all_reduce: bool = True,
                 tp_align_size: int = 1) -> nn.Module:
    """build linear."""
    if is_tp:
        world_size, _ = get_world_rank()
        is_tp = world_size > 1

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
        )

    quant_method = quant_config['quant_method']
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
        return W8A8Linear(
            in_features,
            out_features,
            bias=bias,
            dtype=dtype,
            device=device,
            colwise=colwise,
            is_tp=is_tp,
            all_reduce=all_reduce,
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
                         quant_config: Any = None) -> nn.Module:
    """build columnwise parallel linear layer."""
    return build_linear(in_features=in_features,
                        out_features=out_features,
                        bias=bias,
                        dtype=dtype,
                        device=device,
                        colwise=True,
                        is_tp=is_tp,
                        quant_config=quant_config,
                        all_reduce=False,
                        tp_align_size=tp_align_size)


def build_rowwise_linear(in_features: int,
                         out_features: int,
                         bias: bool,
                         dtype: Optional[torch.dtype] = None,
                         device: Optional[torch.device] = None,
                         is_tp: bool = False,
                         tp_align_size: int = 1,
                         quant_config: Any = None,
                         all_reduce: bool = True) -> nn.Module:
    """build rowwise parallel linear layer."""
    return build_linear(in_features=in_features,
                        out_features=out_features,
                        bias=bias,
                        dtype=dtype,
                        device=device,
                        colwise=False,
                        is_tp=is_tp,
                        quant_config=quant_config,
                        all_reduce=all_reduce,
                        tp_align_size=tp_align_size)


def build_merged_colwise_linear(
    in_features: int,
    all_out_features: List[int],
    bias: bool,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    quant_config: Any = None,
    is_tp: bool = True,
    out_names: List[Any] = None,
):
    """merge linear."""
    if is_tp:
        world_size, _ = get_world_rank()
        is_tp = world_size > 1

    if quant_config is None:
        return MergedBaseLinear(
            in_features=in_features,
            all_out_features=all_out_features,
            bias=bias,
            dtype=dtype,
            device=device,
            is_tp=is_tp,
            out_names=out_names,
        )

    quant_method = quant_config['quant_method']
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
        return MergedW8A8Linear(
            in_features=in_features,
            all_out_features=all_out_features,
            bias=bias,
            dtype=dtype,
            device=device,
            is_tp=is_tp,
            out_names=out_names,
        )
    else:
        raise RuntimeError(f'Unsupported quant method: {quant_method}')


def build_qkv_proj(in_features: int,
                   num_q_heads: int,
                   num_kv_heads: int,
                   head_size: int,
                   head_size_v: int = None,
                   replicate_kv: bool = False,
                   bias: bool = False,
                   quant_config: Any = None,
                   dtype: Optional[torch.dtype] = None,
                   device: Optional[torch.device] = None,
                   is_tp: bool = True):
    """build qkv proj."""
    if is_tp:
        world_size, _ = get_world_rank()
        is_tp = world_size > 1

    if head_size_v is None:
        head_size_v = head_size

    if quant_config is None:
        return QKVBaseLinear(
            in_features=in_features,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size_v,
            replicate_kv=replicate_kv,
            bias=bias,
            dtype=dtype,
            device=device,
            is_tp=is_tp,
        )

    quant_method = quant_config['quant_method']
    if quant_method == 'awq':
        w_bit = quant_config.get('bits', 4)
        group_size = quant_config.get('group_size', 128)
        return QKVAwqLinear(
            in_features=in_features,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size_v,
            replicate_kv=replicate_kv,
            w_bit=w_bit,
            group_size=group_size,
            bias=bias,
            device=device,
            is_tp=is_tp,
        )
    if quant_method == 'smooth_quant':
        return QKVW8A8Linear(
            in_features=in_features,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size_v,
            replicate_kv=replicate_kv,
            bias=bias,
            dtype=dtype,
            device=device,
            is_tp=is_tp,
        )
    else:
        raise RuntimeError(f'Unsupported quant method: {quant_method}')
