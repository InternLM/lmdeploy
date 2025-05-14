# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional

import torch
from torch import nn

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.distributed import get_dist_manager, get_dp_world_rank, get_tp_world_rank
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader
from lmdeploy.utils import get_logger

from ..backends import OpType, get_backend
from ..backends.lora import AdapterInfo
from .utils import chunk_aligned, div_up, get_distribute_size

logger = get_logger('lmdeploy')

QKV_SPLIT_LAYOUTS = ['default', 'hgd']


def _check_qkv_split_layout(layout: str):
    if layout not in QKV_SPLIT_LAYOUTS:
        raise RuntimeError(f'Expect qkv split layout in {QKV_SPLIT_LAYOUTS}, '
                           f'but get: {layout}')


_chunk_align = chunk_aligned


def _is_dp_enabled():
    """is dp."""
    return get_dp_world_rank()[0] > 1


def _get_dp_gather(is_tp: bool):
    """get dp gather."""
    dp_gather = True
    if not _is_dp_enabled():
        # disable if not dp
        dp_gather = False
    if not is_tp:
        dp_gather = False
    return dp_gather


def _gather_input(x: torch.Tensor, tp_sizes: List[int]):
    """gather input."""
    shape0 = x.shape[:-2]
    shape1 = x.shape[-1:]
    shapes = [shape0 + (size, ) + shape1 for size in tp_sizes]
    new_x = [x.new_empty(shape) for shape in shapes]
    dist.all_gather(new_x, x)
    x = torch.cat(new_x, dim=-2)
    return x


def _reduce_scatter_input(out: torch.Tensor, tp_sizes: List[int]):
    """reduce scatter."""
    _, rank = get_tp_world_rank()
    out = out.transpose(0, -2)
    if not out.is_contiguous():
        out = out.contiguous()
    outs = out.split(tp_sizes, 0)
    out = outs[rank]
    dist.reduce_scatter(out, outs)
    out = out.transpose(0, -2)
    return out


def _get_dp_tp_meta(all_reduce: bool = True):
    """get tp meta."""
    dist_ctx = get_dist_manager().current_context()
    dist_attn_cfg = dist_ctx.dist_config.attn_config
    tp = dist_attn_cfg.tp
    is_tp = tp > 1
    all_reduce = all_reduce if is_tp else False
    return is_tp, all_reduce


class QKVMixin:
    """qkv mixin."""

    def _get_qkv_out_features(self,
                              num_q_heads: int,
                              num_kv_heads: int,
                              head_size: int,
                              head_size_v: int,
                              num_replicate_kv_heads: int = 1):
        """get io features."""
        num_kv_heads_real = num_kv_heads // num_replicate_kv_heads
        all_out_features = (num_q_heads * head_size, num_kv_heads_real * head_size, num_kv_heads_real * head_size_v)
        return all_out_features

    def _update_num_heads(self, num_q_heads: int, num_kv_heads: int):
        """update num heads."""
        is_tp = getattr(self, 'is_tp', False)
        if not is_tp:
            return num_q_heads, num_kv_heads
        world_size, rank = get_tp_world_rank()
        num_q_heads = get_distribute_size(num_q_heads, world_size, rank)
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


def _get_tp_world_rank(is_tp: bool):
    """get tp world size."""
    if is_tp:
        return get_tp_world_rank()
    else:
        return 1, 0


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

    def weight_loader_A(self, param: nn.Parameter, loaded_weight: torch.Tensor, adapter_id: int):
        """weight loader."""
        rank = self.adapter_info.ranks[adapter_id].item()
        r_start = self.adapter_info.rank_offsets[adapter_id].item()
        r_end = r_start + rank
        param_r = param.data[r_start:r_end]

        if self.is_tp and not self.colwise:
            world_size, rank = get_tp_world_rank()
            loaded_weight = loaded_weight.to(param_r.device)
            loaded_weight = loaded_weight.chunk(world_size, dim=1)[rank]

        param_r.copy_(loaded_weight)

    def weight_loader_B(self, param: nn.Parameter, loaded_weight: torch.Tensor, adapter_id: int):
        """weight loader."""
        rank = self.adapter_info.ranks[adapter_id].item()
        r_start = self.adapter_info.rank_offsets[adapter_id].item()
        r_end = r_start + rank
        param_r = param.data[r_start:r_end]

        if self.is_tp and self.colwise:
            world_size, rank = get_tp_world_rank()
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


class BlockedF8Linear(nn.Module):
    """blocked f8 linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        colwise: bool = True,
        is_tp: bool = False,
        all_reduce: bool = True,
        dp_gather: bool = False,
        dp_scatter: bool = False,
    ):
        super().__init__()
        self.is_tp = is_tp
        self.dp_gather = dp_gather
        self.dp_scatter = dp_scatter
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16
        if is_tp:
            in_features, out_features = self._get_io_features(in_features, out_features, colwise)
        impl_builder = get_backend().get_layer_impl_builder(OpType.LinearBlockedF8)
        self.impl = impl_builder.build(in_features, out_features, block_size=128, bias=bias is not None, dtype=dtype)
        self.block_size = 128
        self.fp8_dtype = fp8_dtype
        weight, weight_scale_inv, bias = self.create_weights(in_features, out_features, bias, dtype, device)
        weight = torch.nn.Parameter(weight, requires_grad=False)
        weight.weight_loader = self.weight_loader
        weight_scale_inv = torch.nn.Parameter(weight_scale_inv, requires_grad=False)
        weight_scale_inv.weight_loader = self.weight_loader
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
            bias.weight_loader = self.weight_loader
        self.register_parameter('weight', weight)
        self.register_parameter('weight_scale_inv', weight_scale_inv)
        self.register_parameter('bias', bias)

        self.in_features = in_features
        self.out_features = out_features
        self.lora_adapters = nn.ModuleDict()
        self.is_tp = is_tp
        self.colwise = colwise
        self.all_reduce = all_reduce

    def _get_io_features(self, in_features: int, out_features: int, colwise: bool):
        """get io features."""
        world_size, rank = get_tp_world_rank()
        if colwise:
            out_features = get_distribute_size(out_features, world_size, rank)
        else:
            in_features = get_distribute_size(in_features, world_size, rank)
        return in_features, out_features

    def _weight_loader_tp_colwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for colwise linear."""
        weight = loaded_weight.chunk(world_size, 0)[rank]
        return default_weight_loader(param, weight)

    def _weight_loader_tp_rowwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for rowwise linear."""
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.to(param.device)
            weight = loaded_weight.chunk(world_size, 1)[rank]
            return default_weight_loader(param, weight)
        else:
            # bias
            if rank != 0:
                loaded_weight = torch.zeros_like(loaded_weight)
            return default_weight_loader(param, loaded_weight)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """weight loader."""
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = _get_tp_world_rank(self.is_tp)
        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank, world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank, world_size)

    def create_weights(self, in_features: int, out_features: int, bias: bool, dtype: torch.dtype, device: torch.device):
        """create weights."""
        weight = torch.empty((out_features, in_features), dtype=self.fp8_dtype, device=device)
        weight_scale_inv = torch.empty((div_up(out_features, self.block_size), div_up(in_features, self.block_size)),
                                       dtype=torch.float32,
                                       device=device)
        if bias:
            bias = torch.empty((out_features, ), dtype=dtype, device=device)
        else:
            bias = None
        return weight, weight_scale_inv, bias

    def update_weights(self):
        """update weights."""
        weight, weight_scale_inv, bias = self.impl.update_weights(self.weight, self.weight_scale_inv, self.bias)
        weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight.weight_loader = self.weight_loader
        weight_scale_inv = torch.nn.Parameter(weight_scale_inv, requires_grad=False)
        self.weight_scale_inv.weight_loader = self.weight_loader
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
            self.bias.weight_loader = self.weight_loader
        self.register_parameter('weight', weight)
        self.register_parameter('weight_scale_inv', weight_scale_inv)
        self.register_parameter('bias', bias)

    def forward(self, x):
        """forward of blocked fp8 linear."""
        if self.dp_gather or self.dp_scatter:
            step_ctx = get_step_ctx_manager().current_context()
            dp_meta = step_ctx.dp_meta
            tp_sizes = dp_meta.tp_sizes

        if self.dp_gather:
            x = _gather_input(x, tp_sizes)

        all_reduce = False if self.colwise else self.is_tp
        all_reduce = all_reduce and self.all_reduce
        if len(self.lora_adapters) == 0:
            if self.dp_scatter:
                _, rank = get_tp_world_rank()
                return self.impl.forward(x, self.weight, self.weight_scale_inv, self.bias, all_reduce, rank, tp_sizes)
            else:
                return self.impl.forward(x, self.weight, self.weight_scale_inv, self.bias, all_reduce)

        out = self.impl.forward(x, self.weight, self.weight_scale_inv, self.bias, False)
        for lora_adapter in self.lora_adapters.values():
            out = lora_adapter(x, out)
        if all_reduce:
            if self.dp_scatter:
                out = _reduce_scatter_input(out, tp_sizes)
            else:
                dist.all_reduce(out)
        return out


class MergedBlockedF8Linear(BlockedF8Linear):
    """merged blocked fp8 linear."""

    def __init__(self,
                 in_features: int,
                 all_out_features: List[int],
                 bias: bool,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 replicate: Optional[List[bool]] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 out_names: Optional[List[int]] = None,
                 dp_gather: bool = False):
        if replicate is None:
            replicate = tuple(False for _ in all_out_features)
        self.block_size = 128
        self.split_section = all_out_features
        self.is_tp = is_tp
        self.scale_split_section = [section // self.block_size for section in self.split_section]
        all_out_features = self._update_all_out_features(all_out_features, replicate)
        self.all_out_features = all_out_features
        self.replicate = replicate
        if out_names is None:
            out_names = torch.arange(len(self.all_out_features)).tolist()
        assert len(out_names) == len(self.all_out_features)
        self.out_names_map = dict((name, idx) for idx, name in enumerate(out_names))
        out_features = sum(all_out_features)
        super().__init__(in_features,
                         out_features,
                         bias,
                         dtype,
                         device,
                         fp8_dtype=fp8_dtype,
                         colwise=True,
                         is_tp=is_tp,
                         dp_gather=dp_gather)
        self.weight.weight_loader = self.weight_loader
        self.weight._weight_type = 'qweight'
        self.weight_scale_inv.weight_loader = self.weight_loader
        self.weight_scale_inv._weight_type = 'scales'
        self.weight.weight_spliter = self.weight_spliter
        self.weight_scale_inv.weight_spliter = self.weight_spliter
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
            self.bias.weight_spliter = self.weight_spliter
            self.bias._weight_type = 'bias'

    def _get_io_features(self, in_features: int, out_features: int, colwise: bool):
        """get io features."""
        return in_features, out_features

    def _update_all_out_features(self, all_out_features: List[int], replicate: Optional[List[bool]]):
        """update all out features."""
        world_size, rank = _get_tp_world_rank(self.is_tp)
        new_all_out_features = []
        for out_feat, rep in zip(all_out_features, replicate):
            if rep:
                new_all_out_features.append(out_feat)
            new_out_feat = get_distribute_size(out_feat, world_size, rank)
            new_all_out_features.append(new_out_feat)
        return new_all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """weight loader."""
        world_size, rank = _get_tp_world_rank(self.is_tp)
        shard_idx = self.out_names_map[shard_id]
        if loaded_weight.dim() == 2 and loaded_weight.dtype != self.fp8_dtype:
            loaded_weight = loaded_weight.to(torch.float32)
            all_out_features = [feats // self.block_size for feats in self.all_out_features]
            param_w = param.data.split(all_out_features, 0)[shard_idx]
        else:
            param_w = param.data.split(self.all_out_features, 0)[shard_idx]
        if not self.replicate[shard_idx]:
            loaded_weight = loaded_weight.chunk(world_size, 0)[rank]
        param_w.copy_(loaded_weight)

    def weight_spliter(self, loaded_weight: torch.Tensor):
        """weight spliter."""
        if loaded_weight.dim() == 2 and loaded_weight.dtype != self.fp8_dtype:
            return loaded_weight.split(self.scale_split_section, dim=0)
        return loaded_weight.split(self.split_section, dim=0)

    def weight_spliter_lora_b(self, loaded_weight: torch.Tensor):
        return loaded_weight.split(self.split_section, dim=0)


class QKVBlockedF8Linear(MergedBlockedF8Linear, QKVMixin):
    """qkv blockedf8 linear."""

    def __init__(self,
                 in_features: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 head_size: int,
                 head_size_v: int,
                 bias: bool = False,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 dp_gather: bool = False,
                 num_replicate_kv_heads: int = 1):
        self.is_tp = is_tp
        self.block_size = 128
        self.qkv_split_section = self._get_qkv_out_features(num_q_heads, num_kv_heads, head_size, head_size_v,
                                                            num_replicate_kv_heads)

        num_q_heads, num_kv_heads = self._update_num_heads(num_q_heads, num_kv_heads)
        all_out_features = self._get_qkv_out_features(num_q_heads, num_kv_heads, head_size, head_size_v)
        out_names = ('q', 'k', 'v')
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size_v
        self.num_replicate_kv_heads = num_replicate_kv_heads

        super().__init__(in_features,
                         all_out_features,
                         dtype=dtype,
                         fp8_dtype=fp8_dtype,
                         bias=bias,
                         device=device,
                         is_tp=is_tp,
                         out_names=out_names,
                         dp_gather=dp_gather)

    def _update_all_out_features(self, all_out_features: List[int], replicate: Optional[List[bool]]):
        """update all out features."""
        return all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """weight loader."""
        _, rank = _get_tp_world_rank(self.is_tp)
        shard_idx = self.out_names_map[shard_id]

        num_head = self.num_q_heads if shard_id == 'q' \
            else self.num_kv_heads
        head_dim = self.head_size if shard_id in ['q', 'k'] \
            else self.head_size_v
        # update to duplicate k/v for tp_size > num_kv_heads
        rank_idx = rank if shard_id == 'q' \
            else rank // self.num_replicate_kv_heads
        sec_len = num_head * head_dim
        all_out_features = self.all_out_features
        if param._weight_type == 'scales':
            loaded_weight = loaded_weight.to(torch.float32)
            all_out_features = [sec // self.block_size for sec in all_out_features]
            sec_len = sec_len // self.block_size

        sec_start = rank_idx * sec_len

        loaded_weight = loaded_weight.narrow(dim=0, start=sec_start, length=sec_len)
        param_w = param.data.split(all_out_features, 0)[shard_idx]
        param_w.copy_(loaded_weight)

    def weight_spliter(self, loaded_weight: torch.Tensor, layout: str = 'default'):
        """weight spliter."""
        _check_qkv_split_layout(layout)
        assert layout == 'default'
        qkv_split_section = self.qkv_split_section
        if loaded_weight.dim() == 2 and loaded_weight.dtype != self.fp8_dtype:
            qkv_split_section = [sec // self.block_size for sec in qkv_split_section]
        return loaded_weight.split(qkv_split_section, dim=0)


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
        self.is_tp = is_tp
        if device is None:
            device = torch.device('cpu')
        dtype = torch.float16
        if is_tp:
            in_features, out_features = self._get_io_features(in_features, out_features, w_bit, group_size, colwise)
        qweight, scales, qzeros, bias = self.create_weights(in_features, out_features, w_bit, group_size, bias, dtype,
                                                            device)
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
        self.elem_per_int = 32 // w_bit
        self.lora_adapters = nn.ModuleDict()
        self.colwise = colwise
        self.all_reduce = all_reduce

    def _get_io_features(self, in_features: int, out_features: int, w_bit: int, group_size: int, colwise: bool):
        """get io features."""
        align = max(32 // w_bit, group_size)
        world_size, rank = _get_tp_world_rank(self.is_tp)
        if colwise:
            out_features = get_distribute_size(out_features, world_size, rank, align=align)
        else:
            in_features = get_distribute_size(in_features, world_size, rank, align=align)
        return in_features, out_features

    def _weight_loader_tp_colwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
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

    def _weight_loader_tp_rowwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
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

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """weight loader."""
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = _get_tp_world_rank(self.is_tp)
        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank, world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank, world_size)

    def create_weights(self, in_features: int, out_features: int, w_bit: int, group_size: int, bias: bool,
                       dtype: torch.dtype, device: torch.device):
        """create weights."""
        assert in_features % group_size == 0
        elem_per_int = 32 // w_bit
        assert out_features % elem_per_int == 0

        grouped_in_feats = in_features // group_size
        quant_out_feats = out_features // elem_per_int
        qweight = torch.empty((in_features, quant_out_feats), dtype=torch.int32, device=device)
        scales = torch.empty((grouped_in_feats, out_features), dtype=dtype, device=device)
        qzeros = torch.empty((grouped_in_feats, quant_out_feats), dtype=torch.int32, device=device)
        if bias:
            bias = torch.empty((out_features, ), dtype=dtype, device=device)
        else:
            bias = None
        return qweight, scales, qzeros, bias

    def update_weights(self):
        """update weights."""
        qweight, scales, qzeros, bias = self.impl.update_weights(self.qweight, self.scales, self.qzeros, self.bias)
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
            return self.impl.forward(x, self.qweight, self.scales, self.qzeros, self.bias, all_reduce)

        out = self.impl.forward(x, self.qweight, self.scales, self.qzeros, self.bias, False)
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
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 out_names: Optional[List[int]] = None):

        self.split_section_s = all_out_features
        self.is_tp = is_tp
        elem_per_int = 32 // w_bit
        self.split_section_wz = [size // elem_per_int for size in all_out_features]

        all_out_features = self._update_all_out_features(all_out_features, w_bit, group_size)
        self.all_out_features = all_out_features
        if out_names is None:
            out_names = torch.arange(len(self.all_out_features)).tolist()
        assert len(out_names) == len(self.all_out_features)
        self.out_names_map = dict((name, idx) for idx, name in enumerate(out_names))
        out_features = sum(all_out_features)
        super().__init__(in_features, out_features, w_bit, group_size, bias, device, colwise=True, is_tp=is_tp)
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

    def _get_io_features(self, in_features: int, out_features: int, w_bit: int, group_size: int, colwise: bool):
        """get io features."""
        return in_features, out_features

    def _update_all_out_features(self, all_out_features: List[int], w_bit: int, group_size: int):
        """update all out features."""
        world_size, rank = _get_tp_world_rank(self.is_tp)
        new_all_out_features = []
        align = max(32 // w_bit, group_size)
        for out_feat in all_out_features:
            new_out_feat = get_distribute_size(out_feat, world_size, rank, align)
            new_all_out_features.append(new_out_feat)
        return new_all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """weight loader."""
        world_size, rank = _get_tp_world_rank(self.is_tp)
        shard_idx = self.out_names_map[shard_id]
        if loaded_weight.dim() == 1:
            # bias
            align = max(self.elem_per_int, self.group_size)
            param_w = param.data.split(self.all_out_features, 0)[shard_idx]
            weight = _chunk_align(loaded_weight, world_size, 0, align)[rank]
            param_w.copy_(weight)

        if param._weight_type in ['scales', 'bias']:
            # scales
            align = max(self.elem_per_int, self.group_size)
            param_w = param.data.split(self.all_out_features, -1)[shard_idx]
        else:
            # qweight or qzeros
            align = max(self.elem_per_int, self.group_size) // self.elem_per_int
            quanted_out_feats = [feat // self.elem_per_int for feat in self.all_out_features]
            param_w = param.data.split(quanted_out_feats, 1)[shard_idx]

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
                 bias: bool = False,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 num_replicate_kv_heads: int = 1):
        self.is_tp = is_tp
        self.qkv_split_section_s = self._get_qkv_out_features(num_q_heads, num_kv_heads, head_size, head_size_v,
                                                              num_replicate_kv_heads)
        elem_per_int = 32 // w_bit
        self.qkv_split_section_wz = [size // elem_per_int for size in self.qkv_split_section_s]

        num_q_heads, num_kv_heads = self._update_num_heads(num_q_heads, num_kv_heads)
        all_out_features = self._get_qkv_out_features(num_q_heads, num_kv_heads, head_size, head_size_v)
        out_names = ('q', 'k', 'v')
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size_v
        self.num_replicate_kv_heads = num_replicate_kv_heads

        super().__init__(in_features,
                         all_out_features,
                         w_bit=w_bit,
                         group_size=group_size,
                         bias=bias,
                         device=device,
                         is_tp=is_tp,
                         out_names=out_names)

    def _update_all_out_features(self, all_out_features: List[int], w_bit: int, group_size: int):
        """update all out features."""
        return all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """weight loader."""
        world_size, rank = _get_tp_world_rank(self.is_tp)
        chunk_size, chunk_idx = world_size, rank
        shard_idx = self.out_names_map[shard_id]

        if self.num_replicate_kv_heads > 1 and shard_id in ['k', 'v']:
            # update to duplicate k/v for tp_size > num_kv_heads
            chunk_size = world_size // self.num_replicate_kv_heads
            chunk_idx = rank // self.num_replicate_kv_heads

        if loaded_weight.dim() == 1:
            # bias
            align = max(self.elem_per_int, self.group_size)
            param_w = param.data.split(self.all_out_features, 0)[shard_idx]
            weight = _chunk_align(loaded_weight, chunk_size, 0, align)[chunk_idx]
            param_w.copy_(weight)
            return

        if param._weight_type in ['scales', 'bias']:
            # scales
            align = max(self.elem_per_int, self.group_size)
            param_w = param.data.split(self.all_out_features, -1)[shard_idx]
        else:
            # qweight or qzeros
            align = max(self.elem_per_int, self.group_size) // self.elem_per_int
            quanted_out_feats = [feat // self.elem_per_int for feat in self.all_out_features]
            param_w = param.data.split(quanted_out_feats, 1)[shard_idx]

        weight = _chunk_align(loaded_weight, chunk_size, -1, align)[chunk_idx]
        param_w.copy_(weight)

    def weight_spliter_wz(self, loaded_weight: torch.Tensor, layout: str = 'default'):
        """weight spliter."""
        _check_qkv_split_layout(layout)
        if layout == 'default':
            return loaded_weight.split(self.qkv_split_section_wz, dim=1)
        elif layout == 'hgd':
            assert self.head_size == self.head_size_v
            heads = [sec // self.head_size for sec in self.qkv_split_section_s]
            kv_heads = heads[-1]
            loaded_weight = loaded_weight.unflatten(1, (kv_heads, -1, self.head_size // self.elem_per_int))
            q = loaded_weight[:, :, :-2].flatten(1, 3)
            k = loaded_weight[:, :, -2].flatten(1, 2)
            v = loaded_weight[:, :, -1].flatten(1, 2)
            return q, k, v
        else:
            raise RuntimeError(f'Unsupported layout: {layout}')

    def weight_spliter_s(self, loaded_weight: torch.Tensor, layout: str = 'default'):
        """weight spliter."""
        _check_qkv_split_layout(layout)
        if layout == 'default':
            return loaded_weight.split(self.qkv_split_section_s, dim=-1)
        elif layout == 'hgd':
            assert self.head_size == self.head_size_v
            heads = [sec // self.head_size for sec in self.qkv_split_section_s]
            kv_heads = heads[-1]
            loaded_weight = loaded_weight.unflatten(1, (kv_heads, -1, self.head_size))
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

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 colwise: bool = True,
                 is_tp: bool = False,
                 all_reduce: bool = True,
                 quant_dtype: Optional[torch.dtype] = torch.int8):
        super().__init__()
        self.is_tp = is_tp
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16
        if is_tp:
            in_features, out_features = self._get_io_features(in_features, out_features, colwise)
        impl_builder = get_backend().get_layer_impl_builder(OpType.LinearW8A8)
        self.quant_dtype = quant_dtype
        self.impl = impl_builder.build(in_features,
                                       out_features,
                                       bias is not None,
                                       dtype=dtype,
                                       quant_dtype=quant_dtype)
        weight, scale, bias = self.create_weights(in_features, out_features, bias, dtype, device)
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

    def _get_io_features(self, in_features: int, out_features: int, colwise: bool):
        """get io features."""
        world_size, rank = _get_tp_world_rank(self.is_tp)
        if colwise:
            out_features = get_distribute_size(out_features, world_size, rank)
        else:
            in_features = get_distribute_size(in_features, world_size, rank)
        return in_features, out_features

    def _weight_loader_tp_colwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for colwise linear."""
        weight = loaded_weight.chunk(world_size, 0)[rank]
        return default_weight_loader(param, weight)

    def _weight_loader_tp_rowwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for rowwise linear."""
        if loaded_weight.dim() == 2 and param.dtype in (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2):
            loaded_weight = loaded_weight.to(param.device)
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

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """weight loader."""
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = _get_tp_world_rank(self.is_tp)
        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank, world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank, world_size)

    def create_weights(self, in_features: int, out_features: int, bias: bool, dtype: torch.dtype, device: torch.device):
        """create weights."""
        weight = torch.empty((out_features, in_features), dtype=self.quant_dtype, device=device)
        scale = torch.empty((out_features, 1), dtype=torch.float32, device=device)
        if bias:
            bias = torch.empty((out_features, ), dtype=dtype, device=device)
        else:
            bias = None
        return weight, scale, bias

    def update_weights(self):
        """update weights."""
        weight, scale, bias = self.impl.update_weights(self.weight, self.scale, self.bias)
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
            return self.impl.forward(x, self.weight, self.scale, self.bias, all_reduce)

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
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 out_names: Optional[List[int]] = None,
                 quant_dtype: torch.dtype = torch.int8):
        self.split_section = all_out_features
        self.is_tp = is_tp
        all_out_features = self._update_all_out_features(all_out_features)
        self.all_out_features = all_out_features
        if out_names is None:
            out_names = torch.arange(len(self.all_out_features)).tolist()
        assert len(out_names) == len(self.all_out_features)
        self.out_names_map = dict((name, idx) for idx, name in enumerate(out_names))
        out_features = sum(all_out_features)
        super().__init__(in_features,
                         out_features,
                         bias,
                         dtype,
                         device,
                         colwise=True,
                         is_tp=is_tp,
                         quant_dtype=quant_dtype)
        self.weight.weight_loader = self.weight_loader
        self.scale.weight_loader = self.weight_loader
        self.weight.weight_spliter = self.weight_spliter
        self.scale.weight_spliter = self.weight_spliter
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
            self.bias.weight_spliter = self.weight_spliter

    def _get_io_features(self, in_features: int, out_features: int, colwise: bool):
        """get io features."""
        return in_features, out_features

    def _update_all_out_features(self, all_out_features: List[int]):
        """update all out features."""
        world_size, rank = _get_tp_world_rank(self.is_tp)
        new_all_out_features = []
        for out_feat in all_out_features:
            new_out_feat = get_distribute_size(out_feat, world_size, rank)
            new_all_out_features.append(new_out_feat)
        return new_all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """weight loader."""
        world_size, rank = _get_tp_world_rank(self.is_tp)
        shard_idx = self.out_names_map[shard_id]
        param_w = param.data.split(self.all_out_features, 0)[shard_idx]
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
                 bias: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 num_replicate_kv_heads: int = 1,
                 quant_dtype: torch.dtype = torch.int8):

        self.is_tp = is_tp
        self.qkv_split_section = self._get_qkv_out_features(num_q_heads, num_kv_heads, head_size, head_size_v,
                                                            num_replicate_kv_heads)
        num_q_heads, num_kv_heads = self._update_num_heads(num_q_heads, num_kv_heads)
        all_out_features = self._get_qkv_out_features(num_q_heads, num_kv_heads, head_size, head_size_v)
        out_names = ('q', 'k', 'v')
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size_v
        self.num_replicate_kv_heads = num_replicate_kv_heads
        super().__init__(in_features,
                         all_out_features,
                         bias=bias,
                         dtype=dtype,
                         device=device,
                         is_tp=is_tp,
                         out_names=out_names,
                         quant_dtype=quant_dtype)

    def _update_all_out_features(self, all_out_features: List[int]):
        """update all out features."""
        return all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """weight loader."""
        _, rank = _get_tp_world_rank(self.is_tp)
        shard_idx = self.out_names_map[shard_id]
        param_w = param.data.split(self.all_out_features, 0)[shard_idx]
        num_head = self.num_q_heads if shard_id == 'q' \
            else self.num_kv_heads
        head_dim = self.head_size if shard_id in ['q', 'k'] \
            else self.head_size_v
        # update to duplicate k/v for tp_size > num_kv_heads
        rank_idx = rank if shard_id == 'q' \
            else rank // self.num_replicate_kv_heads
        sec_start = rank_idx * num_head * head_dim
        sec_len = num_head * head_dim
        loaded_weight = loaded_weight.narrow(dim=0, start=sec_start, length=sec_len)
        param_w.copy_(loaded_weight)

    def weight_spliter(self, loaded_weight: torch.Tensor, layout: str = 'default'):
        """weight spliter."""
        _check_qkv_split_layout(layout)
        if layout == 'default':
            return loaded_weight.split(self.qkv_split_section, dim=0)
        elif layout == 'hgd':
            assert self.head_size == self.head_size_v
            heads = [sec // self.head_size for sec in self.qkv_split_section]
            kv_heads = heads[-1]
            loaded_weight = loaded_weight.unflatten(0, (kv_heads, -1, self.head_size))
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
        dp_gather: bool = False,
        dp_scatter: bool = False,
    ):
        super().__init__()
        self.tp_align_size = tp_align_size
        self.is_tp = is_tp
        self.dp_gather = dp_gather
        self.dp_scatter = dp_scatter
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float16
        if is_tp:
            in_features, out_features = self._get_io_features(in_features, out_features, colwise)
        impl_builder = get_backend().get_layer_impl_builder(OpType.Linear)
        self.impl = impl_builder.build(in_features, out_features, bias is not None, dtype=dtype)
        weight, bias = self.create_weights(in_features, out_features, bias, dtype, device)
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
        self.colwise = colwise
        self.all_reduce = all_reduce

    def _get_io_features(self, in_features: int, out_features: int, colwise: bool):
        """get io features."""
        world_size, rank = _get_tp_world_rank(self.is_tp)
        if colwise:
            out_features = get_distribute_size(out_features, world_size, rank, align=self.tp_align_size)
        else:
            in_features = get_distribute_size(in_features, world_size, rank, align=self.tp_align_size)
        return in_features, out_features

    def _weight_loader_tp_colwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for colwise linear."""
        weight = _chunk_align(loaded_weight, world_size, 0, self.tp_align_size)[rank]
        return default_weight_loader(param, weight)

    def _weight_loader_tp_rowwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for rowwise linear."""
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.to(param.device)
            weight = _chunk_align(loaded_weight, world_size, 1, self.tp_align_size)[rank]
            return default_weight_loader(param, weight)
        else:
            # bias
            if rank != 0:
                loaded_weight = torch.zeros_like(loaded_weight)
            return default_weight_loader(param, loaded_weight)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """weight loader."""
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = _get_tp_world_rank(self.is_tp)
        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank, world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank, world_size)

    def create_weights(self, in_features: int, out_features: int, bias: bool, dtype: torch.dtype, device: torch.device):
        """create weights."""
        weight = torch.empty((out_features, in_features), dtype=dtype, device=device)
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
        if self.dp_gather or self.dp_scatter:
            step_ctx = get_step_ctx_manager().current_context()
            dp_meta = step_ctx.dp_meta
            tp_sizes = dp_meta.tp_sizes

        if self.dp_gather:
            x = _gather_input(x, tp_sizes)

        all_reduce = False if self.colwise else self.is_tp
        all_reduce = all_reduce and self.all_reduce
        if len(self.lora_adapters) == 0:
            if self.dp_scatter:
                _, rank = get_tp_world_rank()
                return self.impl.forward(x, self.weight, self.bias, all_reduce, rank, tp_sizes)
            else:
                return self.impl.forward(x, self.weight, self.bias, all_reduce)

        out = self.impl.forward(x, self.weight, self.bias, False)
        for lora_adapter in self.lora_adapters.values():
            out = lora_adapter(x, out)
        if all_reduce:
            if self.dp_scatter:
                out = _reduce_scatter_input(out, tp_sizes)
            else:
                dist.all_reduce(out)
        return out


class MergedBaseLinear(BaseLinear):
    """merged base linear."""

    def __init__(self,
                 in_features: int,
                 all_out_features: List[int],
                 bias: bool,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 out_names: Optional[List[int]] = None,
                 dp_gather: bool = False):
        self.split_section = all_out_features
        self.is_tp = is_tp
        all_out_features = self._update_all_out_features(all_out_features)
        self.all_out_features = all_out_features
        if out_names is None:
            out_names = torch.arange(len(self.all_out_features)).tolist()
        assert len(out_names) == len(self.all_out_features)
        self.out_names_map = dict((name, idx) for idx, name in enumerate(out_names))
        out_features = sum(all_out_features)
        super().__init__(in_features, out_features, bias, dtype, device, colwise=True, is_tp=is_tp, dp_gather=dp_gather)
        self.weight.weight_loader = self.weight_loader
        self.weight.weight_spliter = self.weight_spliter
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
            self.bias.weight_spliter = self.weight_spliter

    def _get_io_features(self, in_features: int, out_features: int, colwise: bool):
        """get io features."""
        return in_features, out_features

    def _update_all_out_features(self, all_out_features: List[int]):
        """update all out features."""
        world_size, rank = _get_tp_world_rank(self.is_tp)
        new_all_out_features = []
        for out_feat in all_out_features:
            new_out_feat = get_distribute_size(out_feat, world_size, rank)
            new_all_out_features.append(new_out_feat)
        return new_all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """weight loader."""
        world_size, rank = _get_tp_world_rank(self.is_tp)
        shard_idx = self.out_names_map[shard_id]
        param_w = param.data.split(self.all_out_features, 0)[shard_idx]
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
                 bias: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 num_replicate_kv_heads: int = 1):
        self.is_tp = is_tp

        self.qkv_split_section = self._get_qkv_out_features(num_q_heads, num_kv_heads, head_size, head_size_v,
                                                            num_replicate_kv_heads)
        num_q_heads, num_kv_heads = self._update_num_heads(num_q_heads, num_kv_heads)
        all_out_features = self._get_qkv_out_features(num_q_heads, num_kv_heads, head_size, head_size_v)
        out_names = ('q', 'k', 'v')
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size_v
        self.num_replicate_kv_heads = num_replicate_kv_heads

        super().__init__(in_features,
                         all_out_features,
                         bias=bias,
                         dtype=dtype,
                         device=device,
                         is_tp=is_tp,
                         out_names=out_names)

    def _update_all_out_features(self, all_out_features: List[int]):
        """update all out features."""
        return all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """weight loader."""
        world_size, rank = _get_tp_world_rank(self.is_tp)
        chunk_size, chunk_idx = world_size, rank
        shard_idx = self.out_names_map[shard_id]
        param_w = param.data.split(self.all_out_features, 0)[shard_idx]

        if self.num_replicate_kv_heads > 1 and shard_id in ['k', 'v']:
            # update to duplicate k/v for tp_size > num_kv_heads
            chunk_size = world_size // self.num_replicate_kv_heads
            chunk_idx = rank // self.num_replicate_kv_heads
        if shard_idx in [0, 1]:
            loaded_weight = _chunk_align(loaded_weight, chunk_size, 0, self.head_size)[chunk_idx]
        elif shard_idx == 2:
            loaded_weight = _chunk_align(loaded_weight, chunk_size, 0, self.head_size_v)[chunk_idx]
        param_w.copy_(loaded_weight)

    def weight_spliter(self, loaded_weight: torch.Tensor, layout: str = 'default'):
        """weight spliter."""
        _check_qkv_split_layout(layout)
        if layout == 'default':
            return loaded_weight.split(self.qkv_split_section, dim=0)
        elif layout == 'hgd':
            assert self.head_size == self.head_size_v
            heads = [sec // self.head_size for sec in self.qkv_split_section]
            kv_heads = heads[-1]
            loaded_weight = loaded_weight.unflatten(0, (kv_heads, -1, self.head_size))
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
                 tp_align_size: int = 1,
                 dp_gather: bool = False,
                 dp_scatter: bool = False) -> nn.Module:
    """build linear."""
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
    """build columnwise parallel linear layer."""
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
    """build rowwise parallel linear layer."""
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
    """merge linear."""
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
    """build qkv proj."""
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
    """build down linear."""
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
    """build gate up linear."""
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
    """build down linear."""
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
