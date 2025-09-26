# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional

import torch

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader

from ..utils import chunk_aligned, get_distribute_size
from .base import LinearBase
from .utils import QKVMixin, check_qkv_split_layout


class AwqLinear(LinearBase):
    """W4a16 linear."""

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
        layer_type: str = 'attn',
    ):
        super().__init__(dtype=torch.float16,
                         device=device,
                         colwise=colwise,
                         is_tp=is_tp,
                         all_reduce=all_reduce,
                         layer_type=layer_type)
        if self.is_tp:
            in_features, out_features = self._get_io_features(in_features, out_features, w_bit, group_size, colwise)
        qweight, scales, qzeros, bias = self.create_weights(in_features, out_features, w_bit, group_size, bias,
                                                            self.dtype, self.device)
        impl_builder = get_backend().get_layer_impl_builder(OpType.LinearW4A16)
        self.impl = impl_builder.build(in_features,
                                       out_features,
                                       w_bit,
                                       group_size,
                                       bias is not None,
                                       dtype=scales.dtype)
        self.register_all_parameters(qweight, scales, qzeros, bias)

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size
        self.elem_per_int = 32 // w_bit

    def setup_loaders(self):
        """Setup weight loaders."""
        self.qweight.weight_loader = self.weight_loader
        self.qweight._weight_type = 'qweight'
        self.scales.weight_loader = self.weight_loader
        self.scales._weight_type = 'scales'
        self.qzeros.weight_loader = self.weight_loader
        self.qzeros._weight_type = 'qzeros'
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
            self.bias._weight_type = 'bias'

    def register_all_parameters(self,
                                qweight: torch.Tensor,
                                scales: torch.Tensor,
                                qzeros: torch.Tensor,
                                bias: Optional[torch.Tensor] = None):
        """Register all parameters."""
        qweight = torch.nn.Parameter(qweight, requires_grad=False)
        scales = torch.nn.Parameter(scales, requires_grad=False)
        qzeros = torch.nn.Parameter(qzeros, requires_grad=False)
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
        self.register_parameter('qweight', qweight)
        self.register_parameter('scales', scales)
        self.register_parameter('qzeros', qzeros)
        self.register_parameter('bias', bias)
        self.setup_loaders()

    def _get_io_features(self, in_features: int, out_features: int, w_bit: int, group_size: int, colwise: bool):
        """Get io features."""
        align = max(32 // w_bit, group_size)
        world_size, rank = self.get_tp_world_rank()
        if colwise:
            out_features = get_distribute_size(out_features, world_size, rank, align=align)
        else:
            in_features = get_distribute_size(in_features, world_size, rank, align=align)
        return in_features, out_features

    def _weight_loader_tp_colwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """Weight loader for colwise linear."""
        if loaded_weight.dim() == 1:
            # bias
            align = max(self.elem_per_int, self.group_size)
            weight = chunk_aligned(loaded_weight, world_size, 0, align)[rank]
            return default_weight_loader(param, weight)

        if loaded_weight.size(1) == self.out_features:
            # scaling
            align = max(self.elem_per_int, self.group_size)
            weight = chunk_aligned(loaded_weight, world_size, 1, align)[rank]
            return default_weight_loader(param, weight)

        align = max(self.elem_per_int, self.group_size) // self.elem_per_int
        weight = chunk_aligned(loaded_weight, world_size, 1, align)[rank]
        return default_weight_loader(param, weight)

    def _weight_loader_tp_rowwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """Weight loader for rowwise linear."""
        if loaded_weight.dim() == 1:
            # bias
            if rank == 0:
                loaded_weight = torch.zeros_like(loaded_weight)
            return default_weight_loader(param, loaded_weight)

        if loaded_weight.size(0) == self.in_features:
            # qweight
            align = max(self.elem_per_int, self.group_size)
            weight = chunk_aligned(loaded_weight, world_size, 0, align)[rank]
            return default_weight_loader(param, weight)

        align = max(self.elem_per_int, self.group_size) // self.group_size
        weight = chunk_aligned(loaded_weight, world_size, 0, align)[rank]
        return default_weight_loader(param, weight)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """Weight loader."""
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = self.get_tp_world_rank()
        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank, world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank, world_size)

    def create_weights(self, in_features: int, out_features: int, w_bit: int, group_size: int, bias: bool,
                       dtype: torch.dtype, device: torch.device):
        """Create weights."""
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
        """Update weights."""
        qweight, scales, qzeros, bias = self.impl.update_weights(self.qweight, self.scales, self.qzeros, self.bias)
        self.register_all_parameters(qweight, scales, qzeros, bias)

    def _forward_default(self, x, all_reduce, tp_sizes):
        """Default forward implement."""
        return self.impl.forward(x, self.qweight, self.scales, self.qzeros, self.bias, all_reduce, group=self.tp_group)


class MergedAwqLinear(AwqLinear):
    """Merged awq linear."""

    def __init__(self,
                 in_features: int,
                 all_out_features: List[int],
                 w_bit: int,
                 group_size: int,
                 bias: bool,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 out_names: Optional[List[int]] = None,
                 layer_type: str = 'attn'):
        self.init_tp_args(is_tp, all_reduce=False, colwise=True, layer_type=layer_type)

        self.split_section_s = all_out_features
        elem_per_int = 32 // w_bit
        self.split_section_wz = [size // elem_per_int for size in all_out_features]

        all_out_features = self._update_all_out_features(all_out_features, w_bit, group_size)
        self.all_out_features = all_out_features
        if out_names is None:
            out_names = torch.arange(len(self.all_out_features)).tolist()
        assert len(out_names) == len(self.all_out_features)
        self.out_names_map = dict((name, idx) for idx, name in enumerate(out_names))
        out_features = sum(all_out_features)
        super().__init__(in_features,
                         out_features,
                         w_bit,
                         group_size,
                         bias,
                         device,
                         colwise=True,
                         is_tp=is_tp,
                         layer_type=layer_type)
        self.setup_loaders()

    def setup_loaders(self):
        """Setup weight loaders."""
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
        """Get io features."""
        return in_features, out_features

    def _update_all_out_features(self, all_out_features: List[int], w_bit: int, group_size: int):
        """Update all out features."""
        world_size, rank = self.get_tp_world_rank()
        new_all_out_features = []
        align = max(32 // w_bit, group_size)
        for out_feat in all_out_features:
            new_out_feat = get_distribute_size(out_feat, world_size, rank, align)
            new_all_out_features.append(new_out_feat)
        return new_all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """Weight loader."""
        world_size, rank = self.get_tp_world_rank()
        shard_idx = self.out_names_map[shard_id]
        if loaded_weight.dim() == 1:
            # bias
            align = max(self.elem_per_int, self.group_size)
            param_w = param.data.split(self.all_out_features, 0)[shard_idx]
            weight = chunk_aligned(loaded_weight, world_size, 0, align)[rank]
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

        weight = chunk_aligned(loaded_weight, world_size, -1, align)[rank]
        param_w.copy_(weight)

    def weight_spliter_wz(self, loaded_weight: torch.Tensor):
        """Weight spliter."""
        return loaded_weight.split(self.split_section_wz, dim=1)

    def weight_spliter_s(self, loaded_weight: torch.Tensor):
        """Weight spliter."""
        return loaded_weight.split(self.split_section_s, dim=-1)


class QKVAwqLinear(MergedAwqLinear, QKVMixin):
    """Qkv awq linear."""

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
        self.init_tp_args(is_tp, all_reduce=False, colwise=True, layer_type='attn')
        QKVMixin.__init__(self,
                          num_q_heads=num_q_heads,
                          num_kv_heads=num_kv_heads,
                          head_size=head_size,
                          head_size_v=head_size_v,
                          num_replicate_kv_heads=num_replicate_kv_heads,
                          is_tp=is_tp,
                          tp=self.tp,
                          tp_rank=self.tp_rank)

        elem_per_int = 32 // w_bit
        self.qkv_split_section_s = self.qkv_split_section
        self.qkv_split_section_wz = [size // elem_per_int for size in self.qkv_split_section_s]
        all_out_features = self.get_qkv_out_feautures()
        out_names = ('q', 'k', 'v')
        super().__init__(in_features,
                         all_out_features,
                         w_bit=w_bit,
                         group_size=group_size,
                         bias=bias,
                         device=device,
                         is_tp=is_tp,
                         out_names=out_names,
                         layer_type='attn')

    def _update_all_out_features(self, all_out_features: List[int], w_bit: int, group_size: int):
        """Update all out features."""
        return all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """Weight loader."""
        world_size, rank = self.get_tp_world_rank()
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
            weight = chunk_aligned(loaded_weight, chunk_size, 0, align)[chunk_idx]
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

        weight = chunk_aligned(loaded_weight, chunk_size, -1, align)[chunk_idx]
        param_w.copy_(weight)

    def weight_spliter_wz(self, loaded_weight: torch.Tensor, layout: str = 'default'):
        """Weight spliter."""
        check_qkv_split_layout(layout)
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
        """Weight spliter."""
        check_qkv_split_layout(layout)
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
