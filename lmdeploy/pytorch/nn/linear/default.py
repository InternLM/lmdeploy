# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional

import torch

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader

from ..utils import chunk_aligned, get_distribute_size
from .base import LinearBase
from .utils import QKVMixin, check_qkv_split_layout


class BaseLinear(LinearBase):
    """Linear layer."""

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
        layer_type: str = 'attn',
    ):
        super().__init__(dtype=dtype,
                         device=device,
                         colwise=colwise,
                         is_tp=is_tp,
                         all_reduce=all_reduce,
                         tp_align_size=tp_align_size,
                         dp_gather=dp_gather,
                         layer_type=layer_type)
        if self.is_tp:
            in_features, out_features = self._get_io_features(in_features, out_features, colwise)
        impl_builder = get_backend().get_layer_impl_builder(OpType.Linear)
        self.impl = impl_builder.build(in_features, out_features, bias is not None, dtype=self.dtype)
        weight, bias = self.create_weights(in_features, out_features, bias, self.dtype, self.device)
        self.register_all_parameters(weight, bias)

        self.in_features = in_features
        self.out_features = out_features

    def setup_loaders(self):
        """Setup loaders."""
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader

    def register_all_parameters(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Register all parameters."""
        weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
        self.register_parameter('weight', weight)
        self.register_parameter('bias', bias)
        self.setup_loaders()

    def _get_io_features(self, in_features: int, out_features: int, colwise: bool):
        """Get io features."""
        world_size, rank = self.get_tp_world_rank()
        if colwise:
            out_features = get_distribute_size(out_features, world_size, rank, align=self.tp_align_size)
        else:
            in_features = get_distribute_size(in_features, world_size, rank, align=self.tp_align_size)
        return in_features, out_features

    def _weight_loader_tp_colwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """Weight loader for colwise linear."""
        weight = chunk_aligned(loaded_weight, world_size, 0, self.tp_align_size)[rank]
        return default_weight_loader(param, weight)

    def _weight_loader_tp_rowwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """Weight loader for rowwise linear."""
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.to(param.device)
            weight = chunk_aligned(loaded_weight, world_size, 1, self.tp_align_size)[rank]
            return default_weight_loader(param, weight)
        else:
            # bias
            if rank != 0:
                loaded_weight = torch.zeros_like(loaded_weight)
            return default_weight_loader(param, loaded_weight)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """Weight loader."""
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = self.get_tp_world_rank()
        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank, world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank, world_size)

    def create_weights(self, in_features: int, out_features: int, bias: bool, dtype: torch.dtype, device: torch.device):
        """Create weights."""
        weight = torch.empty((out_features, in_features), dtype=dtype, device=device)
        if bias:
            bias = torch.empty((out_features, ), dtype=dtype, device=device)
        else:
            bias = None
        return weight, bias

    def update_weights(self):
        """Update weights."""
        weight, bias = self.impl.update_weights(self.weight, self.bias)
        self.register_all_parameters(weight, bias)

    def _forward_default(self, x, all_reduce, tp_sizes):
        """Default forward implement."""
        if self.tp_mode == TPMode.DP_TP:
            rank = self.tp_rank
            return self.impl.forward(x,
                                     self.weight,
                                     self.bias,
                                     all_reduce,
                                     group=self.tp_group,
                                     rank=rank,
                                     scatter_size=tp_sizes)
        else:
            return self.impl.forward(x, self.weight, self.bias, all_reduce, group=self.tp_group)


class MergedBaseLinear(BaseLinear):
    """Merged base linear."""

    def __init__(self,
                 in_features: int,
                 all_out_features: List[int],
                 bias: bool,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 out_names: Optional[List[int]] = None,
                 dp_gather: bool = False,
                 layer_type: str = 'attn'):
        self.init_tp_args(is_tp, all_reduce=False, colwise=True, layer_type=layer_type)
        self.split_section = all_out_features
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
                         dp_gather=dp_gather,
                         layer_type=layer_type)
        self.setup_loaders()

    def setup_loaders(self):
        """Update loaders."""
        self.weight.weight_loader = self.weight_loader
        self.weight.weight_spliter = self.weight_spliter
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
            self.bias.weight_spliter = self.weight_spliter

    def _get_io_features(self, in_features: int, out_features: int, colwise: bool):
        """Get io features."""
        return in_features, out_features

    def _update_all_out_features(self, all_out_features: List[int]):
        """Update all out features."""
        world_size, rank = self.get_tp_world_rank()
        new_all_out_features = []
        for out_feat in all_out_features:
            new_out_feat = get_distribute_size(out_feat, world_size, rank)
            new_all_out_features.append(new_out_feat)
        return new_all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """Weight loader."""
        world_size, rank = self.get_tp_world_rank()
        shard_idx = self.out_names_map[shard_id]
        param_w = param.data.split(self.all_out_features, 0)[shard_idx]
        loaded_weight = loaded_weight.chunk(world_size, 0)[rank]
        param_w.copy_(loaded_weight)

    def weight_spliter(self, loaded_weight: torch.Tensor):
        """Weight spliter."""
        return loaded_weight.split(self.split_section, dim=0)

    def weight_spliter_lora_b(self, loaded_weight: torch.Tensor):
        return loaded_weight.split(self.split_section, dim=0)


class QKVBaseLinear(MergedBaseLinear, QKVMixin):
    """Qkv base linear."""

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

        all_out_features = self.get_qkv_out_feautures()
        out_names = ('q', 'k', 'v')
        super().__init__(in_features,
                         all_out_features,
                         bias=bias,
                         dtype=dtype,
                         device=device,
                         is_tp=is_tp,
                         out_names=out_names,
                         layer_type='attn')

    def _update_all_out_features(self, all_out_features: List[int]):
        """Update all out features."""
        return all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """Weight loader."""
        world_size, rank = self.get_tp_world_rank()
        chunk_size, chunk_idx = world_size, rank
        shard_idx = self.out_names_map[shard_id]
        param_w = param.data.split(self.all_out_features, 0)[shard_idx]

        if self.num_replicate_kv_heads > 1 and shard_id in ['k', 'v']:
            # update to duplicate k/v for tp_size > num_kv_heads
            chunk_size = world_size // self.num_replicate_kv_heads
            chunk_idx = rank // self.num_replicate_kv_heads
        if shard_idx in [0, 1]:
            loaded_weight = chunk_aligned(loaded_weight, chunk_size, 0, self.head_size)[chunk_idx]
        elif shard_idx == 2:
            loaded_weight = chunk_aligned(loaded_weight, chunk_size, 0, self.head_size_v)[chunk_idx]
        param_w.copy_(loaded_weight)

    def weight_spliter(self, loaded_weight: torch.Tensor, layout: str = 'default'):
        """Weight spliter."""
        check_qkv_split_layout(layout)
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
