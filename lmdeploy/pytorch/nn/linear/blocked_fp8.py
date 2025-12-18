# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional

import torch

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader

from ..quant_utils import quant_blocked_fp8
from ..utils import div_up, get_distribute_size
from .base import LinearBase
from .utils import QKVMixin, check_qkv_split_layout


class BlockedF8Linear(LinearBase):
    """Blocked f8 linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_fmt: Optional[str] = None,
        colwise: bool = True,
        is_tp: bool = False,
        all_reduce: bool = True,
        dp_gather: bool = False,
        layer_type: str = 'attn',
    ):
        super().__init__(dtype=dtype,
                         device=device,
                         colwise=colwise,
                         is_tp=is_tp,
                         all_reduce=all_reduce,
                         dp_gather=dp_gather,
                         layer_type=layer_type)
        self.block_size = 128
        self.fp8_dtype = fp8_dtype
        self.scale_fmt = scale_fmt
        if self.is_tp:
            in_features, out_features = self._get_io_features(in_features, out_features, colwise)
        impl_builder = get_backend().get_layer_impl_builder(OpType.LinearBlockedF8)
        self.impl = impl_builder.build(in_features,
                                       out_features,
                                       block_size=128,
                                       bias=bias is not None,
                                       dtype=self.dtype)
        self.impl.set_scale_fmt(scale_fmt)
        weight, weight_scale_inv, bias = self.create_weights(in_features, out_features, bias, self.dtype, self.device)
        self.register_all_parameters(weight, weight_scale_inv, bias)

        self.in_features = in_features
        self.out_features = out_features

    def setup_loaders(self):
        """Setup weight loaders."""
        self.weight.weight_loader = self.weight_loader_with_quant
        self.weight_scale_inv.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader

    def register_all_parameters(self,
                                weight: torch.Tensor,
                                weight_scale_inv: torch.Tensor,
                                bias: Optional[torch.Tensor] = None):
        """Register all parameters."""
        weight = torch.nn.Parameter(weight, requires_grad=False)
        weight_scale_inv = torch.nn.Parameter(weight_scale_inv, requires_grad=False)
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
        self.register_parameter('weight', weight)
        self.register_parameter('weight_scale_inv', weight_scale_inv)
        self.register_parameter('bias', bias)
        self.setup_loaders()

    def _get_io_features(self, in_features: int, out_features: int, colwise: bool):
        """Get io features."""
        world_size, rank = self.get_tp_world_rank()
        if colwise:
            out_features = get_distribute_size(out_features, world_size, rank)
        else:
            in_features = get_distribute_size(in_features, world_size, rank)
        return in_features, out_features

    def _weight_loader_tp_colwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """Weight loader for colwise linear."""
        weight = loaded_weight.chunk(world_size, 0)[rank]
        return default_weight_loader(param, weight)

    def _weight_loader_tp_rowwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """Weight loader for rowwise linear."""
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
        """Weight loader."""
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = self.get_tp_world_rank()
        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank, world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank, world_size)

    def weight_loader_with_quant(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """Weight loader with weight quant."""
        if loaded_weight.dtype != param.dtype:
            # quant loaded weight
            quanted_weight, scaling = quant_blocked_fp8(loaded_weight.to(param.device),
                                                        param.dtype,
                                                        self.block_size,
                                                        scale_fmt=self.scale_fmt)
            self.weight_loader(self.weight, quanted_weight)
            self.weight_loader(self.weight_scale_inv, scaling)
        else:
            return self.weight_loader(param, loaded_weight)

    def create_weights(self, in_features: int, out_features: int, bias: bool, dtype: torch.dtype, device: torch.device):
        """Create weights."""
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
        """Update weights."""
        weight, weight_scale_inv, bias = self.impl.update_weights(self.weight, self.weight_scale_inv, self.bias)
        self.register_all_parameters(weight, weight_scale_inv, bias)

    def _forward_default(self, x, all_reduce, tp_sizes):
        """Default forward implement."""
        if self.tp_mode == TPMode.DP_TP:
            rank = self.tp_rank
            return self.impl.forward(x,
                                     self.weight,
                                     self.weight_scale_inv,
                                     self.bias,
                                     all_reduce,
                                     group=self.tp_group,
                                     rank=rank,
                                     scatter_size=tp_sizes)
        else:
            return self.impl.forward(x, self.weight, self.weight_scale_inv, self.bias, all_reduce, group=self.tp_group)


class MergedBlockedF8Linear(BlockedF8Linear):
    """Merged blocked fp8 linear."""

    def __init__(self,
                 in_features: int,
                 all_out_features: List[int],
                 bias: bool,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 scale_fmt: Optional[str] = None,
                 replicate: Optional[List[bool]] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 out_names: Optional[List[int]] = None,
                 dp_gather: bool = False,
                 layer_type: str = 'attn'):
        self.init_tp_args(is_tp, all_reduce=False, colwise=True, layer_type=layer_type)
        if replicate is None:
            replicate = tuple(False for _ in all_out_features)
        self.block_size = 128
        self.split_section = all_out_features
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
                         scale_fmt=scale_fmt,
                         colwise=True,
                         is_tp=is_tp,
                         dp_gather=dp_gather,
                         layer_type=layer_type)
        self.setup_loaders()

    def setup_loaders(self):
        """Setup weight loaders."""
        self.weight.weight_loader = self.weight_loader_with_quant
        self.weight.weight_spliter = self.weight_spliter
        self.weight._weight_type = 'qweight'
        self.weight_scale_inv.weight_loader = self.weight_loader
        self.weight_scale_inv.weight_spliter = self.weight_spliter
        self.weight_scale_inv._weight_type = 'scales'
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
            self.bias.weight_spliter = self.weight_spliter
            self.bias._weight_type = 'bias'

    def _get_io_features(self, in_features: int, out_features: int, colwise: bool):
        """Get io features."""
        return in_features, out_features

    def _update_all_out_features(self, all_out_features: List[int], replicate: Optional[List[bool]]):
        """Update all out features."""
        world_size, rank = self.get_tp_world_rank()
        new_all_out_features = []
        for out_feat, rep in zip(all_out_features, replicate):
            if rep:
                new_all_out_features.append(out_feat)
            new_out_feat = get_distribute_size(out_feat, world_size, rank)
            new_all_out_features.append(new_out_feat)
        return new_all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """Weight loader."""
        world_size, rank = self.get_tp_world_rank()
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

    def weight_loader_with_quant(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """Weight loader with weight quant."""
        if loaded_weight.dtype != param.dtype:
            # quant loaded weight
            quanted_weight, scaling = quant_blocked_fp8(loaded_weight.to(param.device),
                                                        param.dtype,
                                                        self.block_size,
                                                        scale_fmt=self.scale_fmt)
            self.weight_loader(self.weight, quanted_weight, shard_id)
            self.weight_loader(self.weight_scale_inv, scaling, shard_id)
        else:
            return self.weight_loader(param, loaded_weight, shard_id)

    def weight_spliter(self, loaded_weight: torch.Tensor):
        """Weight spliter."""
        if loaded_weight.dim() == 2 and loaded_weight.dtype != self.fp8_dtype:
            return loaded_weight.split(self.scale_split_section, dim=0)
        return loaded_weight.split(self.split_section, dim=0)

    def weight_spliter_lora_b(self, loaded_weight: torch.Tensor):
        return loaded_weight.split(self.split_section, dim=0)


class QKVBlockedF8Linear(MergedBlockedF8Linear, QKVMixin):
    """Qkv blockedf8 linear."""

    def __init__(self,
                 in_features: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 head_size: int,
                 head_size_v: int,
                 bias: bool = False,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 scale_fmt: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 is_tp: bool = True,
                 dp_gather: bool = False,
                 num_replicate_kv_heads: int = 1):
        self.block_size = 128
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
                         dtype=dtype,
                         fp8_dtype=fp8_dtype,
                         scale_fmt=scale_fmt,
                         bias=bias,
                         device=device,
                         is_tp=is_tp,
                         out_names=out_names,
                         dp_gather=dp_gather,
                         layer_type='attn')

    def _update_all_out_features(self, all_out_features: List[int], replicate: Optional[List[bool]]):
        """Update all out features."""
        return all_out_features

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """Weight loader."""
        _, rank = self.get_tp_world_rank()
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

    def weight_loader_with_quant(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """Weight loader with weight quant."""
        if loaded_weight.dtype != param.dtype:
            # quant loaded weight
            quanted_weight, scaling = quant_blocked_fp8(loaded_weight.to(param.device),
                                                        param.dtype,
                                                        self.block_size,
                                                        scale_fmt=self.scale_fmt)
            self.weight_loader(self.weight, quanted_weight, shard_id)
            self.weight_loader(self.weight_scale_inv, scaling, shard_id)
        else:
            return self.weight_loader(param, loaded_weight, shard_id)

    def weight_spliter(self, loaded_weight: torch.Tensor, layout: str = 'default'):
        """Weight spliter."""
        check_qkv_split_layout(layout)
        assert layout == 'default'
        qkv_split_section = self.qkv_split_section
        if loaded_weight.dim() == 2 and loaded_weight.dtype != self.fp8_dtype:
            qkv_split_section = [sec // self.block_size for sec in qkv_split_section]
        return loaded_weight.split(qkv_split_section, dim=0)
