# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, NamedTuple

import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.blockedf8_modules import LinearBlockedF8BuildSpec
from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.models.patch import get_build_model_context
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
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_fmt: str | None = None,
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
        self.impl = get_backend().build_op(
            LinearBlockedF8BuildSpec(
                in_features=in_features,
                out_features=out_features,
                block_size=self.block_size,
                bias=bias,
                output_dtype=self.dtype,
                fp8_dtype=self.fp8_dtype,
                scale_fmt=scale_fmt,
            ),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )
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
                                bias: torch.Tensor | None = None):
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

    def get_unquantized_weight(self, out_dtype: torch.dtype) -> torch.Tensor:
        """Return the local dequantized weight."""
        out_features, in_features = self.weight.shape
        scale_rows, scale_cols = self.weight_scale_inv.shape
        aligned_shape = (scale_rows * self.block_size, scale_cols * self.block_size)
        weight = self.weight
        if weight.shape != aligned_shape:
            weight = weight.new_zeros(aligned_shape)
            weight[:out_features, :in_features].copy_(self.weight)
        weight = weight.reshape(scale_rows, self.block_size, scale_cols, self.block_size)
        scale = self.weight_scale_inv[:, None, :, None]
        weight = (weight.to(scale.dtype) * scale).to(out_dtype).reshape(aligned_shape)
        return weight[:out_features, :in_features]

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
                 all_out_features: list[int],
                 bias: bool,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 scale_fmt: str | None = None,
                 replicate: list[bool] | None = None,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 is_tp: bool = True,
                 out_names: list[int] | None = None,
                 dp_gather: bool = False,
                 layer_type: str = 'attn'):
        self.init_tp_args(is_tp, all_reduce=False, colwise=True, layer_type=layer_type)
        if replicate is None:
            replicate = tuple(False for _ in all_out_features)
        self.block_size = 128
        self.split_section = all_out_features
        self.scale_split_section = [div_up(section, self.block_size) for section in self.split_section]
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

    def _update_all_out_features(self, all_out_features: list[int], replicate: list[bool] | None):
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
            local_scale_sections = [div_up(feats, self.block_size) for feats in self.all_out_features]
            param_w = param.data.split(local_scale_sections, 0)[shard_idx]
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


class _QKVShardPlan(NamedTuple):
    """Physical FP8 shard and its logical output view."""

    shard_idx: int
    weight_start: int
    weight_load_rows: int
    physical_rows: int
    valid_start: int
    logical_rows: int
    scale_start: int
    scale_rows: int


class QKVBlockedF8Linear(MergedBlockedF8Linear, QKVMixin):
    """Blocked-FP8 QKV projection with checkpoint-aware output sharding.

    ``checkpoint_output_shard_sizes`` gives the Q/K/V row counts in one
    native checkpoint TP partition. Runtime TP shards are selected inside
    those partitions so weight rows and scale rows stay paired.
    ``continuous_qkv_scale_layout`` preserves checkpoints whose FP8 block
    coordinate continues across the Q/K/V boundaries within each partition.
    """

    def __init__(self,
                 in_features: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 head_size: int,
                 head_size_v: int,
                 bias: bool = False,
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 scale_fmt: str | None = None,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 is_tp: bool = True,
                 dp_gather: bool = False,
                 num_replicate_kv_heads: int = 1,
                 checkpoint_output_shard_sizes: tuple[int, int, int] | None = None,
                 continuous_qkv_scale_layout: bool = False):
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

        logical_out_features = self.get_qkv_out_feautures()
        self.logical_out_features = logical_out_features
        self.checkpoint_output_shard_sizes = checkpoint_output_shard_sizes
        self.continuous_qkv_scale_layout = continuous_qkv_scale_layout
        self._output_shard_plans = self._build_output_shard_plans(logical_out_features)
        all_out_features = tuple(plan.physical_rows for plan in self._output_shard_plans)
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
        self.scale_split_section = [plan.scale_rows for plan in self._output_shard_plans]

    def _update_all_out_features(self, all_out_features: list[int], replicate: list[bool] | None):
        """Update all out features."""
        return all_out_features

    def _build_output_shard_plans(self, logical_sections: tuple[int, int, int]):
        """Align TP-local Q/K/V inside their checkpoint quant shards."""
        _, rank = self.get_tp_world_rank()
        quant_shards = getattr(self, 'checkpoint_output_shard_sizes', None) or self.qkv_split_section
        continuous_qkv_scales = getattr(self, 'continuous_qkv_scale_layout', False)
        if len(logical_sections) != 3 or len(quant_shards) != 3:
            raise ValueError('Blocked-FP8 QKV sharding requires exactly three Q/K/V sections.')
        trailing_padding = 0
        if continuous_qkv_scales:
            trailing_padding = div_up(sum(logical_sections), self.block_size) * self.block_size \
                - sum(logical_sections)
        plans = []
        for shard_idx, (shard_id, logical_rows, source_rows, quant_shard_rows) in enumerate(
                zip(('q', 'k', 'v'), logical_sections, self.qkv_split_section, quant_shards)):
            rank_idx = rank if shard_id == 'q' else rank // self.num_replicate_kv_heads
            logical_start = rank_idx * logical_rows
            logical_end = logical_start + logical_rows
            if logical_end > source_rows:
                raise ValueError(
                    f'QKV {shard_id} TP shard [{logical_start}:{logical_end}] exceeds {source_rows} source rows.')
            if quant_shard_rows <= 0 or source_rows % quant_shard_rows:
                raise ValueError(
                    f'QKV {shard_id} source rows {source_rows} must be divisible by checkpoint '
                    f'quantization shard size {quant_shard_rows}.')
            quant_shard_idx = logical_start // quant_shard_rows
            quant_shard_start = quant_shard_idx * quant_shard_rows
            quant_shard_end = quant_shard_start + quant_shard_rows
            if logical_end > quant_shard_end:
                raise ValueError(
                    f'QKV {shard_id} TP shard [{logical_start}:{logical_end}] crosses checkpoint '
                    f'quantization shard [{quant_shard_start}:{quant_shard_end}].')
            relative_start = logical_start - quant_shard_start
            relative_end = logical_end - quant_shard_start
            if continuous_qkv_scales:
                if relative_start % self.block_size:
                    raise ValueError(
                        f'Fused QKV {shard_id} TP shard starts at unaligned checkpoint row '
                        f'{relative_start}; its scale blocks cannot be repacked without requantization.')
                physical_rows = logical_rows + (trailing_padding if shard_idx == 2 else 0)
                scales_per_quant_shard = div_up(quant_shard_rows, self.block_size)
                scale_start = quant_shard_idx * scales_per_quant_shard + relative_start // self.block_size
                scale_rows = div_up(logical_rows, self.block_size)
                plans.append(
                    _QKVShardPlan(shard_idx, logical_start, logical_rows, physical_rows, 0,
                                  logical_rows, scale_start, scale_rows))
                continue
            physical_relative_start = relative_start // self.block_size * self.block_size
            physical_relative_end = div_up(relative_end, self.block_size) * self.block_size
            physical_rows = physical_relative_end - physical_relative_start
            weight_start = quant_shard_start + physical_relative_start
            weight_end = min(quant_shard_start + physical_relative_end, quant_shard_end)
            weight_load_rows = weight_end - weight_start
            valid_start = relative_start - physical_relative_start
            scales_per_quant_shard = div_up(quant_shard_rows, self.block_size)
            scale_start = quant_shard_idx * scales_per_quant_shard + physical_relative_start // self.block_size
            scale_rows = physical_rows // self.block_size
            plans.append(
                _QKVShardPlan(shard_idx, weight_start, weight_load_rows, physical_rows, valid_start,
                              logical_rows, scale_start, scale_rows))
        if continuous_qkv_scales:
            physical_scale_rows = div_up(sum(plan.physical_rows for plan in plans), self.block_size)
            source_scale_rows = sum(plan.scale_rows for plan in plans)
            if physical_scale_rows != source_scale_rows:
                raise ValueError(
                    'Fused QKV checkpoint scale rows do not match the packed runtime output: '
                    f'{source_scale_rows} != {physical_scale_rows}.')
        return tuple(plans)

    def _get_output_shard(self, shard_id: Any) -> _QKVShardPlan:
        """Return checkpoint weight/scale rows for a TP-local shard."""
        return self._output_shard_plans[self.out_names_map[shard_id]]

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """Weight loader."""
        plan = self._get_output_shard(shard_id)
        all_out_features = self.all_out_features
        if param._weight_type == 'scales':
            loaded_weight = loaded_weight.to(torch.float32)
            all_out_features = self.scale_split_section
            sec_start, sec_len = plan.scale_start, plan.scale_rows
        else:
            sec_start, sec_len = plan.weight_start, plan.weight_load_rows

        loaded_weight = loaded_weight.narrow(dim=0, start=sec_start, length=sec_len)
        param_w = param.data.split(all_out_features, 0)[plan.shard_idx]
        if param._weight_type == 'scales':
            param_w.copy_(loaded_weight)
        else:
            param_w.zero_()
            param_w.narrow(0, 0, sec_len).copy_(loaded_weight)

    def weight_loader_with_quant(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any):
        """Weight loader with weight quant."""
        if loaded_weight.dtype != param.dtype:
            if self.continuous_qkv_scale_layout:
                raise ValueError('Fused QKV output quantization requires native blocked-FP8 checkpoint weights.')
            plan = self._get_output_shard(shard_id)
            physical_weight = torch.zeros((plan.physical_rows, loaded_weight.shape[1]),
                                          dtype=loaded_weight.dtype,
                                          device=param.device)
            physical_weight[:plan.weight_load_rows].copy_(
                loaded_weight.narrow(0, plan.weight_start, plan.weight_load_rows))
            quanted_weight, scaling = quant_blocked_fp8(physical_weight,
                                                        param.dtype,
                                                        self.block_size,
                                                        scale_fmt=self.scale_fmt)
            self.weight.data.split(self.all_out_features, 0)[plan.shard_idx].copy_(quanted_weight)
            self.weight_scale_inv.data.split(self.scale_split_section, 0)[plan.shard_idx].copy_(scaling)
        else:
            return self.weight_loader(param, loaded_weight, shard_id)

    def split_qkv(self, x: torch.Tensor):
        """Crop physical alignment rows and restore logical Q/K/V heads."""
        physical = x.split(self.all_out_features, dim=-1)
        q, k, v = (
            value.narrow(-1, plan.valid_start, plan.logical_rows)
            for value, plan in zip(physical, self._output_shard_plans)
        )
        q = q.unflatten(-1, (self.num_q_heads, self.head_size))
        k = k.unflatten(-1, (self.num_kv_heads, self.head_size))
        v = v.unflatten(-1, (self.num_kv_heads, self.head_size_v))
        return q, k, v

    def weight_spliter(self, loaded_weight: torch.Tensor, layout: str = 'default'):
        """Weight spliter."""
        check_qkv_split_layout(layout)
        assert layout == 'default'
        qkv_split_section = self.qkv_split_section
        if loaded_weight.dim() == 2 and loaded_weight.dtype != self.fp8_dtype:
            qkv_split_section = [div_up(sec, self.block_size) for sec in qkv_split_section]
        return loaded_weight.split(qkv_split_section, dim=0)
