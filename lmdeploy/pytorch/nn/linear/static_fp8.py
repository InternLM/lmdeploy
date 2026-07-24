# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any

import torch

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.weight_loader.model_weight_loader import (
    default_weight_loader,
)

from ..utils import get_distribute_size
from .base import LinearBase
from .utils import QKVMixin, check_qkv_split_layout


class StaticF8Linear(LinearBase):
    """Static per-tensor FP8 linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        colwise: bool = True,
        is_tp: bool = False,
        all_reduce: bool = True,
        dp_gather: bool = False,
        layer_type: str = 'attn',
    ):
        super().__init__(
            dtype=dtype,
            device=device,
            colwise=colwise,
            is_tp=is_tp,
            all_reduce=all_reduce,
            dp_gather=dp_gather,
            layer_type=layer_type,
        )

        if self.is_tp:
            in_features, out_features = self._get_io_features(
                in_features,
                out_features,
                colwise,
            )

        impl_builder = get_backend().get_layer_impl_builder(
            OpType.LinearStaticF8,
        )
        self.impl = impl_builder.build(
            in_features,
            out_features,
            bias=bias,
            dtype=self.dtype,
        )

        self.in_features = in_features
        self.out_features = out_features
        self.fp8_dtype = fp8_dtype

        weight = torch.empty(
            out_features,
            in_features,
            dtype=fp8_dtype,
            device=self.device,
        )
        input_scale = torch.empty(
            1,
            dtype=torch.float32,
            device=self.device,
        )
        weight_scale = torch.empty(
            out_features,
            dtype=torch.float32,
            device=self.device,
        )
        bias_tensor = None
        if bias:
            bias_tensor = torch.empty(
                out_features,
                dtype=self.dtype,
                device=self.device,
            )

        self.register_all_parameters(
            weight,
            input_scale,
            weight_scale,
            bias_tensor,
        )

    def register_all_parameters(
        self,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
    ):
        """Register parameters."""
        self.register_parameter(
            'weight',
            torch.nn.Parameter(weight, requires_grad=False),
        )
        self.register_parameter(
            'input_scale',
            torch.nn.Parameter(input_scale, requires_grad=False),
        )
        self.register_parameter(
            'weight_scale',
            torch.nn.Parameter(weight_scale, requires_grad=False),
        )

        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
        self.register_parameter('bias', bias)
        self.setup_loaders()

    def _get_io_features(
        self,
        in_features: int,
        out_features: int,
        colwise: bool,
    ):
        """Get tensor-parallel local dimensions."""
        world_size, rank = self.get_tp_world_rank()

        if colwise:
            out_features = get_distribute_size(
                out_features,
                world_size,
                rank,
            )
        else:
            in_features = get_distribute_size(
                in_features,
                world_size,
                rank,
            )

        return in_features, out_features

    def setup_loaders(self):
        """Set parameter loaders."""
        self.weight.weight_loader = self.weight_loader
        self.input_scale.weight_loader = default_weight_loader
        self.weight_scale.weight_loader = self.weight_scale_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        """Load and shard weight or bias."""
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = self.get_tp_world_rank()

        if self.colwise:
            loaded_weight = loaded_weight.chunk(
                world_size,
                dim=0,
            )[rank]
        elif loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.chunk(
                world_size,
                dim=1,
            )[rank]
        elif rank != 0:
            # Row-parallel bias is added only once.
            loaded_weight = torch.zeros_like(loaded_weight)

        return default_weight_loader(param, loaded_weight)

    def weight_scale_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        """Load static weight scale."""
        loaded_weight = loaded_weight.float().reshape(-1)

        if loaded_weight.numel() == 1:
            # Per-tensor checkpoint scale: broadcast to local output channels.
            loaded_weight = (
                loaded_weight.expand(param.shape).contiguous()
            )
        elif self.is_tp and self.colwise:
            # Also support a future per-output-channel checkpoint.
            world_size, rank = self.get_tp_world_rank()
            loaded_weight = loaded_weight.chunk(
                world_size,
                dim=0,
            )[rank]

        # Row-parallel ranks keep the complete output-channel scale.
        return default_weight_loader(param, loaded_weight)

    def update_weights(self):
        """Update backend weights."""
        values = self.impl.update_weights(
            self.weight,
            self.input_scale,
            self.weight_scale,
            self.bias,
        )
        self.register_all_parameters(*values)

    def _forward_default(self, x, all_reduce, tp_sizes):
        """Run static FP8 linear."""
        if self.tp_mode == TPMode.DP_TP:
            return self.impl.forward(
                x,
                self.weight,
                self.input_scale,
                self.weight_scale,
                self.bias,
                all_reduce=all_reduce,
                group=self.tp_group,
                rank=self.tp_rank,
                scatter_size=tp_sizes,
            )

        return self.impl.forward(
            x,
            self.weight,
            self.input_scale,
            self.weight_scale,
            self.bias,
            all_reduce=all_reduce,
            group=self.tp_group,
       )

class MergedStaticF8Linear(StaticF8Linear):
    """Merged static per-tensor FP8 linear."""

    def __init__(
        self,
        in_features: int,
        all_out_features: list[int],
        bias: bool,
        dp_gather: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        is_tp: bool = True,
        out_names: list[Any] | None = None,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        layer_type: str = 'attn',
    ):
        self.init_tp_args(
            is_tp,
            all_reduce=False,
            colwise=True,
            layer_type=layer_type,
        )

        self.split_section = all_out_features
        self.all_out_features = self._update_all_out_features(
            all_out_features,
        )

        if out_names is None:
            out_names = list(range(len(all_out_features)))

        self.out_names_map = {
            name: index
            for index, name in enumerate(out_names)
        }

        super().__init__(
            in_features=in_features,
            out_features=sum(self.all_out_features),
            bias=bias,
            dp_gather=dp_gather,
            dtype=dtype,
            device=device,
            fp8_dtype=fp8_dtype,
            colwise=True,
            is_tp=is_tp,
            all_reduce=False,
            layer_type=layer_type,
        )

    def _get_io_features(
        self,
        in_features: int,
        out_features: int,
        colwise: bool,
    ):
        """Dimensions were already sharded by the merged layer."""
        return in_features, out_features

    def _update_all_out_features(
        self,
        all_out_features: list[int],
    ):
        """Get local output size for every packed projection."""
        world_size, rank = self.get_tp_world_rank()

        return [
            get_distribute_size(
                out_features,
                world_size,
                rank,
            )
            for out_features in all_out_features
        ]

    def setup_loaders(self):
        """Set packed parameter loaders."""
        self.weight.weight_loader = self.weight_loader
        self.weight.weight_spliter = self.weight_spliter

        self.input_scale.weight_loader = (
            self.input_scale_loader
        )

        self.weight_scale.weight_loader = (
            self.merged_weight_scale_loader
        )
        self.weight_scale.weight_spliter = (
            self.weight_spliter
        )

        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
            self.bias.weight_spliter = self.weight_spliter

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: Any,
    ):
        """Load one packed weight or bias shard."""
        world_size, rank = self.get_tp_world_rank()
        shard_index = self.out_names_map[shard_id]

        target = param.data.split(
            self.all_out_features,
            dim=0,
        )[shard_index]

        loaded_weight = loaded_weight.chunk(
            world_size,
            dim=0,
        )[rank]

        target.copy_(loaded_weight)

    def input_scale_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: Any,
    ):
        """Load the shared static activation scale."""
        del shard_id
        default_weight_loader(
            param,
            loaded_weight.float().reshape(1),
        )

    def merged_weight_scale_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: Any,
    ):
        """Broadcast one projection scale to its output channels."""
        world_size, rank = self.get_tp_world_rank()
        shard_index = self.out_names_map[shard_id]

        target = param.data.split(
            self.all_out_features,
            dim=0,
        )[shard_index]

        loaded_weight = loaded_weight.float().reshape(-1)

        if loaded_weight.numel() == 1:
            loaded_weight = (
                loaded_weight.expand(target.shape).contiguous()
            )
        else:
            loaded_weight = loaded_weight.chunk(
                world_size,
                dim=0,
            )[rank]

        target.copy_(loaded_weight)

    def weight_spliter(
        self,
        loaded_weight: torch.Tensor,
    ):
        """Split a packed tensor into original projections."""
        return loaded_weight.split(
            self.split_section,
            dim=0,
        )

class QKVStaticF8Linear(
    MergedStaticF8Linear,
    QKVMixin,
):
    """Packed QKV static FP8 linear."""

    def __init__(
        self,
        in_features: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_size: int,
        head_size_v: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        is_tp: bool = True,
        num_replicate_kv_heads: int = 1,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        self.init_tp_args(
            is_tp,
            all_reduce=False,
            colwise=True,
            layer_type='attn',
        )

        QKVMixin.__init__(
            self,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size_v,
            num_replicate_kv_heads=num_replicate_kv_heads,
            is_tp=is_tp,
            tp=self.tp,
            tp_rank=self.tp_rank,
        )

        super().__init__(
            in_features=in_features,
            all_out_features=list(
                self.get_qkv_out_feautures()
            ),
            bias=bias,
            dtype=dtype,
            device=device,
            is_tp=is_tp,
            out_names=['q', 'k', 'v'],
            fp8_dtype=fp8_dtype,
            layer_type='attn',
        )

    def _update_all_out_features(
        self,
        all_out_features: list[int],
    ):
        """QKV dimensions were already sharded by QKVMixin."""
        return all_out_features

    def _get_source_section(self, shard_id: Any):
        """Get this TP rank's source range."""
        _, rank = self.get_tp_world_rank()

        if shard_id == 'q':
            num_heads = self.num_q_heads
            head_dim = self.head_size
            rank_index = rank
        else:
            num_heads = self.num_kv_heads
            head_dim = (
                self.head_size
                if shard_id == 'k'
                else self.head_size_v
            )
            rank_index = (
                rank // self.num_replicate_kv_heads
            )

        section_length = num_heads * head_dim
        section_start = rank_index * section_length
        return section_start, section_length

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: Any,
    ):
        """Load one QKV weight or bias shard."""
        shard_index = self.out_names_map[shard_id]
        target = param.data.split(
            self.all_out_features,
            dim=0,
        )[shard_index]

        section_start, section_length = (
            self._get_source_section(shard_id)
        )
        loaded_weight = loaded_weight.narrow(
            dim=0,
            start=section_start,
            length=section_length,
        )
        target.copy_(loaded_weight)

    def merged_weight_scale_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: Any,
    ):
        """Load one QKV weight scale."""
        shard_index = self.out_names_map[shard_id]
        target = param.data.split(
            self.all_out_features,
            dim=0,
        )[shard_index]

        loaded_weight = loaded_weight.float().reshape(-1)

        if loaded_weight.numel() == 1:
            loaded_weight = (
                loaded_weight.expand(target.shape).contiguous()
            )
        else:
            section_start, section_length = (
                self._get_source_section(shard_id)
            )
            loaded_weight = loaded_weight.narrow(
                dim=0,
                start=section_start,
                length=section_length,
            )

        target.copy_(loaded_weight)

    def weight_spliter(
        self,
        loaded_weight: torch.Tensor,
        layout: str = 'default',
    ):
        """Split packed QKV parameters."""
        check_qkv_split_layout(layout)

        if layout != 'default':
            raise RuntimeError(
                'Static FP8 currently supports only '
                f'default QKV layout, but got {layout}.'
            )

        return loaded_weight.split(
            self.qkv_split_section,
            dim=0,
        )
