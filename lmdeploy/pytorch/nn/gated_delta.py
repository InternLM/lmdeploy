# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.profiler import record_function

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaMeta
from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def build_rmsnorm_gated(hidden_size: int, eps=1e-6, **kwargs):
    # TODO: used custom kernel
    from fla.modules import FusedRMSNormGated
    try:
        # avoid unwanted specialize
        from fla.modules.fused_norm_gate import layer_norm_gated_fwd_kernel
        keys = layer_norm_gated_fwd_kernel.fn.keys
        if 'NB' in keys:
            keys.remove('NB')
    except Exception:
        logger.debug('patch layer_norm_gated_fwd_kernel autotuning failed.')
    return FusedRMSNormGated(hidden_size, eps=eps, **kwargs)


class GatedDeltaMetaBuilder:
    """Build shared gated-delta metadata through the selected backend."""

    def __init__(self) -> None:
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.GatedDeltaMeta)
        self.impl = builder.build()

    def __call__(
        self,
        num_tokens: int,
        conv_kernel_size: int,
        state_ids: torch.Tensor,
        attn_metadata: Any,
    ) -> GatedDeltaMeta:
        return self.impl.forward(num_tokens, conv_kernel_size, state_ids, attn_metadata)


class CausalConv1dFunc:

    def __init__(self, activation: str = 'silu'):
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.CausalConv1d)
        self.impl = builder.build()
        self.activation = activation

    @record_function('causal_conv1d')
    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        conv_state: torch.Tensor,
        gated_delta_meta: GatedDeltaMeta,
    ):
        return self.impl.forward(x, weight, bias, conv_state, gated_delta_meta, activation=self.activation)


class GatedDelta:

    def __init__(self, use_qk_l2norm_in_kernel: bool = True):
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.GatedDeltaRule)
        self.impl = builder.build()
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log_exp: torch.Tensor,
        recurrent_state: torch.Tensor,
        gated_delta_meta: GatedDeltaMeta,
        kv_ratio: int = 1,
    ):
        """call."""
        return self.impl.forward(
            query,
            key,
            value,
            b,
            a,
            dt_bias,
            a_log_exp,
            recurrent_state,
            gated_delta_meta,
            kv_ratio,
            self.use_qk_l2norm_in_kernel,
        )


class CausalConv1d(nn.Module):
    """Causal conv1d wrapper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        split: Sequence[int],
        groups: int = 1,
        bias: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        tp, rank = get_tp_world_rank()
        self.tp = tp
        self.rank = rank
        in_channels = in_channels // tp
        out_channels = out_channels // tp
        groups = groups // tp
        assert len(split) == 3
        self.split = split

        weight, w_bias = self.make_weight(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.register_weight(weight, w_bias)
        self.causal_conv1d_func = CausalConv1dFunc(activation='silu')

    @staticmethod
    def make_weight(
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        groups: int = 1,
        bias: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        weight_shape = (out_channels, in_channels // groups,
                        kernel_size if isinstance(kernel_size, int) else kernel_size[0])
        bias_shape = (out_channels, ) if bias else None

        weight = torch.empty(weight_shape, device=device, dtype=dtype)
        if bias_shape is not None:
            w_bias = torch.empty(bias_shape, device=device, dtype=dtype)
        else:
            w_bias = None
        return weight, w_bias

    def register_weight(self, weight: torch.Tensor, w_bias: torch.Tensor | None = None):
        self.register_parameter('weight', nn.Parameter(weight))
        self.weight.weight_loader = self.weight_loader
        if w_bias is not None:
            self.register_parameter('bias', nn.Parameter(w_bias))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """Weight loader."""
        q, k, v = loaded_weight.split(self.split, dim=0)
        q = q.chunk(self.tp, dim=0)[self.rank]
        k = k.chunk(self.tp, dim=0)[self.rank]
        v = v.chunk(self.tp, dim=0)[self.rank]
        loaded_weight = torch.cat([q, k, v], dim=0)
        default_weight_loader(param, loaded_weight)

    def forward(self, x: torch.Tensor, conv_state: torch.Tensor, gated_delta_meta: GatedDeltaMeta):
        """forward."""
        return self.causal_conv1d_func(x, self.weight, self.bias, conv_state, gated_delta_meta)
