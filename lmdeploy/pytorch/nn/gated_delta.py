# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn
from torch.profiler import record_function

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.causal_conv1d import CausalConv1dBuildSpec
from lmdeploy.pytorch.backends.gated_delta_rule import (
    GatedDeltaMeta,
    GatedDeltaMetaBuildSpec,
    GatedDeltaRuleBuildSpec,
)
from lmdeploy.pytorch.consts import (
    GDN_CONV_STATE_CACHE_NAME,
    GDN_REPLAY_BETA_CACHE_NAME,
    GDN_REPLAY_CHECKPOINT_CACHE_NAME,
    GDN_REPLAY_CURSOR_CACHE_NAME,
    GDN_REPLAY_D_CACHE_NAME,
    GDN_REPLAY_G_CACHE_NAME,
    GDN_REPLAY_K_CACHE_NAME,
    GDN_REPLAY_RAW_K_CACHE_NAME,
    GDN_REPLAY_RAW_V_CACHE_NAME,
)
from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.models.patch import get_build_model_context
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


@dataclass(frozen=True)
class GatedDeltaReplayCursor:
    """Shared fixed-address ReplaySSM cursor storage."""

    packed_cursor: torch.Tensor
    L: int
    R: int
    Tmax: int

    WRITE_POS: ClassVar[int] = 0
    CACHE_BASE: ClassVar[int] = 1
    IS_FLUSH: ClassVar[int] = 2


@dataclass(frozen=True)
class GatedDeltaReplayStep:
    """Step-local ReplaySSM inputs shared across GDN layers."""

    state_indices: torch.Tensor
    query_start_loc: torch.Tensor
    num_accepted_inputs: torch.Tensor | None = None
    force_flush: torch.Tensor | None = None


@dataclass(frozen=True)
class GatedDeltaReplayState:
    """Layer-local views of ReplaySSM persistent state."""

    mode: str
    conv_state: torch.Tensor
    checkpoint: torch.Tensor
    d_ring: torch.Tensor
    k_ring: torch.Tensor
    g_ring: torch.Tensor
    cursor: GatedDeltaReplayCursor
    raw_v_ring: torch.Tensor | None = None
    raw_k_ring: torch.Tensor | None = None
    beta_ring: torch.Tensor | None = None
    force_flush: torch.Tensor | None = None


def _named_layer(caches: Mapping[str, torch.Tensor], name: str, layer_id: int):
    if hasattr(caches, 'layer'):
        return caches.layer(name, layer_id)
    return caches[name][layer_id]


def build_gated_delta_replay_state(caches: Mapping[str, torch.Tensor],
                                   layer_id: int,
                                   mode: str,
                                   buffer_len: int,
                                   tmax: int,
                                   force_flush: torch.Tensor | None = None) -> GatedDeltaReplayState:
    """Build cheap tensor views for one global GDN layer."""
    cursor = GatedDeltaReplayCursor(caches[GDN_REPLAY_CURSOR_CACHE_NAME],
                                    L=buffer_len,
                                    R=buffer_len,
                                    Tmax=tmax)
    kwargs = {}
    if mode == 'mtp':
        kwargs = dict(
            raw_v_ring=_named_layer(caches, GDN_REPLAY_RAW_V_CACHE_NAME, layer_id),
            raw_k_ring=_named_layer(caches, GDN_REPLAY_RAW_K_CACHE_NAME, layer_id),
            beta_ring=_named_layer(caches, GDN_REPLAY_BETA_CACHE_NAME, layer_id),
        )
    return GatedDeltaReplayState(
        mode=mode,
        conv_state=_named_layer(caches, GDN_CONV_STATE_CACHE_NAME, layer_id),
        checkpoint=_named_layer(caches, GDN_REPLAY_CHECKPOINT_CACHE_NAME, layer_id),
        d_ring=_named_layer(caches, GDN_REPLAY_D_CACHE_NAME, layer_id),
        k_ring=_named_layer(caches, GDN_REPLAY_K_CACHE_NAME, layer_id),
        g_ring=_named_layer(caches, GDN_REPLAY_G_CACHE_NAME, layer_id),
        cursor=cursor,
        force_flush=force_flush,
        **kwargs,
    )


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
        self.impl = get_backend().build_op(
            GatedDeltaMetaBuildSpec(),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

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
        self.impl = get_backend().build_op(
            CausalConv1dBuildSpec(),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )
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
        self.impl = get_backend().build_op(
            GatedDeltaRuleBuildSpec(),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )
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
