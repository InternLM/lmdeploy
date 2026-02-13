# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Sequence, Tuple

import torch
from torch import nn
from torch.profiler import record_function

from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader


def build_rmsnorm_gated(hidden_size: int, eps=1e-6, **kwargs):
    from fla.modules import FusedRMSNormGated
    return FusedRMSNormGated(hidden_size, eps=eps, **kwargs)


class GatedDeltaMeta:

    def __init__(self, num_tokens: int, conv_kernel_size: int, state_ids: torch.Tensor, attn_metadata: Any):
        self.num_tokens = num_tokens
        self.is_decoding = attn_metadata.is_decoding
        self.cu_seqlens = attn_metadata.cu_seqlens_q
        device = self.cu_seqlens.device

        # get seq_idx (1, num_tokens)
        seqlens = attn_metadata.q_seqlens
        batch_size = seqlens.numel()
        batch_idx = torch.arange(0, batch_size, dtype=torch.int32, device=device)
        self.seq_idx = torch.repeat_interleave(batch_idx, seqlens, output_size=num_tokens)[None]

        # conv_idx
        range_idx = torch.arange(-conv_kernel_size, 0, device=device)
        self.conv_idx = self.cu_seqlens[1:, None] + range_idx[None]
        self.conv_idx = self.conv_idx.clamp_min(0)

        # state_ids, fill invalid state with state_ids[0]
        self.valid_state = state_ids >= 0
        self.state_ids = torch.where(self.valid_state, state_ids, state_ids[0])

        # we assume 0 is dummy state, shared by all invalid states.
        self.state_ids = self.state_ids.clamp(0)


class CausalConv1dFunc:

    def __init__(self, activation: str = 'silu'):
        try:
            import causal_conv1d
            self.causal_conv1d_fn = causal_conv1d.causal_conv1d_fn
            self.causal_conv1d_update = causal_conv1d.causal_conv1d_update
        except Exception:
            raise RuntimeError(
                'causal_conv1d is not installed, please refer to https://github.com/Dao-AILab/causal-conv1d')
        self.activation = activation

    def conv1d_func(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, conv_state: torch.Tensor,
                    gated_delta_meta: GatedDeltaMeta):
        """
        x: (b, seqlen, dim)
        seqlen: (b)
        out: (b, seqlen, dim)
        conv_state: (b, dim, kernel_size)
        """
        seq_idx = gated_delta_meta.seq_idx
        conv_idx = gated_delta_meta.conv_idx

        assert x.dim() == 3
        x = x.transpose(-2, -1)
        if weight.dim() == 3:
            assert weight.size(1) == 1
            weight = weight[:, 0]

        # fill conv state
        batch_size = conv_state.size(0)
        conv_idx = conv_idx[:, None].expand(-1, x.size(1), -1)
        torch.gather(x.expand(batch_size, -1, -1), -1, conv_idx, out=conv_state)

        out = self.causal_conv1d_fn(
            x,
            weight,
            bias,
            seq_idx,
            return_final_states=False,
            activation=self.activation,
        )

        out = out.transpose(-2, -1)

        # store conv_state
        return out, conv_state

    def conv1d_update(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, conv_state: torch.Tensor):
        if weight.dim() == 3:
            assert weight.size(1) == 1
            weight = weight[:, 0]
        out = self.causal_conv1d_update(x[0], conv_state, weight, bias, activation=self.activation)
        return out[None], conv_state

    @record_function('causal_conv1d')
    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        conv_state: torch.Tensor,
        gated_delta_meta: GatedDeltaMeta,
    ):
        if gated_delta_meta.is_decoding:
            return self.conv1d_update(x, weight, bias, conv_state)
        return self.conv1d_func(x, weight, bias, conv_state, gated_delta_meta=gated_delta_meta)


class GatedDelta:

    def __init__(self, use_qk_l2norm_in_kernel: bool = True):
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
            self.chunk_gated_delta_rule = chunk_gated_delta_rule
            self.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
        except Exception:
            raise RuntimeError(
                'fla is not installed, please refer to https://github.com/fla-org/flash-linear-attention')
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        gated_delta_meta: GatedDeltaMeta,
    ):
        """call."""
        is_decoding = gated_delta_meta.is_decoding
        cu_seqlens = gated_delta_meta.cu_seqlens

        if not is_decoding:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )
        else:
            # qkvgb (1, seqlen, ...) -> (seqlen, 1, ...)
            core_attn_out, last_recurrent_state = self.fused_recurrent_gated_delta_rule(
                query[0, :, None],
                key[0, :, None],
                value[0, :, None],
                g=g[0, :, None],
                beta=beta[0, :, None],
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
            )
            # out (seqlen, 1, ...) -> (1, seqlen, ...)
            core_attn_out = core_attn_out[None, :, 0]
        return core_attn_out, last_recurrent_state


class CausalConv1d(nn.Module):
    """Causal conv1d wrapper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int],
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
        kernel_size: int | Tuple[int],
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
        return self.causal_conv1d_func(x, self.weight, self.bias, conv_state, gated_delta_meta=gated_delta_meta)


@record_function('gated_delta_load_state')
def load_state(past_key_value: Tuple[torch.Tensor, torch.Tensor], gated_delta_meta: GatedDeltaMeta):
    """Load states from cache."""
    state_ids = gated_delta_meta.state_ids
    conv_cache, recurrent_cache = past_key_value[:2]

    return conv_cache.index_select(0, state_ids), recurrent_cache.index_select(0, state_ids)


@record_function('gated_delta_store_state')
def store_state(conv_state: torch.Tensor, recurrent_state: torch.Tensor,
                past_key_value: Tuple[torch.Tensor, torch.Tensor], gated_delta_meta: GatedDeltaMeta):
    """Store states to cache."""
    conv_cache, recurrent_cache = past_key_value[:2]
    state_ids = gated_delta_meta.state_ids

    conv_cache = conv_cache.index_copy_(0, state_ids, conv_state)
    recurrent_cache = recurrent_cache.index_copy_(0, state_ids, recurrent_state.to(recurrent_cache.dtype))
