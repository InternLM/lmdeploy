# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.profiler import record_function

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
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


class GatedDeltaMeta:

    def __init__(self, num_tokens: int, conv_kernel_size: int, state_ids: torch.Tensor, attn_metadata: Any):
        self.num_tokens = num_tokens
        self.is_decoding = attn_metadata.is_decoding
        self.cu_seqlens = attn_metadata.cu_seqlens_q
        self.cache_seqlens = None
        self.num_spec_tokens = get_step_ctx_manager().build_ctx.num_spec_tokens
        self.spec_conv_offsets = None
        self.spec_state_offsets = None

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

        # for spec decoding
        if self.num_spec_tokens > 0:
            self.cache_seqlens = (attn_metadata.kv_seqlens - attn_metadata.q_seqlens).to(torch.int32)
            if not self.is_decoding:
                spec_conv_offsets = attn_metadata.kv_seqlens[:, None] + range_idx[None]
                self.spec_conv_offsets = torch.remainder(spec_conv_offsets, conv_kernel_size + self.num_spec_tokens)
                read_state_offsets = torch.remainder(self.cache_seqlens, 1 + self.num_spec_tokens)
                write_state_offsets = torch.remainder(attn_metadata.kv_seqlens, 1 + self.num_spec_tokens)
                self.spec_state_offsets = (read_state_offsets, write_state_offsets)

        self.conv_state_indices = state_ids.to(torch.int32)
        # we assume 0 is dummy state, shared by all invalid states.
        self.valid_state = state_ids >= 0
        self.state_ids = state_ids.clamp(0)


class CausalConv1dFunc:

    def __init__(self, activation: str = 'silu'):
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.CausalConv1d)
        impl = builder.build()
        self.causal_conv1d_fn = impl.conv1d_fn
        self.causal_conv1d_update = impl.update_fn
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
        spec_conv_offsets = gated_delta_meta.spec_conv_offsets
        state_ids = gated_delta_meta.state_ids

        assert x.dim() == 3
        if weight.dim() == 3:
            assert weight.size(1) == 1
            weight = weight[:, 0]

        # fill conv state
        final_state = x[0, conv_idx].transpose(-2, -1)
        # Load all initial conv states before overwriting.
        # (num_seqs, dim, ks-1): last ks-1 raw input values per sequence.
        # TODO fix long input that uses inits states
        # all_inits = conv_state[state_ids, :, 1:]
        all_inits = None
        # for prefill with spec tokens
        if spec_conv_offsets is not None:
            selected_conv_state = conv_state[state_ids]
            spec_conv_offsets = spec_conv_offsets.unsqueeze(1).expand(-1, conv_state.size(1), -1)
            final_state = selected_conv_state.scatter_(2, spec_conv_offsets, final_state)

        conv_state = conv_state.index_copy_(0, state_ids, final_state)
        # note that we have not set init states
        x = x.transpose(-2, -1)
        out = self.causal_conv1d_fn(
            x,
            weight,
            bias,
            seq_idx=seq_idx,
            initial_states=all_inits,
            return_final_states=False,
            activation=self.activation,
        )
        out = out.transpose(-2, -1)

        return out, conv_state

    def conv1d_update(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        conv_state: torch.Tensor,
        conv_state_indices: torch.Tensor,
        cache_seqlens: torch.Tensor | None = None,
    ):
        if weight.dim() == 3:
            assert weight.size(1) == 1
            weight = weight[:, 0]
        batch_size = conv_state_indices.size(0)
        q_seqlen = x.size(1) // batch_size
        is_spec_decoding = q_seqlen != 1
        x = x.squeeze(0)
        if is_spec_decoding:
            x = x.unflatten(0, (batch_size, q_seqlen)).transpose(1, 2).contiguous()
        out = self.causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias,
            activation=self.activation,
            conv_state_indices=conv_state_indices,
            cache_seqlens=cache_seqlens,
        )
        if is_spec_decoding:
            out = out.transpose(1, 2).flatten(0, 1)
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
            conv_state_indices = gated_delta_meta.conv_state_indices
            return self.conv1d_update(x,
                                      weight,
                                      bias,
                                      conv_state,
                                      conv_state_indices,
                                      cache_seqlens=gated_delta_meta.cache_seqlens)
        return self.conv1d_func(x, weight, bias, conv_state, gated_delta_meta=gated_delta_meta)


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
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        gated_delta_meta: GatedDeltaMeta,
    ):
        """call."""
        is_decoding = gated_delta_meta.is_decoding
        cu_seqlens = gated_delta_meta.cu_seqlens
        state_ids = gated_delta_meta.state_ids
        spec_state_offsets = gated_delta_meta.spec_state_offsets
        cache_seqlens = gated_delta_meta.cache_seqlens

        if not is_decoding:
            core_attn_out, last_recurrent_state = self.impl.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                state_indices=state_ids,
                output_final_state=True,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
                spec_state_offsets=spec_state_offsets,
            )
        else:
            # qkvgb (1, seqlen, ...) -> (B, seqlen, ...)
            batch_size = state_ids.size(0)
            core_attn_out, last_recurrent_state = self.impl.fused_recurrent_gated_delta_rule(
                query[0].unflatten(0, (batch_size, -1)).contiguous(),
                key[0].unflatten(0, (batch_size, -1)).contiguous(),
                value[0].unflatten(0, (batch_size, -1)).contiguous(),
                g=g[0].unflatten(0, (batch_size, -1)).contiguous(),
                beta=beta[0].unflatten(0, (batch_size, -1)).contiguous(),
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                state_indices=state_ids,
                cache_seqlens=cache_seqlens,
            )
            # out (seqlen, B, ...) -> (1, seqlen * B, ...)
            core_attn_out = core_attn_out.flatten(0, 1).unsqueeze(0)
        return core_attn_out, last_recurrent_state


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
        return self.causal_conv1d_func(x, self.weight, self.bias, conv_state, gated_delta_meta=gated_delta_meta)


@record_function('gated_delta_load_state')
def load_state(past_key_value: tuple[torch.Tensor, torch.Tensor], gated_delta_meta: GatedDeltaMeta):
    """Load states from cache."""
    return past_key_value[:2]
