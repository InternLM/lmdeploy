# Copyright (c) OpenMMLab. All rights reserved.
"""CUDA KDA backend composed from the public FLA operators.

KDA is distinct from LMDeploy's gated-delta rule, but its CUDA implementation
does not need copied GLM kernels. This adapter owns only LMDeploy cache/state
semantics and delegates convolution and recurrence to FLA.
"""

from typing import Any

import torch

from lmdeploy.pytorch.backends.kda import KdaBuilder, KdaImpl


def _select_state(state: torch.Tensor, metadata: Any) -> torch.Tensor:
    selected = state.index_select(0, metadata.state_ids.long())
    clear = ~metadata.valid_state
    if metadata.is_init is not None:
        clear = clear | metadata.is_init
    clear = clear.reshape(-1, *((1, ) * (state.ndim - 1)))
    return selected.masked_fill(clear, 0)


def _store_state(state: torch.Tensor, value: torch.Tensor,
                 metadata: Any) -> None:
    state_ids = metadata.state_ids.long()
    valid = metadata.valid_state.reshape(-1,
                                         *((1, ) * (state.ndim - 1)))
    previous = state.index_select(0, state_ids)
    stored = torch.where(valid, value.to(state.dtype), previous)
    state.index_copy_(0, state_ids, stored)


class CudaKdaImpl(KdaImpl):
    """KDA implemented by public FLA convolution/recurrence kernels."""

    def __init__(self):
        try:
            from fla.modules.conv.triton.ops import (
                causal_conv1d_fwd,
                causal_conv1d_update,
            )
            from fla.ops.kda import chunk_kda, fused_recurrent_kda
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                'GLM-5.3 KDA requires flash-linear-attention==0.5.2.'
            ) from exc
        self.causal_conv1d_fwd = causal_conv1d_fwd
        self.causal_conv1d_update = causal_conv1d_update
        self.chunk_kda = chunk_kda
        self.fused_recurrent_kda = fused_recurrent_kda

    def _conv(
        self,
        mixed_qkv: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
        conv_state: torch.Tensor,
        metadata: Any,
    ) -> torch.Tensor:
        selected_state = _select_state(conv_state, metadata)
        if conv_weight.dim() == 3:
            if conv_weight.size(1) != 1:
                raise ValueError(
                    'KDA depthwise convolution weight must have shape '
                    '[D, 1, K].')
            conv_weight = conv_weight.squeeze(1)
        if metadata.is_decoding:
            mixed_qkv, final_state = self.causal_conv1d_update(
                x=mixed_qkv,
                cache=selected_state,
                weight=conv_weight,
                bias=conv_bias,
                activation='silu',
            )
        else:
            mixed_qkv, final_state = self.causal_conv1d_fwd(
                x=mixed_qkv,
                weight=conv_weight,
                bias=conv_bias,
                residual=None,
                initial_state=selected_state,
                output_final_state=True,
                activation='silu',
                cu_seqlens=metadata.cu_seqlens,
            )
        _store_state(conv_state, final_state, metadata)
        return mixed_qkv

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        metadata: Any,
        num_heads: int,
        head_dim: int,
        lower_bound: float,
    ) -> torch.Tensor:
        if (metadata.spec_state_offsets is not None
                or (getattr(metadata, 'num_spec_tokens', 0) or 0) > 0):
            raise NotImplementedError(
                'GLM-5.3 KDA speculative state rollback is not implemented.')
        batch_size = metadata.state_ids.numel()
        if metadata.is_decoding:
            query_length = mixed_qkv.size(1) // batch_size
            if query_length != 1:
                raise NotImplementedError(
                    'GLM-5.3 KDA supports single-token autoregressive decode only.')

        mixed_qkv = self._conv(mixed_qkv, conv_weight, conv_bias,
                               conv_state, metadata)
        q, k, v = mixed_qkv.split(num_heads * head_dim, dim=-1)
        q = q.unflatten(-1, (num_heads, head_dim)).contiguous()
        k = k.unflatten(-1, (num_heads, head_dim)).contiguous()
        v = v.unflatten(-1, (num_heads, head_dim)).contiguous()
        raw_gate = raw_gate.unflatten(
            -1, (num_heads, head_dim)).contiguous()
        raw_beta = raw_beta.contiguous()
        selected_state = _select_state(recurrent_state, metadata)

        if metadata.is_decoding:
            def decode_view(x: torch.Tensor) -> torch.Tensor:
                return x.squeeze(0).unflatten(
                    0, (batch_size, 1)).contiguous()

            output, final_state = self.fused_recurrent_kda(
                q=decode_view(q),
                k=decode_view(k),
                v=decode_view(v),
                g=decode_view(raw_gate),
                beta=decode_view(raw_beta),
                A_log=a_log,
                dt_bias=dt_bias,
                initial_state=selected_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=lower_bound,
                state_v_first=True,
            )
            output = output.flatten(0, 1).unsqueeze(0)
        else:
            output, final_state = self.chunk_kda(
                q=q,
                k=k,
                v=v,
                g=raw_gate,
                beta=raw_beta,
                A_log=a_log,
                dt_bias=dt_bias,
                initial_state=selected_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=lower_bound,
                state_v_first=True,
                cu_seqlens=metadata.cu_seqlens,
            )
        _store_state(recurrent_state, final_state, metadata)
        return output


class CudaKdaBuilder(KdaBuilder):
    """Build the CUDA KDA implementation."""

    @staticmethod
    def build() -> KdaImpl:
        return CudaKdaImpl()
