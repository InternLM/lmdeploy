# Copyright (c) OpenMMLab. All rights reserved.
"""Ascend implementations for LMDeploy's causal-convolution BuildSpec."""

from __future__ import annotations


import torch

from ...causal_conv1d import CausalConv1dImpl
from ...gated_delta_rule import GatedDeltaMeta


class AscendCausalConv1dImpl(CausalConv1dImpl):
    """Wrap dlinfer's Ascend Triton causal-convolution kernels."""

    def __init__(self) -> None:
        from dlinfer.vendor.ascend.triton_ops import (
            causal_conv1d_fn,
            causal_conv1d_update_npu,
        )

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update_npu

    @staticmethod
    def _weight_2d(weight: torch.Tensor) -> torch.Tensor:
        if weight.dim() == 3:
            assert weight.size(1) == 1
            weight = weight[:, 0]
        return weight

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_state: torch.Tensor,
        gated_delta_meta: GatedDeltaMeta,
        activation: str,
    ) -> torch.Tensor:
        weight = self._weight_2d(weight)
        x = x.squeeze(0)

        if gated_delta_meta.is_decoding or gated_delta_meta.is_multi_token_decoding:
            out = self.update_fn(
                x,
                conv_state,
                weight.t().contiguous(),
                bias,
                activation=activation,
                conv_state_indices=gated_delta_meta.conv_state_indices,
                cache_seqlens=(
                    gated_delta_meta.cache_seqlens
                    if gated_delta_meta.is_multi_token_decoding
                    else None
                ),
                query_start_loc=(
                    gated_delta_meta.cu_seqlens
                    if gated_delta_meta.is_multi_token_decoding
                    else None
                ),
                max_query_len=(
                    gated_delta_meta.max_q_seq_len
                    if gated_delta_meta.is_multi_token_decoding
                    else -1
                ),
                validate_data=not gated_delta_meta.is_multi_token_decoding,
            )
            return out.unsqueeze(0)

        read_conv_offsets = None
        write_conv_offsets = None
        if gated_delta_meta.spec_conv_offsets is not None:
            read_conv_offsets, write_conv_offsets = gated_delta_meta.spec_conv_offsets

        out = self.causal_conv1d_fn(
            x.t(),
            weight,
            bias,
            activation=activation,
            conv_states=conv_state.transpose(1, 2),
            has_initial_state=gated_delta_meta.has_initial_state,
            cache_indices=gated_delta_meta.conv_state_indices,
            query_start_loc=gated_delta_meta.cu_seqlens,
            read_conv_offsets=read_conv_offsets,
            write_conv_offsets=write_conv_offsets,
        )
        return out.t().unsqueeze(0)

    def conv1d_fn(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        return_final_states: bool = False,
        activation: str | None = None,
    ):
        # This entry point is not used by Qwen3.5's cache-aware forward, but
        # implements the common CausalConv1dImpl contract for backend callers.
        return self.causal_conv1d_fn(
            x,
            self._weight_2d(weight),
            bias=bias,
            activation=activation,
            conv_states=initial_states,
            query_start_loc=seq_idx,
        )

    def update_fn(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        conv_state_indices: torch.Tensor | None = None,
        cache_seqlens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        max_query_len: int = -1,
        validate_data: bool = False,
    ):
        return self.causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices=conv_state_indices,
            cache_seqlens=cache_seqlens,
            query_start_loc=query_start_loc,
            max_query_len=max_query_len,
            validate_data=validate_data,
        )
