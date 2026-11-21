# Copyright (c) OpenMMLab. All rights reserved.
"""Ascend implementations for LMDeploy's causal-convolution BuildSpec."""

import torch

from ...causal_conv1d import CausalConv1dImpl
from ...gated_delta_rule import GatedDeltaMeta

PAD_SLOT_ID = -1


class AscendCausalConv1dImpl(CausalConv1dImpl):
    """Wrap dlinfer's Ascend Triton causal-convolution kernels."""

    def __init__(self) -> None:
        from dlinfer.vendor.ascend.triton_ops import (
            causal_conv1d_fn,
            causal_conv1d_update_npu,
        )
        from fla_npu.ops.ascendc import causal_conv1d_update as _ascendc_causal_conv1d_update

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update_npu
        self._ascendc_causal_conv1d_update = _ascendc_causal_conv1d_update

    def _t_weight(self, weight):
        """D6: weight.t().contiguous() per call cost a Transpose
        copy kernel per layer per decode step; the weight is static
        at inference, so cache keyed on tensor identity."""
        cache = getattr(self, "_t_weight_cache", None)
        if cache is None or cache[0] is not weight:
            cache = (weight, weight.t().contiguous())
            self._t_weight_cache = cache
        return cache[1]

    def _ascendc_decode_update(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str,
        conv_state_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run the shared AscendC decode-update kernel on LMDeploy's cache.

        Token-first state pools (``[cache, state_len, dim]`` per layer,
        GLM-5.3's layout) go straight to the kernel: it updates the indexed
        cache lines in place, and the aclnn descriptor carries the per-layer
        leading-stride gap of the pool view.  Legacy dim-first pools
        (``[cache, dim, state_len]``) keep the staged round trip below --
        the kernel rejects their transposed view (non-unit inner stride).
        """
        if conv_state.size(-1) == weight.size(-1):
            return self._ascendc_causal_conv1d_update(
                x,
                conv_state,
                weight.to(x.dtype),
                None if bias is None else bias.to(x.dtype),
                activation=activation,
                conv_state_indices=conv_state_indices.to(torch.int32),
                null_block_id=PAD_SLOT_ID,
            )
        state_view = conv_state.transpose(-1, -2)  # [cache, state_len, dim] view
        indices = conv_state_indices.to(torch.int64)
        valid = indices >= 0
        safe_indices = indices.clamp(min=0)
        packed = state_view.index_select(0, safe_indices).contiguous()
        local_indices = torch.where(
            valid,
            torch.arange(indices.numel(), device=indices.device, dtype=torch.int64),
            torch.full_like(indices, PAD_SLOT_ID, dtype=torch.int64),
        ).to(torch.int32)
        out = self._ascendc_causal_conv1d_update(
            x,
            packed,
            weight.to(x.dtype),
            None if bias is None else bias.to(x.dtype),
            activation=activation,
            conv_state_indices=local_indices,
            null_block_id=PAD_SLOT_ID,
        )
        # Scatter the staged update back into the dim-first cache.  This must
        # stay host-sync-free: ``valid.any()`` / ``nonzero`` D2H-sync and are
        # illegal under ``torch.npu.NPUGraph`` capture.  On Ascend
        # ``AscendGatedDeltaMeta`` clamps ``conv_state_indices`` non-negative
        # at construction, so ``valid`` is always all-true and ``safe_indices``
        # maps 1:1 to the staged ``packed`` rows -- the unconditional
        # ``index_copy_`` is exactly the previous conditional scatter, just
        # capture-safe.
        state_view.index_copy_(0, safe_indices, packed.to(state_view.dtype))
        return out

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

        if gated_delta_meta.is_decoding and not gated_delta_meta.is_multi_token_decoding:
            out = self._ascendc_decode_update(
                x,
                conv_state,
                self._t_weight(weight),
                bias,
                activation,
                gated_delta_meta.conv_state_indices,
            )
            return out.unsqueeze(0)

        if gated_delta_meta.is_decoding or gated_delta_meta.is_multi_token_decoding:
            out = self.update_fn(
                x,
                conv_state,
                self._t_weight(weight),
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
            # dlinfer's prefill reference expects [slot, channel, token],
            # matching LMDeploy's state-cache layout.
            conv_states=conv_state,
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
