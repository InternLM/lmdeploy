# Copyright (c) OpenMMLab. All rights reserved.
"""Ascend implementations for LMDeploy's gated-delta BuildSpecs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


from ...gated_delta_rule import (
    GatedDeltaMeta,
    GatedDeltaMetaImpl,
    GatedDeltaRuleImpl,
)


class AscendGatedDeltaMeta(GatedDeltaMeta):
    """Metadata required by dlinfer's stateful Ascend GDN kernels."""

    def __init__(
        self,
        num_tokens: int,
        conv_kernel_size: int,
        state_ids: torch.Tensor,
        attn_metadata: Any,
    ) -> None:
        super().__init__(num_tokens, conv_kernel_size, state_ids, attn_metadata)
        self.max_q_seq_len = attn_metadata.max_q_seqlen
        self.is_multi_token_decoding = attn_metadata.is_multi_token_decoding
        self.has_initial_state = attn_metadata.has_initial_state
        self.spec_conv_offsets = getattr(attn_metadata, "spec_conv_offsets", None)
        self.spec_state_offsets = getattr(attn_metadata, "spec_state_offsets", None)
        self.cache_seqlens = getattr(attn_metadata, "cache_seqlens", None)
        self.state_ids = state_ids.clamp(0)
        self.conv_state_indices = self.state_ids.to(torch.int32)


class AscendGatedDeltaMetaImpl(GatedDeltaMetaImpl):
    """Build Ascend-specific GDN metadata without backend monkey patching."""

    def forward(
        self,
        num_tokens: int,
        conv_kernel_size: int,
        state_ids: torch.Tensor,
        attn_metadata: Any,
    ) -> GatedDeltaMeta:
        return AscendGatedDeltaMeta(
            num_tokens, conv_kernel_size, state_ids, attn_metadata
        )


class AscendGatedDeltaRuleImpl(GatedDeltaRuleImpl):
    """Dlinfer Ascend Triton implementation of the GDN rule."""

    def __init__(self) -> None:
        from dlinfer.vendor.ascend.triton_ops import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
            fused_sigmoid_gating_delta_rule_update,
        )

        self.chunk_func = chunk_gated_delta_rule
        self.recurrent_func = fused_recurrent_gated_delta_rule
        self.decode_func = (
            fused_sigmoid_gating_delta_rule_update
        )

    @staticmethod
    def _prepare_inputs(
        query: torch.Tensor,
        key: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log_exp: torch.Tensor,
        kv_ratio: int,
    ):
        if b.dim() == 4:
            beta = b.sigmoid().flatten(-2, -1)
            a = a.float().flatten(-2, -1)
        else:
            beta = b.sigmoid()
            a = a.float()
        g = a_log_exp.float() * F.softplus(a + dt_bias)
        # Ascend's fused kernels support GVA directly (HV may be larger
        # than H), so q/k must stay on the key-head layout. Repeating them
        # here would change Qwen3.5's GDN semantics.
        return query, key, g, beta

    def forward(
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
        kv_ratio: int,
        use_qk_l2norm_in_kernel: bool,
    ) -> torch.Tensor:
        meta = gated_delta_meta
        if meta.is_decoding:
            # The Ascend decode kernel consumes the original A_log parameter,
            # while LMDeploy's public API supplies -exp(A_log).
            a_log = torch.log((-a_log_exp).float())
            return self.decode_func(
                A_log=a_log,
                dt_bias=dt_bias,
                q=query,
                k=key,
                v=value,
                a=a.contiguous(),
                b=b.contiguous(),
                initial_state_source=recurrent_state,
                initial_state_indices=meta.state_ids,
                cu_seqlens=meta.cu_seqlens,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                softplus_beta=1.0,
                softplus_threshold=20.0,
            )

        query, key, g, beta = self._prepare_inputs(
            query, key, b, a, dt_bias, a_log_exp, kv_ratio
        )

        if meta.is_multi_token_decoding:
            state_slots = recurrent_state.size(1)
            flat_recurrent_state = recurrent_state.view(
                -1, *recurrent_state.shape[2:]
            )
            core_attn_out, _ = self.recurrent_func(
                q=query.contiguous(),
                k=key.contiguous(),
                v=value.contiguous(),
                g=g.contiguous(),
                beta=beta.contiguous(),
                initial_state=flat_recurrent_state,
                inplace_final_state=True,
                cu_seqlens=meta.cu_seqlens,
                cache_seqlens_rb=meta.cache_seqlens,
                state_ids_rb=meta.state_ids,
                num_state=state_slots,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            return core_attn_out

        if meta.spec_state_offsets is not None:
            read_slots = meta.spec_state_offsets[0]
            initial_state = (
                recurrent_state[meta.state_ids, read_slots]
                .transpose(-1, -2)
                .contiguous()
            )
        else:
            initial_state = recurrent_state[meta.state_ids]
        if meta.has_initial_state is not None:
            initial_state[~meta.has_initial_state, ...] = 0

        core_attn_out, last_recurrent_state = self.chunk_func(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=meta.cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

        if meta.spec_state_offsets is not None:
            write_slots = meta.spec_state_offsets[1]
            recurrent_state[meta.state_ids, write_slots] = (
                last_recurrent_state.transpose(-1, -2).to(recurrent_state.dtype)
            )
        else:
            recurrent_state[meta.state_ids] = last_recurrent_state.to(
                recurrent_state.dtype
            )
        return core_attn_out

    def chunk_gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        scale: float | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        output_final_state: bool = False,
        spec_state_offsets=None,
        transpose_state_layout: bool = False,
    ):
        return self.chunk_func(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def fused_recurrent_gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        scale: float | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        output_final_state: bool = False,
        cache_seqlens: torch.Tensor | None = None,
        transpose_state_layout: bool = False,
        **kwargs,
    ):
        return self.recurrent_func(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            inplace_final_state=True,
            cu_seqlens=kwargs.get("cu_seqlens"),
            cache_seqlens_rb=kwargs.get("cache_seqlens_rb", cache_seqlens),
            state_ids_rb=kwargs.get("state_ids_rb", state_indices),
            num_state=kwargs.get(
                "num_state",
                initial_state.size(1) if initial_state is not None and initial_state.dim() == 5 else 1,
            ),
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
