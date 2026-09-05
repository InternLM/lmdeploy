# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F

from lmdeploy.pytorch.model_inputs import get_step_ctx_manager


class GatedDeltaMeta:
    """Step metadata shared by causal-convolution and gated-delta ops."""

    def __init__(self, num_tokens: int, conv_kernel_size: int, state_ids: torch.Tensor, attn_metadata: Any):
        self.is_decoding = attn_metadata.is_decoding
        self.cu_seqlens = attn_metadata.cu_seqlens_q
        self.cache_seqlens = None
        self.num_spec_tokens = get_step_ctx_manager().build_ctx.num_spec_tokens
        self.spec_conv_offsets = None
        self.spec_state_offsets = None

        device = self.cu_seqlens.device

        seqlens = attn_metadata.q_seqlens
        batch_size = seqlens.numel()
        batch_idx = torch.arange(0, batch_size, dtype=torch.int32, device=device)
        self.seq_idx = torch.repeat_interleave(batch_idx, seqlens, output_size=num_tokens)[None]

        range_idx = torch.arange(-conv_kernel_size, 0, device=device)
        self.conv_idx = self.cu_seqlens[1:, None] + range_idx[None]
        # TODO: fix last chunk with less conv kernel tokens
        self.conv_idx = self.conv_idx.clamp_min(0)

        self.is_init = None
        self.is_init_token = None
        if not self.is_decoding:
            self.is_init = (attn_metadata.kv_seqlens - attn_metadata.q_seqlens) == 0
            self.is_init_token = self.is_init.new_zeros(num_tokens, dtype=torch.bool)
            self.is_init_token.scatter_(0, self.cu_seqlens[:-1].long(), self.is_init)

        if self.num_spec_tokens > 0:
            self.cache_seqlens = (attn_metadata.kv_seqlens - attn_metadata.q_seqlens).to(torch.int32)
            if not self.is_decoding:
                state_len = conv_kernel_size + self.num_spec_tokens
                read_conv_offsets = torch.remainder(self.cache_seqlens[:, None] + range_idx[1:][None], state_len)
                write_conv_offsets = torch.remainder(attn_metadata.kv_seqlens[:, None] + range_idx[None], state_len)
                self.spec_conv_offsets = (read_conv_offsets, write_conv_offsets)

                read_state_offsets = torch.remainder(self.cache_seqlens, 1 + self.num_spec_tokens)
                write_state_offsets = torch.remainder(attn_metadata.kv_seqlens, 1 + self.num_spec_tokens)
                self.spec_state_offsets = (read_state_offsets, write_state_offsets)

        self.conv_state_indices = state_ids.to(torch.int32)
        self.valid_state = state_ids >= 0
        self.origin_state_ids = state_ids
        self.state_ids = state_ids.clamp(0)


class GatedDeltaMetaImpl(ABC):
    """Gated-delta metadata construction API."""

    @abstractmethod
    def forward(
        self,
        num_tokens: int,
        conv_kernel_size: int,
        state_ids: torch.Tensor,
        attn_metadata: Any,
    ) -> GatedDeltaMeta:
        """Build metadata for one model forward."""
        raise NotImplementedError


class GatedDeltaMetaBuilder(ABC):
    """Gated-delta metadata implementation builder."""

    @staticmethod
    @abstractmethod
    def build() -> GatedDeltaMetaImpl:
        """Build the selected implementation."""
        raise NotImplementedError


class GatedDeltaRuleImpl(ABC):
    """Gated Delta Rule implementation api."""

    def prepare_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log_exp: torch.Tensor,
        kv_ratio: int,
        use_qk_l2norm_in_kernel: bool = False,
        is_decoding: bool = False,
        init_token_mask: torch.Tensor | None = None,
    ):
        """Prepare q/k/g/beta for gated delta rule."""
        if b.dim() == 4:
            beta = b.sigmoid().flatten(-2, -1)
            a = a.float().flatten(-2, -1)
        else:
            beta = b.sigmoid()
            a = a.float()
        g = a_log_exp * F.softplus(a + dt_bias)
        if not is_decoding and init_token_mask is not None:
            g = g.masked_fill(init_token_mask[None, :, None], -1.0e6)
        if kv_ratio > 1:
            q = q.repeat_interleave(kv_ratio, dim=-2)
            k = k.repeat_interleave(kv_ratio, dim=-2)
        return q, k, g, beta, False

    def prefill(
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
        """Run the prefill rule and update the recurrent-state cache."""
        meta = gated_delta_meta
        query, key, g, beta, qk_l2norm_done = self.prepare_inputs(
            query,
            key,
            b,
            a,
            dt_bias,
            a_log_exp,
            kv_ratio,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            is_decoding=False,
            init_token_mask=meta.is_init_token,
        )
        core_attn_out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            state_indices=meta.state_ids,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel and not qk_l2norm_done,
            cu_seqlens=meta.cu_seqlens,
            spec_state_offsets=meta.spec_state_offsets,
            transpose_state_layout=True,
        )
        return core_attn_out

    def decode(
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
        """Run the recurrent decode rule and update its state cache."""
        meta = gated_delta_meta
        query, key, g, beta, qk_l2norm_done = self.prepare_inputs(
            query,
            key,
            b,
            a,
            dt_bias,
            a_log_exp,
            kv_ratio,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            is_decoding=True,
            init_token_mask=meta.is_init_token,
        )
        batch_size = meta.state_ids.size(0)
        core_attn_out, _ = self.fused_recurrent_gated_delta_rule(
            query[0].unflatten(0, (batch_size, -1)).contiguous(),
            key[0].unflatten(0, (batch_size, -1)).contiguous(),
            value[0].unflatten(0, (batch_size, -1)).contiguous(),
            g=g[0].unflatten(0, (batch_size, -1)).contiguous(),
            beta=beta[0].unflatten(0, (batch_size, -1)).contiguous(),
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel and not qk_l2norm_done,
            state_indices=meta.origin_state_ids,
            cache_seqlens=meta.cache_seqlens,
            transpose_state_layout=True,
        )
        return core_attn_out.flatten(0, 1).unsqueeze(0)

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
        """Dispatch gated-delta execution by inference phase."""
        if gated_delta_meta.is_decoding:
            return self.decode(
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
                use_qk_l2norm_in_kernel,
            )
        return self.prefill(
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
            use_qk_l2norm_in_kernel,
        )

    @abstractmethod
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
        spec_state_offsets: torch.Tensor | None = None,
        transpose_state_layout: bool = False,
    ):
        """forward."""
        raise NotImplementedError

    @abstractmethod
    def fused_recurrent_gated_delta_rule(self,
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
                                         transpose_state_layout: bool = False):
        """forward."""
        raise NotImplementedError


class GatedDeltaRuleBuilder(ABC):
    """Gated Delta Rule implementation builder."""

    @staticmethod
    @abstractmethod
    def build() -> GatedDeltaRuleImpl:
        """build."""
        raise NotImplementedError
