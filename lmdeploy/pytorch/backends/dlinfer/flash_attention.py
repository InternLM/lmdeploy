# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from ..flash_attention import FlashAttentionBuilder, FlashAttentionImpl


class DlinferFlashAttentionImpl(FlashAttentionImpl):
    """Dlinfer flash attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_dim: int = None,
        causal: bool = True,
        sliding_window: int = None,
        logit_softcapping: float = None,
    ):
        if scale is None:
            scale = 1.0 / (head_dim**0.5)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if v_head_dim is None:
            v_head_dim = head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.v_head_dim = v_head_dim
        self.causal = causal
        self.sliding_window = sliding_window
        self.logit_softcapping = logit_softcapping
        from lmdeploy.pytorch.kernels.dlinfer import flash_attention_fwd
        self.flash_attention_fwd = flash_attention_fwd

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                q_start_loc: Tensor,
                q_seqlens: Tensor,
                kv_start_loc: Tensor,
                kv_seqlens: Tensor,
                max_q_seqlen: int = None):
        """forward."""
        q_shape = query.shape
        o_shape = q_shape[:-1] + (self.v_head_dim, )
        out = query.new_empty(o_shape)
        self.flash_attention_fwd(
            query,
            key,
            value,
            out,
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_start_loc=kv_start_loc,
            kv_seqlens=kv_seqlens,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            max_q_seqlen=max_q_seqlen,
            window_size=self.sliding_window,
            sm_scale=self.scale,
            logit_softcapping=self.logit_softcapping,
            causal=self.causal,
        )
        return out


class DlinferFlashAttentionBuilder(FlashAttentionBuilder):
    """Dlinfer attention builder."""

    @staticmethod
    def build(
        num_heads: int,
        head_dim: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_dim: int = None,
        causal: bool = True,
        sliding_window: int = None,
        logit_softcapping: float = None,
        **kwargs,
    ) -> FlashAttentionImpl:
        """build."""
        return DlinferFlashAttentionImpl(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_dim=v_head_dim,
            causal=causal,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
        )
