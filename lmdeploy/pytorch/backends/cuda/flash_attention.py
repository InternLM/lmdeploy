# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from ..flash_attention import FlashAttentionBuilder, FlashAttentionImpl


class TritonFlashAttentionImpl(FlashAttentionImpl):
    """Triton flash attention implementation."""

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

        from lmdeploy.pytorch.kernels.cuda import flash_attn_varlen_func
        self.flash_attention_fwd = flash_attn_varlen_func

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
        out = self.flash_attention_fwd(
            query,
            key,
            value,
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_start_loc=kv_start_loc,
            kv_seqlens=kv_seqlens,
            max_seqlen_q=max_q_seqlen,
            window_size=self.sliding_window,
            softmax_scale=self.scale,
            softcap=self.logit_softcapping,
            causal=self.causal,
            kv_layout='shd',
        )

        return out


class TritonFlashAttentionBuilder(FlashAttentionBuilder):
    """Triton attention builder."""

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
        return TritonFlashAttentionImpl(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_dim=v_head_dim,
            causal=causal,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
        )
