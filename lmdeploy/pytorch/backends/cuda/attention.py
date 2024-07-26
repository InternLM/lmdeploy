# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..attention import AttentionBuilder, AttentionImpl, AttentionMetadata


class TritonAttentionMetadata(AttentionMetadata):
    """triton attention metadata."""
    pass


class TritonAttentionImpl(AttentionImpl[TritonAttentionMetadata]):
    """triton attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi_scale: float = None,
        sliding_window: int = None,
        logical_softcapping: float = None,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            v_head_size,
            alibi_scale,
            sliding_window,
            **kwargs,
        )

        from lmdeploy.pytorch.kernels.cuda import (fill_kv_cache,
                                                   paged_attention_fwd)
        self.fill_kv_cache = fill_kv_cache
        self.paged_attention_fwd = paged_attention_fwd

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        inplace: bool = True,
    ) -> torch.Tensor:
        """forward."""

        block_offsets = attn_metadata.block_offsets
        q_start_loc = attn_metadata.q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        kv_seqlens = attn_metadata.kv_seqlens
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))

        # fill kv cache
        self.fill_kv_cache(
            key,
            value,
            k_cache,
            v_cache,
            q_start_loc,
            q_seqlens,
            kv_seq_length=kv_seqlens,
            max_q_seq_length=max_q_seqlen,
            block_offsets=block_offsets,
        )

        if inplace:
            attn_output = query[..., :self.v_head_size]
        else:
            q_shape = query.shape
            o_shape = q_shape[:-1] + (self.v_head_size, )
            attn_output = query.new_empty(o_shape)

        self.paged_attention_fwd(
            query,
            k_cache,
            v_cache,
            attn_output,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_seqlens=kv_seqlens,
            max_seqlen=max_q_seqlen,
            window_size=self.sliding_window,
            sm_scale=self.scale,
        )

        return attn_output


class TritonAttentionBuilder(AttentionBuilder[TritonAttentionMetadata]):
    """triton attention builder."""

    @staticmethod
    def build(
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi_scale: float = None,
        sliding_window: int = None,
        logical_softcapping: float = None,
        **kwargs,
    ) -> TritonAttentionImpl:
        """build."""
        return TritonAttentionImpl(num_heads,
                                   head_size,
                                   scale=scale,
                                   num_kv_heads=num_kv_heads,
                                   v_head_size=v_head_size,
                                   alibi_scale=alibi_scale,
                                   sliding_window=sliding_window,
                                   logical_softcapping=logical_softcapping,
                                   **kwargs)
