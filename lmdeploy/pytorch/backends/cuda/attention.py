# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.distributed as dist

from ..attention import AttentionBuilder, AttentionImpl, AttentionMetadata


def get_world_rank():
    """get current world size and rank."""
    world_size = 1
    rank = 0

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    return world_size, rank


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
        alibi: bool = False,
        sliding_window: int = None,
        logit_softcapping: float = None,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_size=v_head_size,
            alibi=alibi,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            **kwargs,
        )

        from lmdeploy.pytorch.kernels.cuda import (alibi_paged_attention_fwd,
                                                   fill_kv_cache,
                                                   paged_attention_fwd)
        self.fill_kv_cache = fill_kv_cache
        self.paged_attention_fwd = paged_attention_fwd
        self.alibi_paged_attention_fwd = alibi_paged_attention_fwd

        # for alibi attention
        world_size, rank = get_world_rank()
        self.alibi_head_offset = self.num_heads * rank
        self.alibi_num_heads = self.num_heads * world_size

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

        if not self.alibi:
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
                logit_softcapping=self.logit_softcapping,
            )
        else:
            self.alibi_paged_attention_fwd(
                query,
                k_cache,
                v_cache,
                attn_output,
                block_offsets,
                b_start_loc=q_start_loc,
                b_seq_len=q_seqlens,
                b_kv_seq_len=kv_seqlens,
                max_input_len=max_q_seqlen,
                head_offset=self.alibi_head_offset,
                num_heads=self.alibi_num_heads,
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
        alibi: bool = False,
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
                                   alibi=alibi,
                                   sliding_window=sliding_window,
                                   logical_softcapping=logical_softcapping,
                                   **kwargs)
