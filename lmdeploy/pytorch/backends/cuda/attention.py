# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Literal

import torch

from lmdeploy.pytorch.distributed import get_world_rank

from ..attention import AttentionBuilder, AttentionImpl, AttentionMetadata


@dataclass
class TritonAttentionMetadata(AttentionMetadata):
    """triton attention metadata."""
    is_decoding: bool
    block_offsets: torch.Tensor
    q_start_loc: torch.Tensor = None
    q_seqlens: torch.Tensor = None
    kv_start_loc: torch.Tensor = None
    kv_seqlens: torch.Tensor = None
    fill_seqlens: torch.Tensor = None
    quant_policy: Literal[0, 4, 8] = 0
    kv_flatten_size: int = None


def _cdiv(a, b):
    """perform div up."""
    return (a + b - 1) // b


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
                                                   flash_attention_fwd,
                                                   flatten_kv_cache,
                                                   paged_attention_fwd)
        self.fill_kv_cache = fill_kv_cache
        self.paged_attention_fwd = paged_attention_fwd
        self.alibi_paged_attention_fwd = alibi_paged_attention_fwd
        self.flatten_kv_cache = flatten_kv_cache
        self.flash_attention_fwd = flash_attention_fwd

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
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        inplace: bool = True,
    ) -> torch.Tensor:
        """forward."""

        block_offsets = attn_metadata.block_offsets
        q_start_loc = attn_metadata.q_start_loc
        fill_q_start_loc = q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        fill_seqlens = q_seqlens
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        fill_max_q_seqlen = max_q_seqlen
        if attn_metadata.fill_seqlens is not None:
            fill_seqlens = attn_metadata.fill_seqlens
            fill_max_q_seqlen = key.numel() // (key.size(-1) * key.size(-2))
            fill_q_start_loc = fill_seqlens.cumsum(0) - fill_seqlens

        # fill kv cache
        if key is not None and value is not None:
            self.fill_kv_cache(
                key,
                value,
                k_cache,
                v_cache,
                fill_q_start_loc,
                fill_seqlens,
                kv_seq_length=kv_seqlens,
                max_q_seq_length=fill_max_q_seqlen,
                block_offsets=block_offsets,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
            )

        q_shape = query.shape
        o_shape = q_shape[:-1] + (self.v_head_size, )
        attn_output = query.new_empty(o_shape)

        is_decoding = attn_metadata.is_decoding
        if not self.alibi:
            if is_decoding:
                self.paged_attention_fwd(
                    query,
                    k_cache,
                    v_cache,
                    attn_output,
                    block_offsets,
                    kv_seqlens=kv_seqlens,
                    k_scales_zeros=k_scales_zeros,
                    v_scales_zeros=v_scales_zeros,
                    quant_policy=quant_policy,
                    window_size=self.sliding_window,
                    sm_scale=self.scale,
                    logit_softcapping=self.logit_softcapping,
                )
            else:
                BLOCK_BS = k_cache.size(1)
                # pad one more block to avoid invalid kv visit
                out_size = (_cdiv(kv_flatten_size, BLOCK_BS) * BLOCK_BS +
                            BLOCK_BS)
                flatten_k, flatten_v = self.flatten_kv_cache(
                    k_cache,
                    v_cache,
                    kv_seqlens,
                    block_offsets,
                    start_loc=kv_start_loc,
                    out_size=out_size,
                    out_dtype=query.dtype,
                    k_scales_zeros=k_scales_zeros,
                    v_scales_zeros=v_scales_zeros,
                    quant_policy=quant_policy,
                )
                self.flash_attention_fwd(
                    query,
                    flatten_k,
                    flatten_v,
                    attn_output,
                    q_start_loc=q_start_loc,
                    q_seqlens=q_seqlens,
                    kv_start_loc=kv_start_loc,
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
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
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
