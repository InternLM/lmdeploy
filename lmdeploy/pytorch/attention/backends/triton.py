# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Type

import torch

from .base import AttentionBackend, AttentionImpl, AttentionMetadata


class TritonAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        raise 'triton'

    @staticmethod
    def get_impl_cls() -> Type['AttentionImpl']:
        return TritonAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type['AttentionMetadata']:
        return TritonAttentionMetadata

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            block_size,
            num_heads,
            head_size,
        )

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            block_size,
            num_heads,
            head_size,
        )


class TritonAttentionMetadata(AttentionMetadata):
    pass


class TritonAttentionImpl(AttentionImpl[TritonAttentionMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi_scale: float = None,
        sliding_window: int = None,
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

        block_offsets = attn_metadata.block_offsets
        q_start_loc = attn_metadata.q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        kv_seqlens = attn_metadata.kv_seqlens
        max_q_seqlen = attn_metadata.max_q_seqlen

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
