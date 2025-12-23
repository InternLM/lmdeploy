# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Literal

import torch

from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.utils import get_logger

from lmdeploy.pytorch.backends.attention import AttentionImpl, AttentionMetadata

logger = get_logger('lmdeploy')


@dataclass
class TritonAttentionMetadata(AttentionMetadata):
    """Triton attention metadata."""
    is_decoding: bool
    block_offsets: torch.Tensor
    q_start_loc: torch.Tensor = None
    q_seqlens: torch.Tensor = None
    kv_start_loc: torch.Tensor = None
    kv_seqlens: torch.Tensor = None
    fill_seqlens: torch.Tensor = None
    quant_policy: Literal[0, 4, 8] = 0
    kv_flatten_size: int = None
    # flash mla
    tile_scheduler_metadata: torch.Tensor = None
    num_splits: torch.Tensor = None
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    # flash attn
    scheduler_metadata: torch.Tensor = None
    max_kv_seqlen: int = None
    max_q_seqlen: int = None


def _cdiv(a, b):
    """Perform div up."""
    return (a + b - 1) // b


class TritonAttentionImpl(AttentionImpl[TritonAttentionMetadata]):
    """Triton attention implementation."""

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
        causal: bool = True,
        block_sparse_size: int = 1,
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
            causal=causal,
            **kwargs,
        )
        assert not (alibi and not causal)

        from lmdeploy.pytorch.kernels.cuda import (alibi_paged_attention_fwd, fill_kv_cache, flash_attn_varlen_func,
                                                   flash_attn_with_kvcache, flatten_kv_cache)

        self.fill_kv_cache = fill_kv_cache
        self.paged_attention_fwd = flash_attn_with_kvcache
        self.alibi_paged_attention_fwd = alibi_paged_attention_fwd
        self.flatten_kv_cache = flatten_kv_cache
        self.flash_attention_fwd = flash_attn_varlen_func

        # for alibi attention
        world_size, rank = get_tp_world_rank('attn')
        self.alibi_head_offset = self.num_heads * rank
        self.alibi_num_heads = self.num_heads * world_size
        self.block_sparse_size = block_sparse_size

    def _get_max_q_seqlen(self,
        query: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,) -> int:
        """get max q seqlen."""
        if attn_metadata.is_decoding:
            max_q_seqlen = self.block_sparse_size
        else:
            if attn_metadata.max_q_seqlen is not None:
                max_q_seqlen = attn_metadata.max_q_seqlen
            else:
                max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        return max_q_seqlen
    
    def _get_fill_meta(
        self,
        key: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
    ):
        """get fill meta."""
        fill_seqlens = attn_metadata.q_seqlens
        fill_max_q_seqlen = max_q_seqlen
        fill_q_start_loc = attn_metadata.q_start_loc
        # For MLlama only
        if attn_metadata.fill_seqlens is not None:
            fill_seqlens = attn_metadata.fill_seqlens
            fill_max_q_seqlen = key.numel() // (key.size(-1) * key.size(-2))
            fill_q_start_loc = fill_seqlens.cumsum(0) - fill_seqlens
        return fill_seqlens, fill_max_q_seqlen, fill_q_start_loc

    def _fill_kv_cache_impl(self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,):
        """fill kv cache."""
        kv_seqlens = attn_metadata.kv_seqlens
        block_offsets = attn_metadata.block_offsets
        quant_policy = attn_metadata.quant_policy

        # fill seqlen args
        fill_seqlens, fill_max_q_seqlen, fill_q_start_loc = self._get_fill_meta(
            key,
            attn_metadata,
            max_q_seqlen,
        )

        # fill kv cache
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
        learnable_sink: torch.Tensor = None,
        inplace: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """forward."""
        block_offsets = attn_metadata.block_offsets
        q_start_loc = attn_metadata.q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy
        max_q_seqlen = self._get_max_q_seqlen(query, attn_metadata)

        # fill kv cache
        if key is not None and value is not None:
            self._fill_kv_cache_impl(key, value,
                                     k_cache=k_cache, v_cache=v_cache,
                                     attn_metadata=attn_metadata,
                                     max_q_seqlen=max_q_seqlen,
                                     k_scales_zeros=k_scales_zeros,
                                     v_scales_zeros=v_scales_zeros)

        is_decoding = attn_metadata.is_decoding

        # for alibi attention, not optimized yet
        if self.alibi:
            q_shape = query.shape
            o_shape = q_shape[:-1] + (self.v_head_size, )
            attn_output = query.new_empty(o_shape)
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

        if is_decoding:
            # decoding
            attn_output = self.paged_attention_fwd(
                query,
                k_cache,
                v_cache,
                block_offsets,
                cu_seqlens_k_new=attn_metadata.cu_seqlens_k,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
                window_size=self.sliding_window,
                softmax_scale=self.scale,
                softcap=self.logit_softcapping,
                sinks=learnable_sink,
            )
        else:
            # prefilling
            BLOCK_BS = k_cache.size(1)
            # pad one more block to avoid invalid kv visit
            out_size = (_cdiv(kv_flatten_size, BLOCK_BS) * BLOCK_BS + BLOCK_BS)
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
            attn_output = self.flash_attention_fwd(
                query,
                flatten_k,
                flatten_v,
                cu_seqlens_q=attn_metadata.cu_seqlens_q,
                cu_seqlens_k=attn_metadata.cu_seqlens_k,
                max_seqlen_q=max_q_seqlen,
                window_size=self.sliding_window,
                softmax_scale=self.scale,
                softcap=self.logit_softcapping,
                sinks=learnable_sink,
                causal=self.causal,
                block_sparse_size=self.block_sparse_size,
            )

        return attn_output
