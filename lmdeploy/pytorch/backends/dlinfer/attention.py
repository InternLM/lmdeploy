# Copyright (c) OpenMMLab. All rights reserved.

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from torch import Tensor

from ..attention import AttentionBuilder, AttentionImpl, AttentionMetadata


@dataclass
class DlinferAttentionMetadata(AttentionMetadata):
    kv_start_indices: Optional[Tensor] = None
    block_size: int = 64
    attention_mask: Sequence[Tensor] = tuple()
    is_unpaged_prefill: Optional[bool] = None
    max_q_seq_len: int = 1
    max_kv_seq_len: int = 1
    quant_meta: Dict = None
    cu_seq_lens_kv: Optional[Tensor] = None


class DlinferAttentionImpl(AttentionImpl[DlinferAttentionMetadata]):
    """Dlinfer attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = None,
        sliding_window: int = None,
        logit_softcapping: float = None,
        causal: bool = True,
        **kwargs,
    ):
        assert causal
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            v_head_size,
            alibi,
            sliding_window,
            logit_softcapping,
            causal=causal,
            **kwargs,
        )

        from lmdeploy.pytorch.kernels.dlinfer import fill_kv_cache, paged_attention_fwd

        self.fill_kv_cache = fill_kv_cache
        self.paged_attention_fwd = paged_attention_fwd

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        attn_metadata: DlinferAttentionMetadata,
        k_scales_zeros: Tensor = None,
        v_scales_zeros: Tensor = None,
        learnable_sink: Tensor = None,
        nsa_indices: Tensor = None,
        inplace: bool = True,
    ) -> Tensor:
        """forward."""

        block_offsets = attn_metadata.block_offsets
        q_start_loc = attn_metadata.q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        kv_seqlens = attn_metadata.kv_seqlens
        is_decoding = attn_metadata.is_decoding
        kv_start_indices = attn_metadata.kv_start_indices
        block_size = attn_metadata.block_size
        attn_mask = attn_metadata.attention_mask
        is_unpaged_prefill = attn_metadata.is_unpaged_prefill
        max_q_seq_len = attn_metadata.max_q_seq_len
        max_kv_seq_len = attn_metadata.max_kv_seq_len
        quant_bits = attn_metadata.quant_policy
        cu_seq_lens_kv = attn_metadata.cu_seq_lens_kv

        if attn_metadata.quant_meta is not None:
            k_scales_zeros = [next(attn_metadata.quant_meta['k_scales']),
                              next(attn_metadata.quant_meta['k_zeros'])
                              ] if 'k_scales' in attn_metadata.quant_meta else []
            v_scales_zeros = [next(attn_metadata.quant_meta['v_scales']),
                              next(attn_metadata.quant_meta['v_zeros'])
                              ] if 'v_scales' in attn_metadata.quant_meta else []
            kv_scales = next(attn_metadata.quant_meta['kv_scales']) if 'kv_scales' in attn_metadata.quant_meta else None
            kv_zeros = next(attn_metadata.quant_meta['kv_zeros']) if 'kv_zeros' in attn_metadata.quant_meta else None
        else:
            k_scales_zeros = []
            v_scales_zeros = []
            kv_scales = None
            kv_zeros = None

        # fill kv cache
        k_cache, v_cache = self.fill_kv_cache(key,
                                              value,
                                              k_cache,
                                              v_cache,
                                              kv_start_indices,
                                              k_scales_zeros=k_scales_zeros,
                                              v_scales_zeros=v_scales_zeros,
                                              quant_bits=quant_bits)

        if inplace:
            attn_output = query[..., :self.v_head_size]
        else:
            q_shape = query.shape
            o_shape = q_shape[:-1] + (self.v_head_size, )
            attn_output = query.new_empty(o_shape)

        attn_output = self.paged_attention_fwd(
            query,
            key,
            value,
            attn_output,
            k_cache,
            v_cache,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_seqlens=kv_seqlens,
            cu_seq_lens_kv=cu_seq_lens_kv,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
            is_decoding=is_decoding,
            block_size=block_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            v_head_size=self.v_head_size,
            attn_mask=attn_mask,
            softmax_scale=self.scale,
            is_unpaged_prefill=is_unpaged_prefill,
            kv_scales=kv_scales,
            kv_zeros=kv_zeros,
            quant_bits=quant_bits,
        )

        return attn_output


class DlinferAttentionBuilder(AttentionBuilder[DlinferAttentionMetadata]):
    """Dlinfer attention builder."""

    @staticmethod
    def build(
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi_scale: float = None,
        sliding_window: int = None,
        logit_softcapping: float = None,
        causal: bool = True,
        learnable_sink: bool = False,
        **kwargs,
    ) -> DlinferAttentionImpl:
        """build."""
        return DlinferAttentionImpl(num_heads,
                                    head_size,
                                    scale=scale,
                                    num_kv_heads=num_kv_heads,
                                    v_head_size=v_head_size,
                                    alibi_scale=alibi_scale,
                                    sliding_window=sliding_window,
                                    logit_softcapping=logit_softcapping,
                                    causal=causal,
                                    **kwargs)
