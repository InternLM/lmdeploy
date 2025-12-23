# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.utils import get_logger

from .default import TritonAttentionImpl, TritonAttentionMetadata

logger = get_logger('lmdeploy')


class FA3Impl(TritonAttentionImpl):
    """Triton attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: tuple = None,
        logit_softcapping: float = 0.0,
        causal: bool = True,
        **kwargs,
    ):
        assert alibi is False, 'alibi not supported for FA3'
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
        from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
        self.flash_attn_varlen_func_v3 = flash_attn_varlen_func
        self.flash_attn_with_kvcache_v3 = flash_attn_with_kvcache

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
    ) -> torch.Tensor:
        """forward."""
        block_offsets = attn_metadata.block_offsets
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy
        max_q_seqlen = self._get_max_q_seqlen(query, attn_metadata)

        # fill kv cache
        if key is not None and value is not None:
            self._fill_kv_cache_impl(key,
                                     value,
                                     k_cache=k_cache,
                                     v_cache=v_cache,
                                     attn_metadata=attn_metadata,
                                     max_q_seqlen=max_q_seqlen,
                                     k_scales_zeros=k_scales_zeros,
                                     v_scales_zeros=v_scales_zeros)

        is_decoding = attn_metadata.is_decoding
        if is_decoding:
            # spec decoding
            if max_q_seqlen > 1:
                sliding_window = (-1, -1) if self.sliding_window is None else self.sliding_window
                if isinstance(sliding_window, int):
                    sliding_window = (sliding_window, sliding_window)
                query = query.unflatten(0, (-1, max_q_seqlen))
                attn_output = self.flash_attn_with_kvcache_v3(
                    query,
                    k_cache,
                    v_cache,
                    cache_seqlens=attn_metadata.kv_seqlens.to(torch.int32),
                    max_seqlen_q=max_q_seqlen,
                    scheduler_metadata=attn_metadata.scheduler_metadata,
                    page_table=block_offsets,
                    softmax_scale=self.scale,
                    causal=self.causal,
                    window_size=sliding_window,
                    softcap=-1.0 if self.logit_softcapping is None else self.logit_softcapping,
                )
            else:
                attn_output = self.paged_attention_fwd(
                    query,
                    k_cache,
                    v_cache,
                    cache_seqlens=attn_metadata.kv_seqlens,
                    page_table=block_offsets,
                    cu_seqlens_q=attn_metadata.cu_seqlens_q,
                    max_seqlen_q=max_q_seqlen,
                    softmax_scale=self.scale,
                    softcap=self.logit_softcapping,
                    window_size=self.sliding_window,
                    # custom args
                    k_scales_zeros=k_scales_zeros,
                    v_scales_zeros=v_scales_zeros,
                    quant_policy=quant_policy,
                )
        else:
            flatten_k, flatten_v = self.flatten_kv_cache(
                k_cache,
                v_cache,
                kv_seqlens,
                block_offsets,
                start_loc=kv_start_loc,
                out_size=kv_flatten_size,
                out_dtype=query.dtype,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
                flatten_kv_layout='shd',
            )
            sliding_window = (-1, -1) if self.sliding_window is None else self.sliding_window
            if isinstance(sliding_window, int):
                sliding_window = (sliding_window, sliding_window)
            attn_output = self.flash_attn_varlen_func_v3(
                q=query,
                k=flatten_k,
                v=flatten_v,
                cu_seqlens_q=attn_metadata.cu_seqlens_q,
                cu_seqlens_k=attn_metadata.cu_seqlens_k,
                max_seqlen_q=max_q_seqlen,
                max_seqlen_k=kv_flatten_size,
                softmax_scale=self.scale,
                causal=self.causal,
                window_size=sliding_window,
                softcap=-1.0 if self.logit_softcapping is None else self.logit_softcapping,
            )
        return attn_output
