# Copyright (c) OpenMMLab. All rights reserved.

import functools
from dataclasses import dataclass
from typing import Literal

import torch

from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.utils import get_logger

from ..attention import AttentionBuilder, AttentionImpl, AttentionMetadata

logger = get_logger('lmdeploy')

use_fa3 = False
try:
    # Now flash-attention only support FA3 for sm90a && cuda >= 12.3
    if (torch.cuda.get_device_capability()[0] == 9) and (torch.version.cuda >= '12.3'):
        import flash_attn_interface  # noqa: F401
        assert torch.ops.flash_attn_3 is not None
        use_fa3 = True
except Exception:
    logger.debug('For higher performance, please install FlashAttention-3 '
                 'https://github.com/Dao-AILab/flash-attention')


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

        from lmdeploy.pytorch.kernels.cuda import (alibi_paged_attention_fwd, fill_kv_cache, flash_attention_fwd,
                                                   flatten_kv_cache, paged_attention_fwd)

        self.fill_kv_cache = fill_kv_cache
        self.paged_attention_fwd = paged_attention_fwd
        self.alibi_paged_attention_fwd = alibi_paged_attention_fwd
        self.flatten_kv_cache = flatten_kv_cache
        self.flash_attention_fwd = flash_attention_fwd

        # for alibi attention
        world_size, rank = get_tp_world_rank()
        self.alibi_head_offset = self.num_heads * rank
        self.alibi_num_heads = self.num_heads * world_size
        self.block_sparse_size = block_sparse_size

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
        fill_q_start_loc = q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        fill_seqlens = q_seqlens
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy
        if attn_metadata.is_decoding:
            max_q_seqlen = self.block_sparse_size
        else:
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

        if self.alibi:
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
                sinks=learnable_sink,
            )
        else:
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
                sinks=learnable_sink,
                causal=self.causal,
                block_sparse_size=self.block_sparse_size,
            )

        return attn_output


@functools.lru_cache
def use_fa3_warning():
    if use_fa3:
        return True
    logger.warning('For higher performance, please install FlashAttention-3 '
                   'https://github.com/Dao-AILab/flash-attention')
    return False


class FlashMLAImpl(TritonAttentionImpl):

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
        **kwargs,
    ):
        assert sliding_window is None, 'sliding window not supported for FlashMLA'
        assert alibi is False, 'alibi not supported for FlashMLA'
        assert logit_softcapping is None, 'logit_softcapping not supported for FlashMLA'
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

        import flash_mla
        self.flash_mla_with_kvcache = flash_mla.flash_mla_with_kvcache
        assert num_kv_heads == 1, 'MLA requires num kv heads equal to 1'

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
        nsa_indices: torch.Tensor = None,
        **kwargs,
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
        if attn_metadata.is_decoding:
            max_q_seqlen = 1
        else:
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

        is_decoding = attn_metadata.is_decoding
        if is_decoding:
            query = query.unsqueeze(1)
            if kv_seqlens.dtype == torch.int64:
                kv_seqlens = kv_seqlens.to(torch.int32)
            causal = False if nsa_indices is not None else self.causal
            if nsa_indices is not None:
                nsa_indices = nsa_indices[:, None]
            attn_output, _ = self.flash_mla_with_kvcache(query,
                                                         k_cache=k_cache,
                                                         block_table=block_offsets,
                                                         cache_seqlens=kv_seqlens,
                                                         head_dim_v=self.v_head_size,
                                                         softmax_scale=self.scale,
                                                         tile_scheduler_metadata=attn_metadata.tile_scheduler_metadata,
                                                         num_splits=attn_metadata.num_splits,
                                                         causal=causal,
                                                         indices=nsa_indices)
            attn_output = attn_output.squeeze(1)
        else:
            BLOCK_BS = k_cache.size(1)
            # pad one more block to avoid invalid kv visit
            out_size = (_cdiv(kv_flatten_size, BLOCK_BS) * BLOCK_BS + BLOCK_BS)
            flatten_k, flatten_v = self.flatten_kv_cache(
                k_cache,
                v_cache,
                kv_seqlens,
                block_offsets,
                start_loc=kv_start_loc,
                out_size=kv_flatten_size if use_fa3 else out_size,
                out_dtype=query.dtype,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
                flatten_kv_layout='shd' if use_fa3 else 'hsd',
            )
            if use_fa3:
                q_rope = query[:, :, self.v_head_size:]
                q_nope = query[:, :, :self.v_head_size]
                k_rope = flatten_k.view(kv_flatten_size, self.num_kv_heads, -1)[:, :, self.v_head_size:]
                c_kv = flatten_k.view(kv_flatten_size, self.num_kv_heads, -1)[:, :, :self.v_head_size]
                from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_varlen_func
                attn_output = flash_attn_varlen_func(
                    q=q_rope,
                    k=k_rope,
                    v=c_kv,
                    qv=q_nope,
                    cu_seqlens_q=attn_metadata.cu_seqlens_q,
                    cu_seqlens_k=attn_metadata.cu_seqlens_k,
                    max_seqlen_q=max_q_seqlen,
                    max_seqlen_k=kv_flatten_size,
                    softmax_scale=self.scale,
                    causal=self.causal,
                    window_size=(-1, -1) if self.sliding_window is None else self.sliding_window,
                    softcap=-1.0 if self.logit_softcapping is None else self.logit_softcapping,
                )
            else:
                attn_output = query.new_empty(o_shape)
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
                    causal=self.causal,
                )
        return attn_output


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
        sliding_window: int = None,
        logit_softcapping: float = None,
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
        from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_varlen_func
        self.flash_attn_varlen_func_v3 = flash_attn_varlen_func

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
        q_start_loc = attn_metadata.q_start_loc
        fill_q_start_loc = q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        fill_seqlens = q_seqlens
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy
        if attn_metadata.is_decoding:
            max_q_seqlen = 1
        else:
            max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        fill_max_q_seqlen = max_q_seqlen
        if attn_metadata.fill_seqlens is not None:
            fill_seqlens = attn_metadata.fill_seqlens
            fill_max_q_seqlen = key.numel() // (key.size(-1) * key.size(-2))
            fill_q_start_loc = fill_seqlens.cumsum(0) - fill_seqlens
        is_decoding = attn_metadata.is_decoding
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


@functools.lru_cache
def _enable_fa3(alibi: bool, learnable_sink: bool, block_sparse_size: int):
    enable = not alibi and not learnable_sink and block_sparse_size == 1
    if enable and not use_fa3_warning():
        enable = False
    return enable


class TritonAttentionBuilder(AttentionBuilder[TritonAttentionMetadata]):
    """Triton attention builder."""

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
        causal: bool = True,
        use_flash_mla: bool = False,
        learnable_sink: bool = False,
        block_sparse_size: int = 1,
        **kwargs,
    ) -> TritonAttentionImpl:
        """build."""
        enable_fa3 = _enable_fa3(alibi, learnable_sink, block_sparse_size)
        if use_flash_mla is True:
            logger.debug('Build FlashMLAImpl Attention')
            return FlashMLAImpl(num_heads,
                                head_size,
                                scale=scale,
                                num_kv_heads=num_kv_heads,
                                v_head_size=v_head_size,
                                alibi=alibi,
                                sliding_window=sliding_window,
                                logical_softcapping=logical_softcapping,
                                causal=causal,
                                **kwargs)
        elif enable_fa3:
            logger.debug('Build FA3Impl Attention')
            return FA3Impl(num_heads,
                           head_size,
                           scale=scale,
                           num_kv_heads=num_kv_heads,
                           v_head_size=v_head_size,
                           alibi=alibi,
                           sliding_window=sliding_window,
                           logical_softcapping=logical_softcapping,
                           causal=causal,
                           **kwargs)
        else:
            logger.debug('Build TritonAttentionImpl Attention')
            return TritonAttentionImpl(num_heads,
                                       head_size,
                                       scale=scale,
                                       num_kv_heads=num_kv_heads,
                                       v_head_size=v_head_size,
                                       alibi=alibi,
                                       sliding_window=sliding_window,
                                       logical_softcapping=logical_softcapping,
                                       causal=causal,
                                       block_sparse_size=block_sparse_size,
                                       **kwargs)
