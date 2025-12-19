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
    # flash attn
    scheduler_metadata: torch.Tensor = None
    max_kv_seqlen: int = None


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
        world_size, rank = get_tp_world_rank('attn')
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


def _try_dynamic_compile(func, *args, **kwargs):
    """Try compile."""
    try:
        compiled_func = torch.compile(func, dynamic=True)
        compiled_func(*args, **kwargs)
        return compiled_func
    except Exception:
        return func


class NSAIndicesUpdater:
    """NSA indices updater.

    Flash MLA sparse attention requires different indice format for prefill and decoding. This module is used to update
    the indices to meet the requirements.
    """

    def __init__(self):
        self._update_decode_func = None
        self._update_prefill_func = None

    def _update_decode_impl(self, nsa_indices: torch.Tensor, block_offsets: torch.Tensor,
                            block_size: int) -> torch.Tensor:
        """Update for decode impl."""
        block_ids = nsa_indices // block_size
        block_ids = block_ids.clamp_min(0)
        block_ids = block_offsets.gather(1, block_ids)
        block_remain = nsa_indices % block_size
        ret = block_ids * block_size + block_remain
        ret[nsa_indices < 0] = -1
        return ret[:, None]

    def update_decode(self, nsa_indices: torch.Tensor, block_offsets: torch.Tensor, block_size: int) -> torch.Tensor:
        """Update for decode."""
        if self._update_decode_func is None:
            self._update_decode_func = _try_dynamic_compile(self._update_decode_impl, nsa_indices, block_offsets,
                                                            block_size)

        return self._update_decode_func(nsa_indices, block_offsets, block_size)

    def _update_prefill_impl(self, nsa_indices: torch.Tensor, q_seqlens: torch.Tensor, cu_seqlens_k: torch.Tensor):
        """Update for prefill impl."""
        num_tokens = nsa_indices.size(0)
        repeat_cu_seqlens_k = torch.repeat_interleave(cu_seqlens_k[:-1], q_seqlens, output_size=num_tokens)
        neg_mask = nsa_indices < 0
        nsa_indices = nsa_indices + repeat_cu_seqlens_k[:, None]
        nsa_indices[neg_mask] = -1
        return nsa_indices[:, None]

    def update_prefill(self, nsa_indices: torch.Tensor, q_seqlens: torch.Tensor, cu_seqlens_k: torch.Tensor):
        """Update for prefill."""
        if self._update_prefill_func is None:
            self._update_prefill_func = _try_dynamic_compile(self._update_prefill_impl, nsa_indices, q_seqlens,
                                                             cu_seqlens_k)

        return self._update_prefill_func(nsa_indices, q_seqlens, cu_seqlens_k)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def build():
        return NSAIndicesUpdater()


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

        from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache_blocked_fp8
        from lmdeploy.pytorch.kernels.cuda.flatten_kv_cache import flatten_kv_cache_mla_fp8
        self.flash_mla_with_kvcache = flash_mla.flash_mla_with_kvcache
        self.flash_mla_sparse_fwd = None
        self.fill_kv_cache_blocked_fp8 = fill_kv_cache_blocked_fp8
        self.flatten_kv_cache_mla_fp8 = flatten_kv_cache_mla_fp8
        assert num_kv_heads == 1, 'MLA requires num kv heads equal to 1'

        self.nsa_updater = NSAIndicesUpdater.build()

    def _get_flash_mla_sparse_fwd(self):
        if self.flash_mla_sparse_fwd is not None:
            return self.flash_mla_sparse_fwd

        try:
            import flash_mla
            self.flash_mla_sparse_fwd = flash_mla.flash_mla_sparse_fwd
            return self.flash_mla_sparse_fwd
        except Exception:
            logger.exception('Can not import flash_mla_sparse_fwd from flash_mla.')

    def flash_mla_decoding(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        nsa_indices: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ):
        """Flash mla decoding."""
        causal = self.causal
        kv_seqlens = attn_metadata.kv_seqlens
        block_offsets = attn_metadata.block_offsets
        is_fp8_kvcache = k_cache.dtype == torch.float8_e4m3fn

        q_seqlens = attn_metadata.q_seqlens
        batch_size = q_seqlens.size(0)
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        max_q_seqlen = max_q_seqlen // batch_size
        query = query.unflatten(0, (batch_size, max_q_seqlen))
        if kv_seqlens.dtype == torch.int64:
            kv_seqlens = kv_seqlens.to(torch.int32)

        # update nsa indice according to flash-mla requirement
        if nsa_indices is not None:
            block_size = k_cache.size(1)
            nsa_indices = self.nsa_updater.update_decode(nsa_indices, block_offsets, block_size)
            causal = False

        attn_output, _ = self.flash_mla_with_kvcache(query,
                                                     k_cache=k_cache,
                                                     block_table=block_offsets,
                                                     cache_seqlens=kv_seqlens,
                                                     head_dim_v=self.v_head_size,
                                                     softmax_scale=self.scale,
                                                     tile_scheduler_metadata=attn_metadata.tile_scheduler_metadata,
                                                     num_splits=attn_metadata.num_splits,
                                                     causal=causal,
                                                     is_fp8_kvcache=is_fp8_kvcache,
                                                     indices=nsa_indices)

        attn_output = attn_output.squeeze(1)
        return attn_output

    def flash_mla_prefill(self, query: torch.Tensor, flatten_k: torch.Tensor, nsa_indices: torch.Tensor,
                          attn_metadata: TritonAttentionMetadata) -> torch.Tensor:
        """Flash mla prefill, only used in sparse attention."""
        q_seqlens = attn_metadata.q_seqlens
        flash_mla_sparse_fwd = self._get_flash_mla_sparse_fwd()

        num_q_heads = query.size(1)
        # flash_mla_sparse_fwd requires query heads to be multiple of 64
        if num_q_heads % 64 != 0:
            query = torch.nn.functional.pad(query, (0, 0, 0, 64 - num_q_heads % 64))

        nsa_indices = self.nsa_updater.update_prefill(nsa_indices, q_seqlens, attn_metadata.cu_seqlens_k)
        output = flash_mla_sparse_fwd(
            query,
            flatten_k,
            nsa_indices,
            sm_scale=self.scale,
        )
        attn_output = output[0]
        attn_output = attn_output[:, :num_q_heads]
        return attn_output

    def flash_attn_triton(
        self,
        query: torch.Tensor,
        flatten_k: torch.Tensor,
        flatten_v: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ):
        """Triton flash attention, used if flash-attn is not available."""
        q_start_loc = attn_metadata.q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))

        q_shape = query.shape
        o_shape = q_shape[:-1] + (self.v_head_size, )
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

    def flash_attn_fa3(
        self,
        query: torch.Tensor,
        flatten_k: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ):
        """Flash attention 3, used if flash-attn 3 is available."""
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        kv_flatten_size = attn_metadata.kv_flatten_size
        causal = self.causal
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
            causal=causal,
            window_size=(-1, -1) if self.sliding_window is None else self.sliding_window,
            softcap=-1.0 if self.logit_softcapping is None else self.logit_softcapping,
        )
        return attn_output

    def run_flatten_kv_cache(self,
                             k_cache: torch.Tensor,
                             v_cache: torch.Tensor,
                             attn_metadata: TritonAttentionMetadata,
                             out_dtype: torch.dtype,
                             is_nsa: bool,
                             k_scales_zeros: torch.Tensor = None,
                             v_scales_zeros: torch.Tensor = None):
        """Flatten kv cache for prefill."""

        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        block_offsets = attn_metadata.block_offsets
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy
        is_fp8_kvcache = k_cache.dtype == torch.float8_e4m3fn
        BLOCK_BS = k_cache.size(1)

        # pad one more block to avoid invalid kv visit
        out_size = (_cdiv(kv_flatten_size, BLOCK_BS) * BLOCK_BS + BLOCK_BS)
        flatten_kv_layout = 'shd' if use_fa3 or is_nsa else 'hsd'
        if is_fp8_kvcache:
            flatten_k = self.flatten_kv_cache_mla_fp8(
                k_cache,
                kv_seqlens,
                block_offsets,
                start_loc=kv_start_loc,
                out_size=out_size,
                out_dtype=out_dtype,
                flatten_kv_layout=flatten_kv_layout,
            )
            flatten_v = flatten_k[..., :512]
        else:
            flatten_k, flatten_v = self.flatten_kv_cache(
                k_cache,
                v_cache,
                kv_seqlens,
                block_offsets,
                start_loc=kv_start_loc,
                out_size=kv_flatten_size if use_fa3 else out_size,
                out_dtype=out_dtype,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
                flatten_kv_layout=flatten_kv_layout,
            )

        return flatten_k, flatten_v

    def run_fill_kv_cache(self,
                          query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          k_cache: torch.Tensor,
                          v_cache: torch.Tensor,
                          attn_metadata: TritonAttentionMetadata,
                          k_scales_zeros: torch.Tensor = None,
                          v_scales_zeros: torch.Tensor = None):
        """Fill kv cache."""
        block_offsets = attn_metadata.block_offsets
        q_start_loc = attn_metadata.q_start_loc
        fill_q_start_loc = q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        fill_seqlens = q_seqlens
        kv_seqlens = attn_metadata.kv_seqlens
        quant_policy = attn_metadata.quant_policy
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        batch_size = q_seqlens.size(0)
        if attn_metadata.is_decoding:
            max_q_seqlen = max_q_seqlen // batch_size

        fill_max_q_seqlen = max_q_seqlen
        if attn_metadata.fill_seqlens is not None:
            fill_seqlens = attn_metadata.fill_seqlens
            fill_max_q_seqlen = key.numel() // (key.size(-1) * key.size(-2))
            fill_q_start_loc = fill_seqlens.cumsum(0) - fill_seqlens

        is_fp8_kvcache = k_cache.dtype == torch.float8_e4m3fn
        if is_fp8_kvcache:
            k_cache_scale = k_cache[..., 512:512 + 16].view(torch.float32)
            k_cache_nope = k_cache[..., :512]
            k_cache_pe = k_cache[..., 512 + 16:].view(key.dtype)
            self.fill_kv_cache_blocked_fp8(
                key[..., :512],
                None,
                k_cache_nope,
                None,
                k_cache_scale,
                None,
                cu_seqlen_q=attn_metadata.cu_seqlens_q,
                kv_seqlens=attn_metadata.kv_seqlens,
                max_q_seqlen=max_q_seqlen,
                block_offsets=block_offsets,
                group_size=128,
            )
            self.fill_kv_cache(
                key[..., 512:],
                None,
                k_cache_pe,
                None,
                fill_q_start_loc,
                fill_seqlens,
                kv_seq_length=kv_seqlens,
                max_q_seq_length=fill_max_q_seqlen,
                block_offsets=block_offsets,
            )
        else:
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
        nsa_indices: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """forward."""

        # check nsa
        is_fp8_kvcache = k_cache.dtype == torch.float8_e4m3fn
        is_nsa = nsa_indices is not None
        if is_nsa:
            assert is_fp8_kvcache

        # fill kv cache
        self.run_fill_kv_cache(
            query,
            key,
            value,
            k_cache,
            v_cache,
            attn_metadata,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
        )

        is_decoding = attn_metadata.is_decoding
        if is_decoding:
            attn_output = self.flash_mla_decoding(query, k_cache, nsa_indices, attn_metadata)
        else:
            flatten_k, flatten_v = self.run_flatten_kv_cache(k_cache,
                                                             v_cache,
                                                             attn_metadata,
                                                             out_dtype=query.dtype,
                                                             is_nsa=nsa_indices is not None,
                                                             k_scales_zeros=k_scales_zeros,
                                                             v_scales_zeros=v_scales_zeros)
            if is_nsa:
                attn_output = self.flash_mla_prefill(query, flatten_k, nsa_indices, attn_metadata)
            elif use_fa3:
                attn_output = self.flash_attn_fa3(query, flatten_k, attn_metadata)
            else:
                attn_output = self.flash_attn_triton(query, flatten_k, flatten_v, attn_metadata)

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
        q_start_loc = attn_metadata.q_start_loc
        fill_q_start_loc = q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        fill_seqlens = q_seqlens
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy
        batch_size = q_seqlens.size(0)

        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        if attn_metadata.is_decoding:
            max_q_seqlen = max_q_seqlen // batch_size

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
                q_shape = query.shape
                o_shape = q_shape[:-1] + (self.v_head_size, )
                attn_output = query.new_empty(o_shape)
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
