# Copyright (c) OpenMMLab. All rights reserved.

import functools

import torch

from lmdeploy.utils import get_logger

from .default import TritonAttentionMetadata
from .mla import FlashMLAAttentionMetaBuilder, FlashMLAImpl

logger = get_logger('lmdeploy')


def _try_dynamic_compile(func, *args, **kwargs):
    """Try compile."""
    try:
        compiled_func = torch.compile(func, dynamic=True)
        compiled_func(*args, **kwargs)
        return compiled_func
    except Exception:
        return func


class FlashMLAIndexMapper:
    """Map logical DSA top-k indices to FlashMLA storage indices."""

    def __init__(self):
        self._map_decode_funcs = dict()
        self._map_prefill_func = None

    def _map_decode_impl(self, indices: torch.Tensor, block_offsets: torch.Tensor,
                         block_size: int, block_stride: int, token_stride: int,
                         index_stride: int, expand_block_offsets: bool) -> torch.Tensor:
        """Map logical decode indices to the selected cache layout."""
        batch_size = block_offsets.size(0)
        query_len = indices.size(0) // batch_size
        if expand_block_offsets:
            block_offsets = block_offsets[:, None, :].expand(-1, query_len, -1).flatten(0, 1)
        block_ids = indices // block_size
        block_ids = block_ids.clamp_min(0)
        block_ids = block_offsets.gather(1, block_ids)
        block_remain = indices % block_size
        mapped_indices = (block_ids * block_stride + block_remain * token_stride) // index_stride
        mapped_indices[indices < 0] = -1
        return mapped_indices.unflatten(0, (batch_size, query_len))

    def _map_decode(self, indices: torch.Tensor, block_offsets: torch.Tensor,
                    max_q_seqlen: int, block_size: int, block_stride: int,
                    token_stride: int, index_stride: int) -> torch.Tensor:
        """Dispatch a cached specialization for one decode layout."""
        expand_block_offsets = max_q_seqlen != 1
        key = (expand_block_offsets, block_size, block_stride, token_stride, index_stride)
        args = (indices, block_offsets, block_size, block_stride, token_stride,
                index_stride, expand_block_offsets)
        func = self._map_decode_funcs.get(key)
        if func is None:
            func = _try_dynamic_compile(self._map_decode_impl, *args)
            self._map_decode_funcs[key] = func
        return func(*args)

    def map_paged_decode(self, indices: torch.Tensor, block_offsets: torch.Tensor,
                         max_q_seqlen: int, block_size: int) -> torch.Tensor:
        """Map logical indices to paged-cache token offsets."""
        return self._map_decode(indices, block_offsets, max_q_seqlen, block_size,
                                block_stride=block_size, token_stride=1, index_stride=1)

    def map_strided_decode(self, indices: torch.Tensor, block_offsets: torch.Tensor,
                           max_q_seqlen: int, block_size: int, block_stride: int,
                           token_stride: int, index_stride: int) -> torch.Tensor:
        """Map logical indices to aligned offsets in a strided cache view."""
        return self._map_decode(indices, block_offsets, max_q_seqlen, block_size,
                                block_stride, token_stride, index_stride)

    def _map_flat_prefill_impl(self, indices: torch.Tensor, q_seqlens: torch.Tensor,
                               cu_seqlens_k: torch.Tensor):
        """Map request-local prefill indices into the flattened KV buffer."""
        num_tokens = indices.size(0)
        kv_offsets = torch.repeat_interleave(cu_seqlens_k[:-1], q_seqlens, output_size=num_tokens)
        invalid = indices < 0
        indices = indices + kv_offsets[:, None]
        indices[invalid] = -1
        return indices[:, None]

    def map_flat_prefill(self, indices: torch.Tensor, q_seqlens: torch.Tensor,
                         cu_seqlens_k: torch.Tensor):
        """Map request-local prefill indices into the flattened KV buffer."""
        if self._map_prefill_func is None:
            self._map_prefill_func = _try_dynamic_compile(self._map_flat_prefill_impl,
                                                          indices, q_seqlens, cu_seqlens_k)
        return self._map_prefill_func(indices, q_seqlens, cu_seqlens_k)

    @staticmethod
    @functools.cache
    def build():
        """Return the process-local mapper shared by all DSA layers."""
        return FlashMLAIndexMapper()


class FlashMLASparseImpl(FlashMLAImpl):
    """Sparse DSA attention using FlashMLA kernels.

    Prefill: dense MLA when top-k covers the sequence; otherwise
    ``flash_mla_sparse_fwd`` over flattened BF16 KV.
    Decode: ``flash_mla_sparse_fwd`` over a zero-copy BF16 cache view, or
    ``flash_mla_with_kvcache`` over the packed FP8 cache.
    """

    _MLA_HEAD_ALIGNMENT = 64
    _BF16_CACHE_INDEX_STRIDE = 64

    def __init__(self, mla_index_topk: int, **kwargs):
        super().__init__(**kwargs)
        self.mla_index_topk = mla_index_topk
        self.flash_mla_sparse_fwd = None
        self.index_mapper = FlashMLAIndexMapper.build()

    def get_step_metadata_provider(self):
        """Describe metadata required by sparse FlashMLA."""
        return FlashMLAAttentionMetaBuilder(num_attention_heads=self.num_heads,
                                            index_topk=self.mla_index_topk)

    def _get_flash_mla_sparse_fwd(self):
        if self.flash_mla_sparse_fwd is not None:
            return self.flash_mla_sparse_fwd

        try:
            import flash_mla
            self.flash_mla_sparse_fwd = flash_mla.flash_mla_sparse_fwd
            return self.flash_mla_sparse_fwd
        except Exception:
            logger.exception('Can not import flash_mla_sparse_fwd from flash_mla.')

    def _flash_mla_sparse_forward(self, query: torch.Tensor, indexed_kv: torch.Tensor,
                                  indices: torch.Tensor) -> torch.Tensor:
        """Run sparse FlashMLA over index-addressable BF16 KV storage."""
        flash_mla_sparse_fwd = self._get_flash_mla_sparse_fwd()
        num_q_heads = query.size(1)
        pad_heads = -num_q_heads % self._MLA_HEAD_ALIGNMENT
        if pad_heads:
            query = torch.nn.functional.pad(query, (0, 0, 0, pad_heads))

        attn_output = flash_mla_sparse_fwd(query, indexed_kv, indices, sm_scale=self.scale)[0]
        return attn_output[:, :num_q_heads]

    def _prefill_sparse(self, query: torch.Tensor, flatten_k: torch.Tensor,
                        nsa_indices: torch.Tensor,
                        attn_metadata: TritonAttentionMetadata) -> torch.Tensor:
        """Run sparse prefill over flattened BF16 KV."""
        indices = self.index_mapper.map_flat_prefill(nsa_indices,
                                                     attn_metadata.q_seqlens,
                                                     attn_metadata.cu_seqlens_k)
        return self._flash_mla_sparse_forward(query, flatten_k, indices)

    def _decoding_sparse_bf16(self, query: torch.Tensor, k_cache: torch.Tensor,
                              nsa_indices: torch.Tensor,
                              attn_metadata: TritonAttentionMetadata) -> torch.Tensor:
        """Run sparse decode over a zero-copy BF16 paged-cache view."""
        assert query.dtype == torch.bfloat16, 'BF16 sparse MLA requires a bfloat16 query'
        assert k_cache.dtype == torch.bfloat16, 'BF16 sparse MLA requires a bfloat16 KV cache'
        block_size = k_cache.size(1)
        max_q_seqlen = self._get_max_q_seqlen(query, attn_metadata)

        # Expose the paged cache in FlashMLA's aligned addressing units without
        # copying its full capacity.
        index_stride = self._BF16_CACHE_INDEX_STRIDE
        block_stride, token_stride = k_cache.stride()[:2]
        last_token_offset = ((k_cache.size(0) - 1) * block_stride
                             + (block_size - 1) * token_stride)
        storage_rows = last_token_offset // index_stride + 1
        storage_k = k_cache.as_strided((storage_rows, *k_cache.shape[2:]),
                                       (index_stride, *k_cache.stride()[2:]))
        indices = self.index_mapper.map_strided_decode(
            nsa_indices,
            attn_metadata.block_offsets,
            max_q_seqlen,
            block_size,
            block_stride,
            token_stride,
            index_stride,
        )
        indices = indices.flatten(0, 1)[:, None]
        return self._flash_mla_sparse_forward(query, storage_k, indices)

    def _decoding_sparse_fp8(self, query: torch.Tensor, k_cache: torch.Tensor,
                             nsa_indices: torch.Tensor,
                             attn_metadata: TritonAttentionMetadata) -> torch.Tensor:
        """Run sparse decode directly over the packed FP8 paged cache."""
        max_q_seqlen = query.size(0) // attn_metadata.q_seqlens.size(0)
        indices = self.index_mapper.map_paged_decode(
            nsa_indices,
            attn_metadata.block_offsets,
            max_q_seqlen,
            k_cache.size(1),
        )

        num_q_heads = query.size(1)
        scheduler_metadata, _ = self._get_scheduler_metadata(attn_metadata)
        if not isinstance(scheduler_metadata, torch.Tensor):
            pad_heads = -num_q_heads % self._MLA_HEAD_ALIGNMENT
            if pad_heads:
                query = torch.nn.functional.pad(query, (0, 0, 0, pad_heads))

        output = self._decoding_paged(query,
                                      k_cache,
                                      attn_metadata,
                                      indices=indices,
                                      causal=False)
        return output[:, :num_q_heads]

    def _forward_decoding(self, query: torch.Tensor, k_cache: torch.Tensor,
                          attn_metadata: TritonAttentionMetadata,
                          nsa_indices: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for sparse MLA decoding."""
        if nsa_indices is None:
            raise RuntimeError('Sparse MLA requires DSA top-k indices.')
        if k_cache.dtype == torch.float8_e4m3fn:
            return self._decoding_sparse_fp8(query, k_cache, nsa_indices, attn_metadata)
        return self._decoding_sparse_bf16(query, k_cache, nsa_indices, attn_metadata)

    def _forward_prefill(self,
                         query: torch.Tensor,
                         k_cache: torch.Tensor,
                         v_cache: torch.Tensor,
                         attn_metadata: TritonAttentionMetadata,
                         nsa_indices: torch.Tensor = None,
                         k_scales_zeros: torch.Tensor = None,
                         v_scales_zeros: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for sparse MLA prefill."""
        if attn_metadata.max_kv_seqlen <= self.mla_index_topk:
            # Top-k contains every valid key, so dense attention is equivalent
            # and avoids sparse index mapping and kernel overhead.
            return super()._forward_prefill(query,
                                            k_cache,
                                            v_cache,
                                            attn_metadata,
                                            nsa_indices=None,
                                            k_scales_zeros=k_scales_zeros,
                                            v_scales_zeros=v_scales_zeros)
        if nsa_indices is None:
            raise RuntimeError('Sparse MLA requires DSA top-k indices.')
        flatten_k, _ = self._flatten_prefill_kv_cache(
            k_cache,
            v_cache,
            attn_metadata,
            out_dtype=query.dtype,
            kv_layout='shd',
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
        )
        return self._prefill_sparse(query, flatten_k, nsa_indices, attn_metadata)
