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

    def _flash_mla_sparse_forward(
            self, query: torch.Tensor, indexed_kv: torch.Tensor,
            indices: torch.Tensor, return_lse: bool = False,
            topk_length: torch.Tensor = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run sparse FlashMLA over index-addressable BF16 KV storage."""
        flash_mla_sparse_fwd = self._get_flash_mla_sparse_fwd()
        num_q_heads = query.size(1)
        pad_heads = -num_q_heads % self._MLA_HEAD_ALIGNMENT
        if pad_heads:
            query = torch.nn.functional.pad(query, (0, 0, 0, pad_heads))

        flash_kwargs = dict(sm_scale=self.scale)
        if topk_length is not None:
            flash_kwargs['topk_length'] = topk_length
        results = flash_mla_sparse_fwd(query, indexed_kv, indices,
                                       **flash_kwargs)
        attn_output = results[0]
        attn_output = attn_output[:, :num_q_heads]
        if return_lse:
            softmax_lse = results[2]
            if softmax_lse.dim() == 3 and softmax_lse.size(-1) == 1:
                softmax_lse = softmax_lse.squeeze(-1)
            return attn_output, softmax_lse[:, :num_q_heads]
        return attn_output

    def _prefill_sparse(self, query: torch.Tensor, flatten_k: torch.Tensor,
                        nsa_indices: torch.Tensor,
                        attn_metadata: TritonAttentionMetadata) -> torch.Tensor:
        """Run sparse prefill over flattened BF16 KV."""
        indices = self.index_mapper.map_flat_prefill(nsa_indices,
                                                     attn_metadata.q_seqlens,
                                                     attn_metadata.cu_seqlens_k)
        return self._flash_mla_sparse_forward(query, flatten_k, indices)

    def _map_dcp_prefill_partition(
            self, nsa_indices: torch.Tensor,
            attn_metadata: TritonAttentionMetadata,
            partition_starts: torch.Tensor,
            partition_cu_lens: torch.Tensor) -> torch.Tensor:
        """Map request-local global indices into one flattened partition."""
        num_tokens = nsa_indices.size(0)
        request_ids = torch.repeat_interleave(
            torch.arange(attn_metadata.q_seqlens.numel(),
                         device=nsa_indices.device,
                         dtype=attn_metadata.q_start_loc.dtype),
            attn_metadata.q_seqlens,
            output_size=num_tokens)
        starts = partition_starts[request_ids]
        ends = starts + (partition_cu_lens[1:] -
                         partition_cu_lens[:-1])[request_ids]
        valid = (nsa_indices >= starts[:, None]) & (nsa_indices < ends[:, None])
        mapped = (nsa_indices - starts[:, None] +
                  partition_cu_lens[:-1][request_ids, None])
        return torch.where(valid, mapped, -1).to(torch.int32)[:, None]

    def _prefill_sparse_partition(
            self, query: torch.Tensor, flatten_k: torch.Tensor,
            indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one sparse partition and neutralize rows without selected KV."""
        from lmdeploy.pytorch.backends.cp_utils import compact_valid_indices

        indices, valid_counts = compact_valid_indices(indices)
        valid_counts = valid_counts.flatten()
        output, lse = self._flash_mla_sparse_forward(query,
                                                     flatten_k,
                                                     indices,
                                                     return_lse=True,
                                                     topk_length=valid_counts)
        valid_rows = valid_counts > 0
        output.masked_fill_(~valid_rows[:, None, None], 0)
        lse = torch.where(valid_rows[:, None] & torch.isfinite(lse), lse,
                          torch.full_like(lse, -torch.inf)).contiguous()
        return output, lse

    def _prefill_sparse_dcp(
        self,
        query: torch.Tensor,
        current_key: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        nsa_indices: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run sparse prefill over bounded current and cached partitions."""
        from lmdeploy.pytorch.kernels.cuda.dcp import merge_attention_states

        prefix_lens = attn_metadata.kv_seqlens - attn_metadata.q_seqlens
        current_indices = self._map_dcp_prefill_partition(
            nsa_indices,
            attn_metadata,
            prefix_lens,
            attn_metadata.cu_seqlens_q,
        )
        output, output_lse = self._prefill_sparse_partition(
            query, current_key, current_indices)

        for chunk in attn_metadata.dcp_prefill_chunks:
            context_k, context_cu_lens = self._gather_dcp_prefill_context_chunk(
                k_cache,
                v_cache,
                attn_metadata,
                chunk,
                out_dtype=query.dtype,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
            )
            starts = torch.full_like(prefix_lens, chunk.start)
            context_indices = self._map_dcp_prefill_partition(
                nsa_indices,
                attn_metadata,
                starts,
                context_cu_lens,
            )
            context_output, context_lse = self._prefill_sparse_partition(
                query, context_k, context_indices)
            output, output_lse = merge_attention_states(
                output, output_lse, context_output, context_lse)
            del context_k, context_indices, context_output, context_lse
        return output.to(query.dtype)

    def _decoding_sparse_bf16(
            self, query: torch.Tensor, k_cache: torch.Tensor,
            nsa_indices: torch.Tensor,
            attn_metadata: TritonAttentionMetadata,
            return_lse: bool = False, topk_length: torch.Tensor = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        return self._flash_mla_sparse_forward(query,
                                              storage_k,
                                              indices,
                                              return_lse=return_lse,
                                              topk_length=topk_length)

    def _decoding_sparse_fp8(
            self, query: torch.Tensor, k_cache: torch.Tensor,
            nsa_indices: torch.Tensor,
            attn_metadata: TritonAttentionMetadata, return_lse: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
                                      causal=False,
                                      return_lse=return_lse)
        if return_lse:
            attn_output, softmax_lse = output
            return attn_output[:, :num_q_heads], softmax_lse[:, :num_q_heads]
        return output[:, :num_q_heads]

    def _forward_decoding(self,
                          query: torch.Tensor,
                          k_cache: torch.Tensor,
                          attn_metadata: TritonAttentionMetadata,
                          nsa_indices: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for sparse MLA decoding."""
        if nsa_indices is None:
            raise RuntimeError('Sparse MLA requires DSA top-k indices.')
        if self.dcp_world_size > 1:
            from lmdeploy.pytorch.backends.cp_utils import compact_dcp_local_indices

            dcp_world_rank = self.dcp_world_size, self.dcp_rank
            local_indices, local_counts = compact_dcp_local_indices(
                nsa_indices, dcp_world_rank)
            query = self._gather_dcp_query(query)
            if k_cache.dtype == torch.float8_e4m3fn:
                # FlashMLA V3.2 masks -1 indices but rejects dynamic
                # topk_length. The compacted indices retain their fixed width
                # and use -1 padding, so local_counts is needed only when
                # merging empty local shards below.
                local_output, local_lse = self._decoding_sparse_fp8(
                    query,
                    k_cache,
                    local_indices,
                    attn_metadata,
                    return_lse=True)
            else:
                local_output, local_lse = self._decoding_sparse_bf16(
                    query,
                    k_cache,
                    local_indices,
                    attn_metadata,
                    return_lse=True,
                    topk_length=local_counts)
            return self._merge_dcp_attention(local_output,
                                             local_lse,
                                             valid_rows=local_counts > 0)
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
                         v_scales_zeros: torch.Tensor = None,
                         current_key: torch.Tensor = None) -> torch.Tensor:
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
                                            v_scales_zeros=v_scales_zeros,
                                            current_key=current_key)
        if nsa_indices is None:
            raise RuntimeError('Sparse MLA requires DSA top-k indices.')
        if self.dcp_world_size > 1:
            if attn_metadata.dcp_prefill_chunks:
                return self._prefill_sparse_dcp(
                    query,
                    current_key,
                    k_cache,
                    v_cache,
                    nsa_indices,
                    attn_metadata,
                    k_scales_zeros=k_scales_zeros,
                    v_scales_zeros=v_scales_zeros,
                )
            flatten_k = current_key
        else:
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
