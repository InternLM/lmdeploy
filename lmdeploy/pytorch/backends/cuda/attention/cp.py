# Copyright (c) OpenMMLab. All rights reserved.
"""CUDA DCP attention implementation and shared collectives."""

import torch

from lmdeploy.pytorch.backends.cp_utils import DCPPrefixChunk, get_dcp_local_causal_seq_lens

from .default import TritonAttentionImpl, TritonAttentionMetaBuilder, TritonAttentionMetadata


def gather_dcp_prefix_kv(local_kv: torch.Tensor, chunk: DCPPrefixChunk,
                         *, zero_padding: bool = False) -> torch.Tensor:
    """Gather a flattened prefix chunk and restore global token order.

    Each rank supplies [chunk.local_capacity, ...] with valid tokens packed by request. The result has
    [requests * chunk.size, ...] capacity and valid tokens packed according to chunk.cu_seqlens.
    Set zero_padding for consumers such as FA3 that may load masked V lanes: 0 * NaN can contaminate valid outputs.
    """
    from lmdeploy.pytorch.distributed import all_gather_into_tensor
    from lmdeploy.pytorch.kernels.cuda.dcp import reorder_dcp_prefill_kv

    gathered = local_kv.new_empty(chunk.kv_seqlens.numel() * chunk.size, *local_kv.shape[1:])
    all_gather_into_tensor(gathered, local_kv, group='dcp')
    context = torch.zeros_like(gathered) if zero_padding else torch.empty_like(gathered)
    reorder_dcp_prefill_kv(gathered, context, chunk_kv_seqlens=chunk.kv_seqlens,
                           kv_start_loc=chunk.cu_seqlens[:-1], local_lens=chunk.local_kv_seqlens)
    return context


def init_dcp_query_gather(num_heads: int, head_size: int):
    """Initialize one query workspace shared by attention layers in this DCP
    group."""
    from lmdeploy.pytorch.distributed import get_dist_manager

    group = get_dist_manager().current_context().dcp_group
    if group.communicator is not None and group.query_gather_workspace is None:
        width = num_heads * head_size * torch.distributed.get_world_size(group.gpu_group)
        group.query_gather_workspace = group.communicator.create_all_gather_workspace(
            width, device=torch.device('cuda'), dtype=torch.bfloat16)


def gather_dcp_query(query: torch.Tensor, *, dcp_world_size: int) -> torch.Tensor:
    """Gather [tokens, local_heads, dim] queries along the head axis.

    Consume the result on the same stream before the next query gather; the optimized path borrows a shared arena.
    """
    if dcp_world_size == 1:
        return query
    from lmdeploy.pytorch.distributed import get_dist_manager

    group = get_dist_manager().current_context().dcp_group
    output = group.communicator.all_gather(query.flatten(1), workspace=group.query_gather_workspace, copy_output=False)
    return output.view(query.size(0), -1, query.size(2))


def merge_dcp_attention(local_output: torch.Tensor,
                        local_lse: torch.Tensor,
                        valid_counts: torch.Tensor,
                        *,
                        dcp_world_rank: tuple[int, int]) -> torch.Tensor:
    """Merge normalized CUDA shard outputs and scatter heads back to each rank.

    Inputs are [tokens, gathered_heads, value_dim] outputs and natural-log [tokens, gathered_heads] LSE. Empty rows
    contribute zero. LSE and correction arithmetic use FP32; reduce-scatter uses the input output dtype.
    """
    dcp_world_size, dcp_rank = dcp_world_rank
    if dcp_world_size == 1:
        return local_output
    from lmdeploy.pytorch.distributed import all_gather_into_tensor, reduce_scatter_tensor
    from lmdeploy.pytorch.kernels.cuda.dcp import correct_dcp_attention_output, sanitize_dcp_lse

    local_lse = sanitize_dcp_lse(local_lse, valid_counts)
    gathered_lse = local_lse.new_empty(
        dcp_world_size * local_lse.size(0), local_lse.size(1))
    all_gather_into_tensor(gathered_lse, local_lse, group='dcp')
    gathered_lse = gathered_lse.view(dcp_world_size, *local_lse.shape)
    contribution = correct_dcp_attention_output(
        local_output, gathered_lse, dcp_rank=dcp_rank)

    num_heads = contribution.size(0)
    # The correction kernel writes [heads, tokens, dim] directly so the
    # head-sharded reduce-scatter needs no separate transpose/copy.
    assert num_heads % dcp_world_size == 0
    local_heads = num_heads // dcp_world_size
    scattered_output = contribution.new_empty(local_heads,
                                              contribution.size(1),
                                              contribution.size(2))
    reduce_scatter_tensor(scattered_output, contribution, group='dcp')
    return scattered_output.transpose(0, 1)


class DCPAttentionImpl(TritonAttentionImpl):
    """Keep TP query heads local in prefill and gather them for decode.

    Model configuration ensures that every rank in a DCP group holds the same KV head. Cached prefill gathers bounded
    K/V chunks; decode evaluates each context shard and reduce-scatters softmax-corrected outputs to the TP heads.
    """

    def __init__(self, num_heads: int, head_size: int, *, use_fa3: bool = False, **kwargs):
        super().__init__(num_heads=num_heads, head_size=head_size, **kwargs)
        self.use_fa3 = use_fa3
        if use_fa3:
            from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_varlen_func
            self._fa3_prefill = flash_attn_varlen_func
        init_dcp_query_gather(self.num_heads, self.head_size)

    def get_step_metadata_provider(self):
        """Use common DCP lengths without a separate kernel scheduler."""
        return TritonAttentionMetaBuilder()

    def supports_piecewise_cuda_graph(self) -> bool:
        return True

    def _decode(self, query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                metadata: TritonAttentionMetadata, k_scales_zeros: torch.Tensor = None,
                v_scales_zeros: torch.Tensor = None) -> torch.Tensor:
        """Evaluate interleaved shards and restore rank-local query heads."""
        query = gather_dcp_query(query, dcp_world_size=self.dcp_world_size)
        query_len = query.size(0) // metadata.q_seqlens.numel()
        local_lens = metadata.dcp_local_kv_seqlens
        blocks = metadata.block_offsets
        if query_len > 1:
            # A global query step need not advance this shard's causal length.
            local_lens = get_dcp_local_causal_seq_lens(
                metadata.kv_seqlens, query_len, (self.dcp_world_size, self.dcp_rank))
            blocks = blocks.repeat_interleave(query_len, dim=0)
        output, lse = self.paged_attention_fwd(
            query, k_cache, v_cache, cache_seqlens=local_lens,
            page_table=blocks, max_seqlen_q=1, softmax_scale=self.scale,
            softcap=self.logit_softcapping, return_lse=True, quant_policy=metadata.quant_policy,
            k_scales_zeros=k_scales_zeros, v_scales_zeros=v_scales_zeros)
        return merge_dcp_attention(output, lse, valid_counts=local_lens,
                                   dcp_world_rank=(self.dcp_world_size, self.dcp_rank))

    def _prefill_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *,
                           cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int,
                           causal: bool, return_lse: bool):
        """Evaluate one prefill partition with optional natural-log LSE."""
        kwargs = dict(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                      max_seqlen_q=max_q, max_seqlen_k=max_k,
                      softmax_scale=self.scale, softcap=max(0.0, self.logit_softcapping), causal=causal)
        if self.use_fa3:
            return self._fa3_prefill(query, key, value, return_lse=return_lse, **kwargs)
        return self.flash_attention_fwd(query, key, value, kv_layout='shd', return_lse=return_lse, **kwargs)

    def _gather_prefix(self, k_cache: torch.Tensor, v_cache: torch.Tensor,
                       metadata: TritonAttentionMetadata, chunk: DCPPrefixChunk,
                       dtype: torch.dtype, k_scales_zeros: torch.Tensor = None,
                       v_scales_zeros: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather and reorder separate K/V caches within the chunk budget."""
        local_k, local_v = self.flatten_kv_cache(
            k_cache, v_cache, chunk.local_kv_seqlens[self.dcp_rank],
            metadata.block_offsets[:, chunk.local_block_slice(k_cache.size(1))],
            start_loc=chunk.local_cu_seqlens[:-1], out_size=chunk.local_capacity,
            out_dtype=dtype, flatten_kv_layout='shd', quant_policy=metadata.quant_policy,
            k_scales_zeros=k_scales_zeros, v_scales_zeros=v_scales_zeros)
        return (gather_dcp_prefix_kv(local_k, chunk, zero_padding=True),
                gather_dcp_prefix_kv(local_v, chunk, zero_padding=True))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                k_cache: torch.Tensor, v_cache: torch.Tensor, attn_metadata: TritonAttentionMetadata,
                k_scales_zeros: torch.Tensor = None, v_scales_zeros: torch.Tensor = None,
                learnable_sink: torch.Tensor = None, inplace: bool = True, **kwargs) -> torch.Tensor:
        """Write owned KV tokens, then evaluate GQA prefill or decode."""
        assert learnable_sink is None, 'GQA DCP does not support attention sinks'
        max_q = (query.size(0) // attn_metadata.q_seqlens.numel()
                 if attn_metadata.is_decoding else self._get_max_q_seqlen(query, attn_metadata))
        if key is not None and value is not None:
            self._fill_kv_cache_impl(key, value, k_cache, v_cache, attn_metadata, max_q,
                                     k_scales_zeros=k_scales_zeros, v_scales_zeros=v_scales_zeros)
        if attn_metadata.is_decoding:
            return self._decode(query, k_cache, v_cache, attn_metadata,
                                k_scales_zeros=k_scales_zeros, v_scales_zeros=v_scales_zeros)

        assert key is not None and value is not None, 'GQA DCP prefill requires current K/V'
        chunks = attn_metadata.dcp_prefix_chunks
        result = self._prefill_attention(
            query, key, value, cu_q=attn_metadata.cu_seqlens_q, cu_k=attn_metadata.cu_seqlens_q,
            max_q=max_q, max_k=max_q, causal=True, return_lse=bool(chunks))
        if not chunks:
            return result
        from lmdeploy.pytorch.kernels.cuda.dcp import merge_attention_states
        output, lse = result
        del result
        for chunk in chunks:
            context_k, context_v = self._gather_prefix(k_cache, v_cache, attn_metadata, chunk, query.dtype,
                                                      k_scales_zeros=k_scales_zeros, v_scales_zeros=v_scales_zeros)
            chunk_output, chunk_lse = self._prefill_attention(
                query, context_k, context_v, cu_q=attn_metadata.cu_seqlens_q, cu_k=chunk.cu_seqlens,
                max_q=max_q, max_k=chunk.size, causal=False, return_lse=True)
            output, lse = merge_attention_states(output, lse, chunk_output, chunk_lse)
            del context_k, context_v, chunk_output, chunk_lse
        return output
