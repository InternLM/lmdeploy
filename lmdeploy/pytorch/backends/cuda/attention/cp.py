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


class DCPManager:
    """Own shared DCP gathers and the output combine that orders query reuse.

    Layers in one distributed context share this manager. Prepare workspaces during model construction, before CUDA
    graph capture, and close only after all consumers and graphs have retired.
    """

    def __init__(self, group):
        self.group = group.gpu_group
        self.communicator = group.communicator
        self.world_size = torch.distributed.get_world_size(self.group)
        self.rank = group.rank
        self._query_workspace = None
        self._candidate_workspace = None

    def prepare_query_gather(self, num_heads: int, head_size: int):
        """Prepare one query arena shared across attention layers."""
        if self._query_workspace is None:
            width = num_heads * head_size * self.world_size
            # combine() orders all ranks after attention consumes these queries,
            # so the next query gather does not need another entry barrier.
            self._query_workspace = self.communicator.create_all_gather_workspace(
                width, device=torch.device('cuda'), dtype=torch.bfloat16, reuse_sync=False)

    def prepare_candidate_gather(self, topk: int):
        """Prepare one candidate arena shared across indexer layers."""
        if self._candidate_workspace is None:
            self._candidate_workspace = self.communicator.create_all_gather_workspace(
                topk * 2, device=torch.device('cuda'), dtype=torch.float32, dim=0)

    def gather_query(self, query: torch.Tensor) -> torch.Tensor:
        """Gather query heads; consume and combine on the same stream before
        reuse."""
        if self.world_size == 1:
            return query
        output = self.communicator.all_gather(
            query.flatten(1), workspace=self._query_workspace, copy_output=False)
        return output.view(query.size(0), -1, query.size(2))

    def gather_candidates(self, packed: torch.Tensor) -> torch.Tensor:
        """Gather [rows, topk, score/ID] payloads without changing their bits.

        The entry barrier protects the previous top-k consumer before arena reuse. Consume the borrowed output on the
        same stream before the next candidate gather.
        """
        output = self.communicator.all_gather(
            packed.flatten(1), dim=0, workspace=self._candidate_workspace, copy_output=False)
        return output.view(self.world_size, *packed.shape)

    def combine(self, local_output: torch.Tensor, local_lse: torch.Tensor,
                valid_counts: torch.Tensor) -> torch.Tensor:
        """Merge shard outputs and scatter heads, ordering the next query
        gather.

        Inputs are [tokens, gathered_heads, value_dim] outputs and natural-log [tokens, gathered_heads] LSE. Empty rows
        contribute zero. LSE and correction arithmetic use FP32; reduce-scatter uses the input output dtype.
        """
        if self.world_size == 1:
            return local_output
        from lmdeploy.pytorch.distributed import all_gather_into_tensor, reduce_scatter_tensor
        from lmdeploy.pytorch.kernels.cuda.dcp import correct_dcp_attention_output, sanitize_dcp_lse

        local_lse = sanitize_dcp_lse(local_lse, valid_counts)
        gathered_lse = local_lse.new_empty(
            self.world_size * local_lse.size(0), local_lse.size(1))
        all_gather_into_tensor(gathered_lse, local_lse, group=self.group)
        gathered_lse = gathered_lse.view(self.world_size, *local_lse.shape)
        contribution = correct_dcp_attention_output(
            local_output, gathered_lse, dcp_rank=self.rank)

        num_heads = contribution.size(0)
        # The correction kernel writes [heads, tokens, dim] directly so the
        # head-sharded reduce-scatter needs no separate transpose/copy.
        assert num_heads % self.world_size == 0
        local_heads = num_heads // self.world_size
        scattered_output = contribution.new_empty(local_heads,
                                                  contribution.size(1),
                                                  contribution.size(2))
        reduce_scatter_tensor(scattered_output, contribution, group=self.group)
        return scattered_output.transpose(0, 1)

    def close(self):
        """Release shared arenas before their process group is destroyed."""
        if self._query_workspace is not None:
            self._query_workspace.close()
            self._query_workspace = None
        if self._candidate_workspace is not None:
            self._candidate_workspace.close()
            self._candidate_workspace = None


def get_dcp_manager() -> DCPManager:
    """Get the context-owned manager during layer initialization."""
    from lmdeploy.pytorch.distributed import get_dist_manager

    context = get_dist_manager().current_context()
    if context.dcp_manager is None:
        context.dcp_manager = DCPManager(context.dcp_group)
    return context.dcp_manager


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
        self.dcp_manager = get_dcp_manager()
        self.dcp_manager.prepare_query_gather(self.num_heads, self.head_size)

    def get_step_metadata_provider(self):
        """Use common DCP lengths without a separate kernel scheduler."""
        return TritonAttentionMetaBuilder()

    def supports_piecewise_cuda_graph(self) -> bool:
        return True

    def _decode(self, query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                metadata: TritonAttentionMetadata, k_scales_zeros: torch.Tensor = None,
                v_scales_zeros: torch.Tensor = None) -> torch.Tensor:
        """Evaluate interleaved shards and restore rank-local query heads."""
        query = self.dcp_manager.gather_query(query)
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
        return self.dcp_manager.combine(output, lse, valid_counts=local_lens)

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
