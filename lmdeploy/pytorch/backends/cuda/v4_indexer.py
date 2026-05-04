# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.distributed as dist

from ..indexer import BaseV4Indexer, BaseV4IndexerBuilder, V4IndexerMetadata, V4IndexerOutput
from lmdeploy.pytorch.kernels.cuda.bitonic_topk import bitonic_topk
from lmdeploy.pytorch.models.deepseek_v4_utils import build_prefix_positions, gather_compressed_cache_entries


class TritonV4IndexerImpl(BaseV4Indexer):

    def __init__(self, index_topk: int, compress_ratio: int, world_size: int = 1) -> None:
        super().__init__()
        self.index_topk = index_topk
        self.compress_ratio = compress_ratio
        self.world_size = world_size

    def _logical_to_physical(self, logical_topk: torch.Tensor, block_offsets: torch.Tensor,
                             block_size: int, index_kv_cache: torch.Tensor) -> torch.Tensor:
        """Convert logical compressed positions to physical KV cache indices."""
        bsz = logical_topk.size(0)
        safe_logical_topk = logical_topk.clamp(min=0)
        token_positions = safe_logical_topk * self.compress_ratio
        block_idx = torch.div(token_positions, block_size, rounding_mode='floor').long()
        phys_block = block_offsets.gather(1, block_idx.view(bsz, -1)).view_as(logical_topk).long()
        page_size = index_kv_cache.size(1)
        block_off = torch.remainder(safe_logical_topk, page_size).long()
        phys_indices = phys_block * page_size + block_off
        return torch.where(logical_topk >= 0, phys_indices, phys_indices.new_full((), -1))

    def forward(self,
                query: torch.Tensor,
                weights: torch.Tensor,
                index_kv_cache: torch.Tensor,
                meta: V4IndexerMetadata,
                block_size: int,
                layer_id: int,
                index_scratch: torch.Tensor,
                offset: int,
                is_decoding: bool) -> V4IndexerOutput:
        block_offsets = meta.block_offsets.long()
        start_pos = meta.start_pos
        cu_q_seqlens = meta.cu_q_seqlens
        kv_seqlens = meta.kv_seqlens

        if cu_q_seqlens is not None and not is_decoding:
            return self._forward_prefill_batched(
                query, weights, index_kv_cache, block_offsets,
                cu_q_seqlens, kv_seqlens, layer_id, block_size, offset)

        bsz = query.size(0)
        seqlen = query.size(1)

        total_lens = start_pos + seqlen
        num_index = torch.div(total_lens, self.compress_ratio, rounding_mode='floor')
        max_index = max(block_offsets.size(1) * block_size // self.compress_ratio, 1)

        if index_scratch is not None:
            max_index = index_scratch.size(1)

        if max_index == 0:
            empty = query.new_empty((bsz, 1, 0), dtype=torch.long)
            return V4IndexerOutput(indices_in_kvcache=empty,
                                   topk_length=num_index.new_zeros((bsz,), dtype=torch.int32))

        positions, pos_mask = build_prefix_positions(num_index, max_index)
        if index_scratch is not None:
            index_scratch.copy_(
                gather_compressed_cache_entries(index_kv_cache[layer_id], block_offsets, positions,
                                                block_size, self.compress_ratio))
            gathered = index_scratch
        else:
            gathered = gather_compressed_cache_entries(index_kv_cache[layer_id], block_offsets, positions,
                                                       block_size, self.compress_ratio)

        score = torch.einsum('bshd,btd->bsht', query, gathered)
        score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        score = score.masked_fill(~pos_mask.unsqueeze(1), float('-inf'))
        if self.world_size > 1:
            dist.all_reduce(score)

        topk_width = min(self.index_topk, max_index)
        topk = score.topk(topk_width, dim=-1)[1]
        topk_length = num_index.clamp(max=topk_width).to(torch.int32)

        if is_decoding:
            valid_topk = torch.arange(topk_width, device=query.device).view(1, 1, -1)
            valid_topk = valid_topk < topk_length.view(-1, 1, 1)
            logical_topk = torch.where(valid_topk, topk, topk.new_full((), -1))
            phys_indices = self._logical_to_physical(logical_topk, block_offsets, block_size,
                                                     index_kv_cache)
            return V4IndexerOutput(indices_in_kvcache=phys_indices, topk_length=topk_length)
        else:
            return V4IndexerOutput(indices_in_kvcache=topk + offset, topk_length=topk_length)

    def _forward_prefill_batched(self,
                                 query: torch.Tensor,
                                 weights: torch.Tensor,
                                 index_kv_cache: torch.Tensor,
                                 block_offsets: torch.Tensor,
                                 cu_q_seqlens: torch.Tensor,
                                 kv_seqlens: torch.Tensor,
                                 layer_id: int,
                                 block_size: int,
                                 offset: int) -> V4IndexerOutput:
        """Batched prefill: all operations are token-wise on flat tensors.

        query is [1, total_tokens, n_heads, head_dim], gathered from paged
        cache as [bsz, max_index, head_dim], then expanded to
        [total_tokens, max_index, head_dim] via searchsorted.

        No padding, no CUDA sync, no .item() calls.
        """
        bsz = kv_seqlens.size(0)
        total_tokens = query.size(1)
        q_seqlens = cu_q_seqlens[1:] - cu_q_seqlens[:-1]

        # Per-token sequence id (GPU tensor, no sync)
        token_seq = torch.arange(total_tokens, device=query.device)
        seq_id = torch.searchsorted(cu_q_seqlens[1:], token_seq, right=True)

        total_lens = kv_seqlens
        num_index = torch.div(total_lens, self.compress_ratio, rounding_mode='floor')
        max_index = max(block_offsets.size(1) * block_size // self.compress_ratio, 1)

        if max_index == 0:
            empty = query.new_empty((1, total_tokens, 0), dtype=torch.long)
            return V4IndexerOutput(indices_in_kvcache=empty,
                                   topk_length=num_index.new_zeros((bsz,), dtype=torch.int32))

        # Gather compressed cache entries: [bsz, max_index, head_dim]
        positions, pos_mask = build_prefix_positions(num_index, max_index)
        gathered = gather_compressed_cache_entries(index_kv_cache[layer_id], block_offsets, positions,
                                                   block_size, self.compress_ratio)

        # Expand gathered from [bsz, max_index, head_dim] to [total_tokens, max_index, head_dim]
        gathered_flat = gathered[seq_id]  # [total_tokens, max_index, head_dim]

        # Score: einsum on flat tensors
        # q_flat: [T, n_heads, head_dim], gathered_flat: [T, max_index, head_dim]
        # result: [T, n_heads, max_index] = sum_d q[t,h,d] * gathered[t,m,d]
        q_flat = query.squeeze(0)
        score = torch.einsum('qhd,qmd->qhm', q_flat, gathered_flat)

        # ReLU * weights, sum over heads: [T, M]
        score = (score.relu_() * weights.squeeze(0).unsqueeze(-1)).sum(dim=1)

        # Mask invalid compression positions (per-token via seq_id)
        pos_mask_flat = pos_mask[seq_id]  # [total_tokens, max_index]
        score = score.masked_fill(~pos_mask_flat, float('-inf'))

        if self.world_size > 1:
            dist.all_reduce(score)

        # Topk using bitonic_topk (no sync, uses GPU q_seqlens/num_index)
        topk_width = min(self.index_topk, max_index)
        # bitonic_topk expects [num_tokens, max_kv_len], q_seqlens, kv_seqlens
        # For us: num_index is per-sequence, expand to per-token
        num_index_flat = num_index[seq_id]  # [total_tokens]
        topk = bitonic_topk(score, q_seqlens, num_index_flat, k=topk_width, fill=-1, descending=True)

        # Apply per-token offset
        # In prefill, offset is per-sequence [bsz]; expand to per-token via seq_id
        token_offset = torch.as_tensor(offset, device=seq_id.device)[seq_id].unsqueeze(-1)
        topk = topk + token_offset

        # topk_length: [bsz] -> [total_tokens] per-token
        topk_length = num_index.clamp(max=topk_width).to(torch.int32)

        return V4IndexerOutput(indices_in_kvcache=topk.unsqueeze(0), topk_length=topk_length)


class TritonV4IndexerBuilder(BaseV4IndexerBuilder):

    @staticmethod
    def build(index_topk: int, compress_ratio: int, world_size: int = 1) -> BaseV4Indexer:
        return TritonV4IndexerImpl(index_topk=index_topk, compress_ratio=compress_ratio, world_size=world_size)
