# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.kernels.cuda.bitonic_topk import bitonic_topk
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.pytorch.kernels.cuda.ds_index import fp8_index

from ..indexer import BaseV4Indexer, BaseV4IndexerBuilder, V4IndexerMetadata, V4IndexerOutput


class TritonV4IndexerImpl(BaseV4Indexer):

    def __init__(self, index_topk: int, compress_ratio: int) -> None:
        super().__init__()
        self.index_topk = index_topk
        self.compress_ratio = compress_ratio

    def _logical_to_physical(self, logical_topk: torch.Tensor, block_offsets: torch.Tensor,
                             block_size: int) -> torch.Tensor:
        """Convert logical compressed positions to physical KV cache
        indices."""
        bsz = logical_topk.size(0)
        safe_logical_topk = logical_topk.clamp(min=0)
        token_positions = safe_logical_topk * self.compress_ratio
        block_idx = torch.div(token_positions, block_size, rounding_mode='floor').long()
        max_block_idx = block_offsets.size(1)
        safe_block_idx = block_idx.clamp(max=max_block_idx - 1)
        block_idx_valid = block_idx < max_block_idx
        phys_block = block_offsets.gather(1, safe_block_idx.view(bsz, -1)).view_as(logical_topk).long()
        entries_per_block = block_size // self.compress_ratio
        block_off = torch.remainder(safe_logical_topk, entries_per_block).long()
        phys_indices = phys_block * entries_per_block + block_off
        valid = (logical_topk >= 0) & block_idx_valid
        return torch.where(valid, phys_indices, phys_indices.new_full((), -1))

    def forward(self,
                query: torch.Tensor,
                weights: torch.Tensor,
                index_kv_cache: torch.Tensor,
                index_kv_scale_cache: torch.Tensor,
                meta: V4IndexerMetadata,
                block_size: int,
                offset: int) -> V4IndexerOutput:
        block_offsets = meta.block_offsets.long()
        cu_q_seqlens = meta.cu_q_seqlens
        kv_seqlens = meta.kv_seqlens
        is_decoding = meta.is_decoding
        q_seqlens = cu_q_seqlens[1:] - cu_q_seqlens[:-1]
        bsz = kv_seqlens.size(0)

        # quant query
        # FP8 quantize Indexer Q (replaces fp4_act_quant for better precision)
        # we might need to do quant fp4 in the future
        q_2d = query.reshape(-1, query.size(-1) * query.size(-2))
        q_fp8, q_scale_2d = quant_fp8(q_2d, group_size=128,
                                       dtype=torch.float8_e4m3fn, scale_fmt='ue8m0')
        query = q_fp8.view_as(query)
        q_scale = q_scale_2d.view(query.shape[:-1])

        # reshape q and weights
        q_3d = query.flatten(0, 1)
        q_scale = q_scale.flatten(0, -2)
        weights = weights.flatten(0, -2)
        q_scale_weighted = q_scale * weights  # [bsz, n_heads]

        total_lens = kv_seqlens
        num_index = torch.div(total_lens, self.compress_ratio, rounding_mode='floor')
        max_kv_seqlen = meta.max_kv_seqlen if meta.max_kv_seqlen is not None else block_offsets.size(1) * block_size
        max_index = max(max_kv_seqlen // self.compress_ratio, 1)

        if max_index == 0:
            if is_decoding:
                empty = query.new_empty((1, bsz, 0), dtype=torch.long)
            else:
                empty = query.new_empty((bsz, 1, 0), dtype=torch.long)
            return V4IndexerOutput(indices_in_kvcache=empty,
                                   topk_length=num_index.new_zeros((bsz,), dtype=torch.int32))

        # fp8_index: fused scoring kernel
        # k_cache already sliced by layer_id: [num_blocks, entries_per_block, head_dim]
        # k_seqlens must be in compressed entries (not tokens) since BLOCK_N=entries_per_block
        k_cache = index_kv_cache
        k_s_cache = index_kv_scale_cache.squeeze(-1)  # [num_blocks, entries_per_block]
        scores = fp8_index(q_3d, q_scale_weighted,
                           k_cache, k_s_cache,
                           cu_q_seqlens, num_index.to(torch.int32), block_offsets,
                           max_q_seqlen=meta.max_q_seqlen, max_k_seqlen=max_index, causal=True)

        topk_width = min(self.index_topk, max_index)
        topk_length = num_index.clamp(max=topk_width).to(torch.int32)

        # bitonic_topk requires K to be a power of 2; fall back to torch.topk otherwise
        if topk_width > 0 and (topk_width & (topk_width - 1)) == 0:
            topk = bitonic_topk(scores, q_seqlens, num_index.to(torch.int32),
                                k=topk_width, fill=-1, descending=True).long()
        else:
            topk = scores.topk(topk_width, dim=-1)[1]

        if is_decoding:
            topk = topk.unsqueeze(1)  # [bsz, 1, topk_width]
            phys_indices = self._logical_to_physical(topk, block_offsets, block_size)
            return V4IndexerOutput(indices_in_kvcache=phys_indices, topk_length=topk_length)
        else:
            total_tokens = q_3d.size(0)

            # Apply per-token offset
            token_seq = torch.arange(total_tokens, device=q_3d.device)
            seq_id = torch.searchsorted(cu_q_seqlens[1:], token_seq, right=True)
            token_offset = torch.as_tensor(offset, device=seq_id.device)[seq_id].unsqueeze(-1)
            topk = topk + token_offset

            topk_length = num_index.clamp(max=topk_width).to(torch.int32)

            return V4IndexerOutput(indices_in_kvcache=topk.unsqueeze(0), topk_length=topk_length)


class TritonV4IndexerBuilder(BaseV4IndexerBuilder):

    @staticmethod
    def build(index_topk: int, compress_ratio: int) -> BaseV4Indexer:
        return TritonV4IndexerImpl(index_topk=index_topk, compress_ratio=compress_ratio)
