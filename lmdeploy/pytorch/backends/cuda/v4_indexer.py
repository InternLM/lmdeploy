# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.distributed as dist

from ..indexer import BaseV4Indexer, BaseV4IndexerBuilder, V4IndexerMetadata, V4IndexerOutput
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
        bsz = query.size(0)
        seqlen = query.size(1)

        total_lens = start_pos + seqlen
        num_index = torch.div(total_lens, self.compress_ratio, rounding_mode='floor')
        max_index = int(num_index.max().item()) if num_index.numel() > 1 else int(num_index.item())

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


class TritonV4IndexerBuilder(BaseV4IndexerBuilder):

    @staticmethod
    def build(index_topk: int, compress_ratio: int, world_size: int = 1) -> BaseV4Indexer:
        return TritonV4IndexerImpl(index_topk=index_topk, compress_ratio=compress_ratio, world_size=world_size)
