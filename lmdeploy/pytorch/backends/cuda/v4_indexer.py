# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.distributed as dist

from ..indexer import BaseV4Indexer, BaseV4IndexerBuilder, V4IndexerMetadata, V4IndexerOutput


def _build_prefix_positions(lengths: torch.Tensor, max_len: int):
    """Build `[0, ..., len-1]` positions padded with `-1`."""
    device = lengths.device
    if max_len == 0:
        empty = torch.empty((lengths.numel(), 0), dtype=torch.long, device=device)
        return empty, empty.bool()
    arange = torch.arange(max_len, device=device).unsqueeze(0)
    mask = arange < lengths.unsqueeze(1)
    positions = torch.where(mask, arange, arange.new_full((), -1))
    return positions, mask


class TritonV4IndexerImpl(BaseV4Indexer):

    def __init__(self, index_topk: int, compress_ratio: int, world_size: int = 1) -> None:
        super().__init__()
        self.index_topk = index_topk
        self.compress_ratio = compress_ratio
        self.world_size = world_size

    @staticmethod
    def _gather_cache_entries(cache: torch.Tensor, block_offsets: torch.Tensor, positions: torch.Tensor,
                              block_size: int):
        if positions.numel() == 0:
            return cache.new_empty((*positions.shape, cache.size(-1)))
        safe_positions = positions.clamp(min=0)
        block_idx = torch.div(safe_positions, block_size, rounding_mode='floor').long()
        max_block_idx = block_offsets.size(1)
        valid = (positions >= 0) & (block_idx < max_block_idx)
        safe_block_idx = block_idx.clamp(max=max_block_idx - 1)
        block_off = torch.remainder(safe_positions, block_size).long()
        phys_blocks = block_offsets.gather(1, safe_block_idx).long()
        gathered = cache[phys_blocks, block_off]
        return torch.where(valid.unsqueeze(-1), gathered, gathered.new_zeros(()))

    @staticmethod
    def _write_cache_entries(cache: torch.Tensor,
                             block_offsets: torch.Tensor,
                             batch_idx: torch.Tensor,
                             positions: torch.Tensor,
                             values: torch.Tensor,
                             block_size: int,
                             write_mask: torch.Tensor | None = None):
        if positions.numel() == 0:
            return
        block_idx = torch.div(positions, block_size, rounding_mode='floor')
        max_block_idx = block_offsets.size(1)
        valid = (positions >= 0) & (block_idx >= 0) & (block_idx < max_block_idx)
        if write_mask is not None:
            valid = valid & write_mask
        safe_block_idx = block_idx.clamp(min=0, max=max_block_idx - 1)
        safe_positions = positions.clamp(min=0)
        block_off = torch.remainder(safe_positions, block_size).long()
        phys_blocks = block_offsets[batch_idx, safe_block_idx].long()
        if not valid.any():
            return
        target = cache[phys_blocks, block_off]
        values = values.to(target.dtype)
        blend_mask = valid.view(-1, *([1] * (values.dim() - 1)))
        cache[phys_blocks, block_off] = torch.where(blend_mask, values, target)

    def forward_decode(self,
                       query: torch.Tensor,
                       weights: torch.Tensor,
                       index_kv_cache: torch.Tensor,
                       meta: V4IndexerMetadata,
                       block_size: int,
                       layer_id: int,
                       index_scratch: torch.Tensor) -> V4IndexerOutput:
        block_offsets = meta.block_offsets.long()
        valid_mask = meta.valid_mask
        start_pos = torch.where(valid_mask, meta.start_pos, meta.start_pos.new_zeros(()))
        bsz = query.size(0)

        max_index = index_scratch.size(1)
        if max_index == 0:
            empty = query.new_empty((bsz, 1, 0), dtype=torch.long)
            return V4IndexerOutput(indices_in_kvcache=empty, topk_length=start_pos.new_zeros((bsz, ),
                                                                                             dtype=torch.int32))

        num_index = torch.where(valid_mask,
                                torch.div(start_pos + 1, self.compress_ratio, rounding_mode='floor'),
                                start_pos.new_zeros(()))
        positions, pos_mask = _build_prefix_positions(num_index, max_index)
        index_scratch.copy_(
            self._gather_cache_entries(index_kv_cache[layer_id], block_offsets, positions, block_size))

        score = torch.einsum('bshd,btd->bsht', query, index_scratch)
        score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        score = score.masked_fill(~pos_mask.unsqueeze(1), float('-inf'))
        if self.world_size > 1:
            dist.all_reduce(score)

        topk_width = min(self.index_topk, max_index)
        topk = score.topk(topk_width, dim=-1)[1]
        topk_length = num_index.clamp(max=topk_width).to(torch.int32)
        valid_topk = torch.arange(topk_width, device=query.device).view(1, 1, -1)
        valid_topk = valid_topk < topk_length.view(-1, 1, 1)
        logical_topk = torch.where(valid_topk, topk, topk.new_full((), -1))

        safe_logical_topk = logical_topk.clamp(min=0)
        block_idx = torch.div(safe_logical_topk, block_size, rounding_mode='floor').long()
        phys_block = block_offsets.gather(1, block_idx.view(bsz, -1)).view_as(logical_topk).long()
        block_off = torch.remainder(safe_logical_topk, block_size).long()
        phys_indices = phys_block * block_size + block_off
        phys_indices = torch.where(logical_topk >= 0, phys_indices, phys_indices.new_full((), -1))
        return V4IndexerOutput(indices_in_kvcache=phys_indices, topk_length=topk_length)


class TritonV4IndexerBuilder(BaseV4IndexerBuilder):

    @staticmethod
    def build(index_topk: int, compress_ratio: int, world_size: int = 1) -> BaseV4Indexer:
        return TritonV4IndexerImpl(index_topk=index_topk, compress_ratio=compress_ratio, world_size=world_size)
