# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
from typing import Optional

import torch


@dataclasses.dataclass
class UnmaskingMeta:
    strategy: Optional[str]
    block_sparse_size: int
    topk: int = 1
    threshold: float = 0


DLLM_MASKED = 0
DLLM_UNMASKED = 1
DLLM_CACHED = 2


class UnmaskingProcessor:

    def __init__(self, meta: UnmaskingMeta):
        self.meta = meta

    def _get_scores(self, logits: torch.Tensor, token_ids: torch.Tensor):
        """Get scores."""
        scores = logits.softmax(dim=-1)
        scores = scores.gather(-1, token_ids.unsqueeze(-1)).flatten()
        return scores

    def low_confidence_static(self, logits: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """static."""
        block_sparse_size = self.meta.block_sparse_size
        topk = min(self.meta.topk, block_sparse_size)
        scores = self._get_scores(logits, token_ids)
        is_masked = dllm_mask == DLLM_MASKED
        scores = torch.where(is_masked, scores, scores.new_zeros((1, )))

        scores = scores.view(-1, block_sparse_size)
        dllm_mask = dllm_mask.view(-1, block_sparse_size)
        _, indices = scores.topk(topk, dim=-1)
        dllm_unmasked = dllm_mask.scatter(-1, indices, DLLM_UNMASKED)

        is_masked = is_masked.view_as(dllm_mask)
        dllm_mask = torch.where(is_masked, dllm_unmasked, dllm_mask)
        return dllm_mask.flatten()

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """call."""
        strategy = self.meta.strategy
        if strategy is None:
            return dllm_mask

        # reshape to [num_blocks, block_sparse_size]
        block_sparse_size = self.meta.block_sparse_size
        dllm_mask = dllm_mask.unflatten(0, (-1, block_sparse_size))

        is_same = (dllm_mask == dllm_mask[:, :1]).all(dim=1)
        first_mask = dllm_mask[:, 0]

        # unmasked to cache
        is_block_unmasked = is_same & (first_mask == DLLM_UNMASKED)
        dllm_mask[is_block_unmasked] = DLLM_CACHED

        dllm_mask = dllm_mask.flatten()
        token_ids = torch.where(dllm_mask != DLLM_MASKED, input_ids, token_ids)
        if strategy == 'low_confidence_static':
            dllm_mask = self.low_confidence_static(logits, token_ids, dllm_mask)
        else:
            raise RuntimeError(f'strategy {strategy} not supported.')

        return dllm_mask, token_ids
