# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..embedding import EmbeddingBuilder, EmbeddingImpl


def get_masked_input_and_mask(input: torch.Tensor, start_index: int, end_index: int):
    vocab_mask = (input >= start_index) & (input < end_index)
    mask_input = (input - start_index) * vocab_mask
    return mask_input, vocab_mask


class DefaultEmbeddingImpl(EmbeddingImpl):
    """Embedding implementation api."""

    def __init__(self, start_index: int, end_index: int):
        self.start_index = start_index
        self.end_index = end_index

    def forward(self, x, weight: torch.Tensor, all_reduce: bool = False, group: dist.ProcessGroup = None):
        """forward."""
        if all_reduce:
            mask_input, vocab_mask = get_masked_input_and_mask(x, self.start_index, self.end_index)
            out = F.embedding(mask_input, weight)
            out.masked_fill_((~vocab_mask).unsqueeze(-1), 0)
        else:
            out = F.embedding(x, weight)

        if all_reduce:
            dist.all_reduce(out, group=group)

        return out


class DefaultEmbeddingBuilder(EmbeddingBuilder):
    """Embedding implementation builder."""

    @staticmethod
    def build(start_index: int, end_index: int):
        """build."""
        return DefaultEmbeddingImpl(start_index=start_index, end_index=end_index)
