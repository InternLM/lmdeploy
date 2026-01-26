# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..embedding import EmbeddingBuilder, EmbeddingImpl


def get_masked_input_and_mask(input: torch.Tensor, start_index: int, end_index: int):
    input = input - start_index
    masked_input = input.clamp(0, end_index - start_index - 1)
    inv_vocab_mask = masked_input != input
    return masked_input, inv_vocab_mask


class DefaultEmbeddingImpl(EmbeddingImpl):
    """Embedding implementation api."""

    def __init__(self, start_index: int, end_index: int):
        self.start_index = start_index
        self.end_index = end_index

    def forward(self, x, weight: torch.Tensor, all_reduce: bool = False, group: dist.ProcessGroup = None):
        """forward."""
        if all_reduce:
            mask_input, inv_vocab_mask = get_masked_input_and_mask(x, self.start_index, self.end_index)
            out = F.embedding(mask_input, weight)
            out.masked_fill_(inv_vocab_mask.unsqueeze(-1), 0)
            dist.all_reduce(out, group=group)
        else:
            out = F.embedding(x, weight)

        return out


class DefaultEmbeddingBuilder(EmbeddingBuilder):
    """Embedding implementation builder."""

    @staticmethod
    def build(start_index: int, end_index: int):
        """build."""
        return DefaultEmbeddingImpl(start_index=start_index, end_index=end_index)
