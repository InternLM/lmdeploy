# Copyright (c) 2024, DeepLink. All rights reserved.
import torch


def _process_bad_words_(
    scores: torch.Tensor,
    bad_words: torch.LongTensor,
    mask: torch.BoolTensor,
    filter_value: float = -float('inf'),
):
    """Process bad words."""
    indices = bad_words.where(mask, 0)
    is_zero = indices == 0
    # Padding scatters to token 0 so all those writes must agree with its
    # actual ban status. Keep the scatter tensors fixed-size for graph replay.
    ban_zero = (is_zero & mask).any(dim=1, keepdim=True)
    zero_scores = scores[:, :1].masked_fill(ban_zero, filter_value)
    updates = torch.where(is_zero, zero_scores, filter_value)
    scores.scatter_(1, indices, updates)
    return scores
