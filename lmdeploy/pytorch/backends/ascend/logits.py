# Copyright (c) 2024, DeepLink. All rights reserved.
import torch


def _process_bad_words_(
    scores: torch.Tensor,
    bad_words: torch.LongTensor,
    mask: torch.BoolTensor,
    filter_value: float = -99999.9999,
):
    """Process bad words."""
    # aclnnGather does not accept negative indices; replace invalid entries
    # (which are negative padding values) with 0 before gather/scatter.
    valid_bad_words = bad_words.where(mask, 0)
    filtered_scores = scores.gather(1, valid_bad_words)
    filtered_scores = mask.to(filtered_scores.dtype) * filter_value + filtered_scores
    scores.scatter_(1, valid_bad_words, filtered_scores)
    return scores
