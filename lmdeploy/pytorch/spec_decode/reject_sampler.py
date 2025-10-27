# Copyright (c) OpenMMLab. All rights reserved.
import enum
from typing import Optional

import torch
from torch import LongTensor, Tensor, nn
from torch.profiler import record_function


class SamplePolicy(enum.Enum):
    """Sample policy."""

    ALL_GREEDY = enum.auto()


class RejectionSampler(nn.Module):

    def __init__(self, sample_policy: SamplePolicy = SamplePolicy.ALL_GREEDY):
        super().__init__()
        self.sample_policy = sample_policy

    def forward(
        self,
        target_logits: Tensor,
        draft_token_ids: LongTensor,
        bonus_token_ids: LongTensor,
        draft_probs: Optional[Tensor] = None,
    ):
        """forward
        Args:
            target_logits (Tensor): The logits of target model in shape of [batch_size, num_spec_tokens, vocab_size].
            draft_token_ids (LongTensor): The input draft tokens ishape of [batch_size, num_spec_tokens]
            bonus_token_ids (LongTensor): The bonus token ids in shape of [batch_size, 1].
            draft_probs (Tensor): The probability of draft model in shape of [batch_size, num_spec_tokens, vocab_size].
                Default to ``None``.
        """
        output_token_ids, num_rejected_tokens, last_token_ids = rejection_sample(
            target_logits,
            draft_token_ids,
            bonus_token_ids,
            draft_probs=draft_probs,
        )
        return output_token_ids, num_rejected_tokens, last_token_ids


@record_function('rejection_sample')
def rejection_sample(
    target_probs: Tensor,
    draft_token_ids: LongTensor,
    bonus_token_ids: LongTensor,
    sample_policy: SamplePolicy = SamplePolicy.ALL_GREEDY,
    draft_probs: Optional[Tensor] = None,
):
    """rejection sample
    Args:
        target_probs (Tensor):

    """
    assert draft_probs is None or draft_probs.is_contiguous()
    assert sample_policy == SamplePolicy.ALL_GREEDY, 'only support all greedy sampling policy'

    target_argmax_tokens = target_probs.argmax(dim=-1)
    return greedy_reject_sampler(draft_token_ids, target_argmax_tokens, bonus_token_ids)


def greedy_reject_sampler(draft_token_ids, target_token_ids, bonus_token_ids):
    """Greedy reject sampler
    1. keep targets tokens that are equal to draft tokens
    2. keep first not equal target tokens
    3. add bonus tokens if all equal
    Args:
        draft_token_ids: (batch_size, num_spec_tokens)
        target_token_ids: (batch_size, num_spec_tokens)
        bonus_token_ids: (batch_size, 1)
    Returns:
        output_token_ids: (batch_size, num_spec_tokens + 1)
    """
    masks = draft_token_ids == target_token_ids
    batch_size, num_spec_tokens = draft_token_ids.shape
    # check rest draft tokens
    range_data = torch.arange(num_spec_tokens, device=draft_token_ids.device)[None, :]
    equals = (masks.cumsum(dim=1) - 1) == range_data
    num_rejected_tokens = num_spec_tokens - equals.sum(dim=1)
    first_diff_indices = torch.argmin(equals.int(), dim=1, keepdim=True)
    keeps = range_data.repeat(batch_size, 1) <= first_diff_indices
    keeps = keeps | equals
    keep_token_ids = torch.where(keeps, target_token_ids, -1)
    # add bonus tokens
    keep_bonus_ids = torch.where(equals[:, -1:], bonus_token_ids, -1)
    output_token_ids = torch.cat([keep_token_ids, keep_bonus_ids], dim=1)
    # get last token ids
    last_indices = (torch.cat([keeps, equals[:, -1:]], dim=1).cumsum(dim=1) - 1)[:, -1].flatten()
    last_token_ids = output_token_ids[torch.arange(batch_size, device=draft_token_ids.device), last_indices]
    return output_token_ids, num_rejected_tokens, last_token_ids
