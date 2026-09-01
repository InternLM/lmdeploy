# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch

PLACEHOLDER_TOKEN_ID = -1


class RejectionSamplingImpl(ABC):
    """Backend implementation of speculative rejection sampling."""

    @abstractmethod
    def forward_greedy(
        self,
        target_token_ids: torch.LongTensor,
        draft_token_ids: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Verify an all-greedy batch.

        ``target_token_ids`` contains one target token for every draft
        position followed by the bonus token, with shape
        ``[batch_size, num_spec_tokens + 1]``.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        target_logits: torch.Tensor,
        draft_token_ids: torch.LongTensor,
        bonus_token_ids: torch.LongTensor,
        *,
        is_greedy: torch.Tensor | None,
        draft_probs: torch.Tensor | None = None,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Verify processed target logits against draft tokens.

        ``is_greedy`` is a per-request mask for a mixed batch and ``None``
        for an all-random batch. All-greedy batches use ``forward_greedy``.
        """
        raise NotImplementedError


class RejectionSamplingBuilder(ABC):
    """Build a device-specific rejection-sampling implementation."""

    @staticmethod
    @abstractmethod
    def build() -> RejectionSamplingImpl:
        """Build the implementation."""
        raise NotImplementedError
