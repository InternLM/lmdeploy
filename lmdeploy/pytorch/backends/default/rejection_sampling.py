# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..rejection_sampling import (
    PLACEHOLDER_TOKEN_ID,
    RejectionSamplingBuilder,
    RejectionSamplingImpl,
)


def _format_rejection_outputs(
    selected_token_ids: torch.LongTensor,
    accepted: torch.Tensor,
    bonus_token_ids: torch.LongTensor,
):
    """Keep the accepted prefix, first replacement, and optional bonus."""
    batch_size, num_spec_tokens = selected_token_ids.shape
    prefix_before = torch.ones_like(accepted, dtype=torch.bool)
    if num_spec_tokens > 1:
        prefix_before[:, 1:] = accepted[:, :-1].cumprod(dim=1).bool()

    placeholder = selected_token_ids.new_full((), PLACEHOLDER_TOKEN_ID)
    output_token_ids = selected_token_ids.new_full(
        (batch_size, num_spec_tokens + 1),
        PLACEHOLDER_TOKEN_ID,
    )
    output_token_ids[:, :-1] = torch.where(
        prefix_before,
        selected_token_ids,
        placeholder,
    )
    all_accepted = accepted.all(dim=1)
    output_token_ids[:, -1] = torch.where(
        all_accepted,
        bonus_token_ids,
        placeholder,
    )

    num_accepted = prefix_before.sum(dim=1) + all_accepted
    num_rejected_tokens = num_spec_tokens + 1 - num_accepted
    last_indices = num_accepted - 1
    last_token_ids = output_token_ids.gather(1, last_indices[:, None]).flatten()
    return output_token_ids, num_rejected_tokens, last_token_ids


def torch_greedy_rejection_sample(
    target_token_ids: torch.LongTensor,
    draft_token_ids: torch.LongTensor,
):
    """Portable all-greedy longest-prefix verification."""
    target_draft_token_ids = target_token_ids[:, :-1]
    bonus_token_ids = target_token_ids[:, -1]
    accepted = draft_token_ids == target_draft_token_ids
    return _format_rejection_outputs(
        target_draft_token_ids,
        accepted,
        bonus_token_ids,
    )


class DefaultRejectionSamplingImpl(RejectionSamplingImpl):
    """Portable Torch rejection-sampling implementation."""

    def forward_greedy(
        self,
        target_token_ids: torch.LongTensor,
        draft_token_ids: torch.LongTensor,
    ):
        return torch_greedy_rejection_sample(target_token_ids, draft_token_ids)

    def forward(
        self,
        target_logits: torch.Tensor,
        draft_token_ids: torch.LongTensor,
        bonus_token_ids: torch.LongTensor,
        *,
        is_greedy: torch.Tensor | None,
        draft_probs: torch.Tensor | None = None,
    ):
        target_token_ids = None
        greedy_accepted = None
        if is_greedy is not None:
            target_token_ids = target_logits.argmax(dim=-1)
            greedy_accepted = draft_token_ids == target_token_ids

        batch_size, num_spec_tokens = draft_token_ids.shape
        vocab_size = target_logits.shape[-1]
        target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
        uniform_probs = torch.rand(
            (batch_size, num_spec_tokens),
            dtype=torch.float64,
            device=target_logits.device,
        )
        q = torch.empty(
            (batch_size, vocab_size),
            dtype=torch.float32,
            device=target_logits.device,
        )
        q.exponential_()
        inv_q = q.reciprocal()

        draft_indices = draft_token_ids.unsqueeze(-1)
        target_draft_probs = target_probs.gather(-1, draft_indices).squeeze(-1)
        if draft_probs is None:
            random_accepted = target_draft_probs >= uniform_probs
            recovered_scores = target_probs * inv_q[:, None, :]
            recovered_scores.scatter_(-1, draft_indices, 0.0)
        else:
            draft_token_probs = draft_probs.gather(-1, draft_indices).squeeze(-1)
            random_accepted = (
                (draft_token_probs > 0)
                & (target_draft_probs / draft_token_probs >= uniform_probs)
            )
            recovered_scores = (
                (target_probs - draft_probs).clamp_min_(0)
                * inv_q[:, None, :]
            )

        recovered_token_ids = recovered_scores.argmax(dim=-1)
        random_token_ids = torch.where(
            random_accepted,
            draft_token_ids,
            recovered_token_ids,
        )
        if is_greedy is None:
            selected_token_ids = random_token_ids
            accepted = random_accepted
        else:
            greedy_rows = is_greedy[:, None]
            selected_token_ids = torch.where(
                greedy_rows,
                target_token_ids,
                random_token_ids,
            )
            accepted = torch.where(
                greedy_rows,
                greedy_accepted,
                random_accepted,
            )

        return _format_rejection_outputs(
            selected_token_ids,
            accepted,
            bonus_token_ids,
        )


class DefaultRejectionSamplingBuilder(RejectionSamplingBuilder):
    """Build the portable Torch rejection sampler."""

    @staticmethod
    def build() -> RejectionSamplingImpl:
        return DefaultRejectionSamplingImpl()
