# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..rejection_sampling import (
    PLACEHOLDER_TOKEN_ID,
    RejectionSamplingImpl,
)


def rejection_sampling_random(seeds, offsets, num_spec_tokens, vocab_size):
    """Portable Philox4x32-10 reference for request-local rejection randomness.

    The counter is (draw index, decoding step, 0, 0). The key's upper word
    separates acceptance (1), recovery (2), and ordinary/bonus sampling (0).
    Recovery noise is shared across draft positions: only the first rejection
    is emitted, and its acceptance draws use an independent stream.
    """
    def uniform(count, stream):
        mask = 0xffffffff
        c0 = torch.arange(count, device=seeds.device, dtype=torch.long)[None, :]
        c1, c2, c3 = offsets[:, None] & mask, 0, 0
        k0, k1 = seeds[:, None] & mask, (seeds[:, None] >> 32) ^ stream
        for _ in range(10):
            product0, product2 = c0 * 0xD2511F53, c2 * 0xCD9E8D57
            c0, c1, c2, c3 = (((product2 >> 32) & mask) ^ c1 ^ k0, product2 & mask,
                              ((product0 >> 32) & mask) ^ c3 ^ k1, product0 & mask)
            k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
        # Use the open interval (0, 1), including after float32 rounding.
        return ((c0 >> 9).float() + .5) * (1. / (1 << 23))

    return uniform(num_spec_tokens, 1), -uniform(vocab_size, 2).log().reciprocal()


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
        target_probs: torch.Tensor,
        draft_token_ids: torch.LongTensor,
        bonus_token_ids: torch.LongTensor,
        *,
        random_seeds: torch.LongTensor,
        random_offsets: torch.LongTensor,
        is_greedy: torch.Tensor | None,
        draft_probs: torch.Tensor | None = None,
    ):
        target_token_ids = None
        greedy_accepted = None
        if is_greedy is not None:
            target_token_ids = target_probs.argmax(dim=-1)
            greedy_accepted = draft_token_ids == target_token_ids

        batch_size, num_spec_tokens = draft_token_ids.shape
        vocab_size = target_probs.shape[-1]
        uniform_probs, inv_q = rejection_sampling_random(random_seeds, random_offsets, num_spec_tokens, vocab_size)

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
