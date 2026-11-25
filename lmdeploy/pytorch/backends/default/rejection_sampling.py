# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..rejection_sampling import (
    PLACEHOLDER_TOKEN_ID,
    RejectionSamplingImpl,
)


def _philox_uniform(seed, offset):
    """Portable Philox4x32-10 reference matching Triton's request RNG."""
    mask = 0xffffffff
    c0, c1 = offset & mask, (offset >> 32) & mask
    c2 = c3 = torch.zeros_like(c0)
    k0, k1 = seed & mask, (seed >> 32) & mask
    for _ in range(10):
        a, b = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c2 = ((b >> 32) & mask) ^ c1 ^ k0, ((a >> 32) & mask) ^ c3 ^ k1
        c1, c3 = b & mask, a & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    c0 = c0.to(torch.int32).long()
    return torch.where(c0 < 0, -c0 - 1, c0).float() * 4.6566127342e-10


def _rejection_randoms(target_logits, num_spec_tokens, random_seeds, random_offsets):
    batch_size, _, vocab_size = target_logits.shape
    device = target_logits.device
    if random_seeds is None:
        uniform = torch.rand((batch_size, num_spec_tokens), dtype=torch.float64, device=device)
        inv_q = torch.empty((batch_size, 1, vocab_size), dtype=torch.float32,
                            device=device).exponential_().reciprocal_()
        return uniform, inv_q
    positions = random_offsets[:, None] + torch.arange(num_spec_tokens, device=device)
    uniform = _philox_uniform(random_seeds[:, None] ^ (0xA0761D6478BD642F - (1 << 64)), positions)
    keys = (random_seeds[:, None] | (positions << 32)) ^ (0xE7037ED1A0B428DB - (1 << 64))
    recovered_uniform = _philox_uniform(keys[:, :, None], torch.arange(vocab_size, device=device))
    return uniform, -recovered_uniform.clamp_min(1.0e-7).log().reciprocal()


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
        random_seeds: torch.Tensor | None = None,
        random_offsets: torch.Tensor | None = None,
    ):
        target_token_ids = None
        greedy_accepted = None
        if is_greedy is not None:
            target_token_ids = target_logits.argmax(dim=-1)
            greedy_accepted = draft_token_ids == target_token_ids

        _, num_spec_tokens = draft_token_ids.shape
        target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
        uniform_probs, inv_q = _rejection_randoms(
            target_logits, num_spec_tokens, random_seeds, random_offsets)

        draft_indices = draft_token_ids.unsqueeze(-1)
        target_draft_probs = target_probs.gather(-1, draft_indices).squeeze(-1)
        if draft_probs is None:
            random_accepted = target_draft_probs >= uniform_probs
            recovered_scores = target_probs * inv_q
            recovered_scores.scatter_(-1, draft_indices, 0.0)
        else:
            draft_token_probs = draft_probs.gather(-1, draft_indices).squeeze(-1)
            random_accepted = (
                (draft_token_probs > 0)
                & (target_draft_probs / draft_token_probs >= uniform_probs)
            )
            recovered_scores = (
                (target_probs - draft_probs).clamp_min_(0)
                * inv_q
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
