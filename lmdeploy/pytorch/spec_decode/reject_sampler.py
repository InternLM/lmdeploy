# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import LongTensor, Tensor, nn
from torch.profiler import record_function

from lmdeploy.pytorch.engine.logits_process import SamplingInputs

PLACEHOLDER_TOKEN_ID = -1


class RejectionSampler(nn.Module):
    """Rejection sampler for speculative decoding.

    Implements the rejection sampling algorithm from the speculative decoding paper (
    https://arxiv.org/abs/2211.17192).
    Supports both greedy (argmax)
    and random (probabilistic) rejection, with per-sequence greedy detection
    via sampling_inputs.top_k.
    """

    def forward(
        self,
        target_logits: Tensor,
        draft_token_ids: LongTensor,
        bonus_token_ids: LongTensor,
        sampling_inputs: SamplingInputs,
        draft_probs: Optional[Tensor] = None,
    ):
        """forward.

        Args:
            target_logits (Tensor): Processed target logits in shape of
                [batch_size, num_spec_tokens, vocab_size].
            draft_token_ids (LongTensor): The input draft tokens in shape of
                [batch_size, num_spec_tokens].
            bonus_token_ids (LongTensor): The bonus token ids in shape of
                [batch_size].
            sampling_inputs (SamplingInputs): Sampling parameters.
            draft_probs (Tensor): The probability of draft model in shape of
                [batch_size, num_spec_tokens, vocab_size]. Default to ``None``.
        """
        output_token_ids, num_rejected_tokens, last_token_ids = rejection_sample(
            target_logits,
            draft_token_ids,
            bonus_token_ids,
            sampling_inputs=sampling_inputs,
            draft_probs=draft_probs,
        )
        return output_token_ids, num_rejected_tokens, last_token_ids


@record_function('rejection_sample')
def torch_greedy_rejection_sample(
    target_probs: Tensor,
    draft_token_ids: LongTensor,
    bonus_token_ids: LongTensor,
    sampling_inputs: SamplingInputs = None,
    draft_probs: Optional[Tensor] = None,
):
    """Greedy reject sampler
    1. keep targets tokens that are equal to draft tokens
    2. keep first not equal target tokens
    3. add bonus tokens if all equal
    Args:
        target_probs: (batch_size, num_spec_tokens, vocab_size)
        draft_token_ids: (batch_size, num_spec_tokens)
        bonus_token_ids: (batch_size, 1)
    Returns:
        output_token_ids: (batch_size, num_spec_tokens + 1)
    """
    assert draft_probs is None or draft_probs.is_contiguous()
    if bonus_token_ids.ndim == 1:
        bonus_token_ids = bonus_token_ids.unsqueeze(-1)
    target_token_ids = target_probs.argmax(dim=-1)

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


def _extract_outputs(output_token_ids: Tensor, num_spec_tokens: int):
    """Extract num_rejected_tokens and last_token_ids from output_token_ids.

    Args:
        output_token_ids: [batch_size, num_spec_tokens + 1]
        num_spec_tokens: number of speculative tokens

    Returns:
        output_token_ids, num_rejected_tokens, last_token_ids
    """
    batch_size = output_token_ids.size(0)
    valid_mask = output_token_ids != PLACEHOLDER_TOKEN_ID
    num_accepted = valid_mask.sum(dim=1)
    num_rejected_tokens = num_spec_tokens + 1 - num_accepted
    last_token_ids = output_token_ids[torch.arange(batch_size, device=output_token_ids.device), num_accepted - 1]
    return output_token_ids, num_rejected_tokens, last_token_ids


@record_function('rejection_sample')
def rejection_sample(
    target_logits: Tensor,
    draft_token_ids: LongTensor,
    bonus_token_ids: LongTensor,
    sampling_inputs: SamplingInputs,
    draft_probs: Optional[Tensor] = None,
):
    """Rejection sampling.

    Args:
        target_logits (Tensor): Processed target logits in shape of
            [batch_size, num_spec_tokens, vocab_size]. Already processed
            by FusedLogitsProcessor (temperature, top-k, top-p applied).
        draft_token_ids (LongTensor): [batch_size, num_spec_tokens]
        bonus_token_ids (LongTensor): [batch_size]
        sampling_inputs (SamplingInputs): Sampling parameters.
        draft_probs (Tensor): [batch_size, num_spec_tokens, vocab_size] or None.
    """
    assert draft_probs is None or draft_probs.is_contiguous()
    if not draft_token_ids.is_contiguous():
        draft_token_ids = draft_token_ids.contiguous()

    if not target_logits.is_contiguous():
        target_logits = target_logits.contiguous()

    batch_size, num_spec_tokens = draft_token_ids.shape
    vocab_size = target_logits.shape[-1]
    device = target_logits.device

    # Determine sampling policy
    is_all_greedy = (sampling_inputs.max_top_k == 1)
    is_all_random = False
    is_greedy = None
    if not is_all_greedy:
        if sampling_inputs.top_k is not None:
            is_greedy = (sampling_inputs.top_k == 1)
            is_all_random = not is_greedy.any().item()
        else:
            is_all_random = True

    # Create output buffer
    output_token_ids = torch.full(
        (batch_size, num_spec_tokens + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )

    # 1. Greedy path (skip if all_random)
    if not is_all_random:
        target_argmax = target_logits.argmax(dim=-1)  # [batch, num_spec]
        rejection_greedy_sample_kernel[(batch_size, )](
            output_token_ids,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            num_spec_tokens,
        )
        if is_all_greedy:
            return _extract_outputs(output_token_ids, num_spec_tokens)

    # 2. Compute target probs from processed logits
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)

    # 3. Uniform random [batch, num_spec] (float64 to avoid exact 0.0)
    uniform_probs = torch.rand(
        (batch_size, num_spec_tokens),
        dtype=torch.float64,
        device=device,
    )

    # 4. Recovered tokens via Gumbel-max trick
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    inv_q = q.reciprocal()

    recovered_token_ids = torch.empty(
        (batch_size, num_spec_tokens),
        dtype=torch.long,
        device=device,
    )
    BLOCK_SIZE = 8192
    sample_recovered_tokens_kernel[(batch_size, num_spec_tokens)](
        recovered_token_ids,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        num_spec_tokens,
        vocab_size,
        BLOCK_SIZE,
        NO_DRAFT_PROBS=draft_probs is None,
    )

    # 5. Random rejection
    rejection_random_sample_kernel[(batch_size, )](
        output_token_ids,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        num_spec_tokens,
        vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )

    return _extract_outputs(output_token_ids, num_spec_tokens)


@triton.jit(do_not_specialize=['num_spec_tokens'])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_tokens + 1]
    draft_token_ids_ptr,  # [batch_size, num_spec_tokens]
    target_argmax_ptr,  # [batch_size, num_spec_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    num_spec_tokens,
):
    """Greedy rejection sampling kernel.

    Grid: (batch_size,)
    For each request: if greedy, accept matching tokens, reject at first
    mismatch.
    """
    req_idx = tl.program_id(0)
    if is_greedy_ptr is not None:
        is_greedy = tl.load(is_greedy_ptr + req_idx).to(tl.int1)
        if not is_greedy:
            return

    out_stride = num_spec_tokens + 1
    draft_stride = num_spec_tokens

    rejected = False
    for pos in range(num_spec_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + req_idx * draft_stride + pos)
            target_argmax_id = tl.load(target_argmax_ptr + req_idx * draft_stride + pos)
            tl.store(
                output_token_ids_ptr + req_idx * out_stride + pos,
                target_argmax_id,
            )
            if draft_token_id != target_argmax_id:
                rejected = True

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * out_stride + num_spec_tokens,
            bonus_token_id,
        )


@triton.jit(do_not_specialize=['num_spec_tokens'])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_tokens + 1]
    draft_token_ids_ptr,  # [batch_size, num_spec_tokens]
    draft_probs_ptr,  # [batch_size, num_spec_tokens, vocab_size] or None
    target_probs_ptr,  # [batch_size, num_spec_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [batch_size, num_spec_tokens]
    uniform_probs_ptr,  # [batch_size, num_spec_tokens]
    is_greedy_ptr,  # [batch_size]
    num_spec_tokens,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """Random rejection sampling kernel.

    Grid: (batch_size,)
    For each non-greedy request: accept if target_prob/draft_prob >=
    uniform, else use recovered token.
    """
    req_idx = tl.program_id(0)
    if is_greedy_ptr is not None:
        is_greedy = tl.load(is_greedy_ptr + req_idx).to(tl.int1)
        if is_greedy:
            return

    out_stride = num_spec_tokens + 1
    draft_stride = num_spec_tokens

    rejected = False
    for pos in range(num_spec_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + req_idx * draft_stride + pos)
            if NO_DRAFT_PROBS:
                draft_prob = 1
            else:
                draft_prob = tl.load(draft_probs_ptr + (req_idx * draft_stride + pos) * vocab_size + draft_token_id)
            target_prob = tl.load(target_probs_ptr + (req_idx * draft_stride + pos) * vocab_size + draft_token_id)
            uniform_prob = tl.load(uniform_probs_ptr + req_idx * draft_stride + pos)

            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + req_idx * draft_stride + pos)
            tl.store(
                output_token_ids_ptr + req_idx * out_stride + pos,
                token_id,
            )

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * out_stride + num_spec_tokens,
            bonus_token_id,
        )


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_tokens]
    draft_token_ids_ptr,  # [batch_size, num_spec_tokens]
    draft_probs_ptr,  # [batch_size, num_spec_tokens, vocab_size] or None
    target_probs_ptr,  # [batch_size, num_spec_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    num_spec_tokens,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """Recovered token sampling kernel via Gumbel-max trick.

    Grid: (batch_size, num_spec_tokens)
    For each position: find argmax of max(0, target_prob - draft_prob)
    * inv_q.
    """
    req_idx = tl.program_id(0)
    pos = tl.program_id(1)
    if pos >= num_spec_tokens:
        return

    draft_stride = num_spec_tokens
    token_idx = req_idx * draft_stride + pos

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    max_val = float('-inf')
    recovered_id = 0
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
            prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=(vocab_mask & (vocab_offset != draft_token_id)),
                other=0.0,
            )
        else:
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            prob = tl.maximum(target_prob - draft_prob, 0.0)

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        score = prob * inv_q
        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    tl.store(output_token_ids_ptr + token_idx, recovered_id)
