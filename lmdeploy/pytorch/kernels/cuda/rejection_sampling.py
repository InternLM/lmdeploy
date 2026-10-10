# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl

from lmdeploy.pytorch.backends.rejection_sampling import PLACEHOLDER_TOKEN_ID

_PLACEHOLDER_TOKEN_ID = tl.constexpr(PLACEHOLDER_TOKEN_ID)


@triton.jit
def _rejection_sampling_random_kernel(
    seeds_ptr, offsets_ptr, uniform_ptr, inv_q_ptr,
    seed_stride, offset_stride, num_spec_tokens: tl.constexpr, vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Philox streams keyed by request, decoding step, and sampling purpose."""
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    seed = tl.load(seeds_ptr + req_idx * seed_stride).to(tl.uint64)
    step = tl.load(offsets_ptr + req_idx * offset_stride).to(tl.uint64)
    vocab = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    counter = (step << 32) | vocab.to(tl.uint64)
    bits = tl.randint4x(seed ^ (2 << 32), counter)[0]
    uniform = ((bits >> 9).to(tl.float32) + .5) * (1. / (1 << 23))
    tl.store(inv_q_ptr + req_idx * vocab_size + vocab, -1. / tl.log(uniform), vocab < vocab_size)
    if block_idx == 0:
        pos = tl.arange(0, triton.next_power_of_2(num_spec_tokens))
        accept_counter = (step << 32) | pos.to(tl.uint64)
        accept_bits = tl.randint4x(seed ^ (1 << 32), accept_counter)[0]
        accept_uniform = ((accept_bits >> 9).to(tl.float32) + .5) * (1. / (1 << 23))
        tl.store(uniform_ptr + req_idx * num_spec_tokens + pos, accept_uniform, pos < num_spec_tokens)


def rejection_sampling_random(seeds, offsets, num_spec_tokens, vocab_size):
    """Generate acceptance uniforms and recovery noise without global RNG state."""
    batch_size = seeds.numel()
    uniform = torch.empty((batch_size, num_spec_tokens), device=seeds.device, dtype=torch.float32)
    inv_q = torch.empty((batch_size, vocab_size), device=seeds.device, dtype=torch.float32)
    _rejection_sampling_random_kernel[(batch_size, triton.cdiv(vocab_size, 1024))](
        seeds, offsets, uniform, inv_q, seeds.stride(0), offsets.stride(0), num_spec_tokens, vocab_size, 1024)
    return uniform, inv_q


@triton.jit(do_not_specialize=['num_spec_tokens'])
def _rejection_greedy_sample_kernel(
    output_token_ids_ptr,
    num_rejected_tokens_ptr,
    last_token_ids_ptr,
    draft_token_ids_ptr,
    target_token_ids_ptr,
    bonus_token_ids_ptr,
    is_greedy_ptr,
    output_stride_b,
    output_stride_s,
    rejected_stride_b,
    last_stride_b,
    draft_stride_b,
    draft_stride_s,
    target_stride_b,
    target_stride_s,
    bonus_stride_b,
    greedy_stride_b,
    num_spec_tokens,
):
    """Write the greedy accepted prefix and causal state outputs."""
    req_idx = tl.program_id(0)
    if is_greedy_ptr is not None:
        is_greedy = tl.load(is_greedy_ptr + req_idx * greedy_stride_b).to(tl.int1)
        if not is_greedy:
            return

    rejected = False
    num_accepted = 0
    last_token_id = tl.full((), 0, tl.int64)
    for pos in range(num_spec_tokens):
        output_ptr = output_token_ids_ptr + req_idx * output_stride_b + pos * output_stride_s
        if not rejected:
            draft_token_id = tl.load(
                draft_token_ids_ptr + req_idx * draft_stride_b + pos * draft_stride_s)
            target_token_id = tl.load(
                target_token_ids_ptr + req_idx * target_stride_b + pos * target_stride_s)
            tl.store(output_ptr, target_token_id)
            last_token_id = target_token_id
            num_accepted += 1
            if draft_token_id != target_token_id:
                rejected = True
        else:
            tl.store(output_ptr, _PLACEHOLDER_TOKEN_ID)

    bonus_output_ptr = (
        output_token_ids_ptr
        + req_idx * output_stride_b
        + num_spec_tokens * output_stride_s
    )
    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx * bonus_stride_b)
        tl.store(bonus_output_ptr, bonus_token_id)
        last_token_id = bonus_token_id
        num_accepted += 1
    else:
        tl.store(bonus_output_ptr, _PLACEHOLDER_TOKEN_ID)

    tl.store(
        num_rejected_tokens_ptr + req_idx * rejected_stride_b,
        num_spec_tokens + 1 - num_accepted,
    )
    tl.store(last_token_ids_ptr + req_idx * last_stride_b, last_token_id)


@triton.jit(do_not_specialize=['num_spec_tokens'])
def _rejection_random_sample_kernel(
    output_token_ids_ptr,
    num_rejected_tokens_ptr,
    last_token_ids_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    bonus_token_ids_ptr,
    recovered_token_ids_ptr,
    uniform_probs_ptr,
    is_greedy_ptr,
    num_spec_tokens,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """Write random rejection results for the non-greedy rows."""
    req_idx = tl.program_id(0)
    if is_greedy_ptr is not None:
        is_greedy = tl.load(is_greedy_ptr + req_idx).to(tl.int1)
        if is_greedy:
            return

    out_stride = num_spec_tokens + 1
    draft_stride = num_spec_tokens
    rejected = False
    num_accepted = 0
    last_token_id = tl.full((), 0, tl.int64)
    for pos in range(num_spec_tokens):
        output_ptr = output_token_ids_ptr + req_idx * out_stride + pos
        if not rejected:
            draft_token_id = tl.load(
                draft_token_ids_ptr + req_idx * draft_stride + pos)
            if NO_DRAFT_PROBS:
                draft_prob = 1
            else:
                draft_prob = tl.load(
                    draft_probs_ptr
                    + (req_idx * draft_stride + pos) * vocab_size
                    + draft_token_id)
            target_prob = tl.load(
                target_probs_ptr
                + (req_idx * draft_stride + pos) * vocab_size
                + draft_token_id)
            uniform_prob = tl.load(
                uniform_probs_ptr + req_idx * draft_stride + pos)

            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(
                    recovered_token_ids_ptr + req_idx * draft_stride + pos)
            tl.store(output_ptr, token_id)
            last_token_id = token_id
            num_accepted += 1
        else:
            tl.store(output_ptr, _PLACEHOLDER_TOKEN_ID)

    bonus_output_ptr = output_token_ids_ptr + req_idx * out_stride + num_spec_tokens
    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(bonus_output_ptr, bonus_token_id)
        last_token_id = bonus_token_id
        num_accepted += 1
    else:
        tl.store(bonus_output_ptr, _PLACEHOLDER_TOKEN_ID)

    tl.store(
        num_rejected_tokens_ptr + req_idx,
        num_spec_tokens + 1 - num_accepted,
    )
    tl.store(last_token_ids_ptr + req_idx, last_token_id)


@triton.jit
def _sample_recovered_tokens_kernel(
    output_token_ids_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    inv_q_ptr,
    num_spec_tokens,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """Sample recovered tokens with the Gumbel-max trick."""
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


def greedy_rejection_sample(
    output_token_ids: torch.LongTensor,
    num_rejected_tokens: torch.LongTensor,
    last_token_ids: torch.LongTensor,
    draft_token_ids: torch.LongTensor,
    target_token_ids: torch.LongTensor,
    bonus_token_ids: torch.LongTensor,
    is_greedy: torch.Tensor | None,
):
    """Launch greedy longest-prefix verification."""
    batch_size, num_spec_tokens = draft_token_ids.shape
    greedy_stride = 0 if is_greedy is None else is_greedy.stride(0)
    _rejection_greedy_sample_kernel[(batch_size, )](
        output_token_ids,
        num_rejected_tokens,
        last_token_ids,
        draft_token_ids,
        target_token_ids,
        bonus_token_ids,
        is_greedy,
        *output_token_ids.stride(),
        num_rejected_tokens.stride(0),
        last_token_ids.stride(0),
        *draft_token_ids.stride(),
        *target_token_ids.stride(),
        bonus_token_ids.stride(0),
        greedy_stride,
        num_spec_tokens,
    )


def random_rejection_sample(
    output_token_ids: torch.LongTensor,
    num_rejected_tokens: torch.LongTensor,
    last_token_ids: torch.LongTensor,
    draft_token_ids: torch.LongTensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.LongTensor,
    recovered_token_ids: torch.LongTensor,
    uniform_probs: torch.Tensor,
    is_greedy: torch.Tensor | None,
):
    """Launch probabilistic rejection for non-greedy rows."""
    batch_size, num_spec_tokens = draft_token_ids.shape
    vocab_size = target_probs.shape[-1]
    _rejection_random_sample_kernel[(batch_size, )](
        output_token_ids,
        num_rejected_tokens,
        last_token_ids,
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


def sample_recovered_tokens(
    output_token_ids: torch.LongTensor,
    draft_token_ids: torch.LongTensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    inv_q: torch.Tensor,
):
    """Launch recovered-token sampling."""
    batch_size, num_spec_tokens = draft_token_ids.shape
    vocab_size = target_probs.shape[-1]
    block_size = 8192
    _sample_recovered_tokens_kernel[(batch_size, num_spec_tokens)](
        output_token_ids,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        num_spec_tokens,
        vocab_size,
        block_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )
