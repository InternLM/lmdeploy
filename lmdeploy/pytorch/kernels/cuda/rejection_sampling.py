# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from lmdeploy.pytorch.backends.rejection_sampling import PLACEHOLDER_TOKEN_ID

_PLACEHOLDER_TOKEN_ID = tl.constexpr(PLACEHOLDER_TOKEN_ID)


@triton.jit
def _rejection_uniform(seed, position, lane, stream: tl.constexpr, dtype: tl.constexpr):
    # Philox counters separate token positions, vocabulary lanes, and RNG uses.
    # Stream 0 is reserved for AR sampling; 1 is acceptance and 2 is recovery.
    bits, _, _, _ = tl.philox(seed, lane.to(tl.uint32), position.to(tl.uint32),
                             (position.to(tl.uint64) >> 32).to(tl.uint32),
                             tl.full((), stream, tl.uint32))
    uniform = (bits.to(dtype) + 0.5) * 2.3283064365386963e-10
    if dtype == tl.float32:
        # Rounding uint32 to fp32 can produce 1, making the exponential infinite.
        uniform = tl.minimum(uniform, 1.0 - 5.960464477539063e-8)
    return uniform


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
    random_seeds_ptr,
    random_offsets_ptr,
    seed_stride,
    offset_stride,
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
    seed = tl.load(random_seeds_ptr + req_idx * seed_stride)
    offset = tl.load(random_offsets_ptr + req_idx * offset_stride)
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
            uniform_prob = _rejection_uniform(seed, offset + pos, tl.full((), 0, tl.uint32), 1, tl.float64)

            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.full((), _PLACEHOLDER_TOKEN_ID, tl.int64)
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
    partial_scores_ptr,
    partial_tokens_ptr,
    num_rejected_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    random_seeds_ptr,
    random_offsets_ptr,
    is_greedy_ptr,
    seed_stride,
    offset_stride,
    num_spec_tokens,
    vocab_size,
    num_vocab_blocks,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """Sample only the first rejected position, in parallel vocabulary
    blocks."""
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
    if num_rejected == 0:
        return
    if is_greedy_ptr is not None:
        if tl.load(is_greedy_ptr + req_idx):
            return

    pos = num_spec_tokens - num_rejected
    token_idx = req_idx * num_spec_tokens + pos
    seed = tl.load(random_seeds_ptr + req_idx * seed_stride)
    position = tl.load(random_offsets_ptr + req_idx * offset_stride) + pos
    vocab_offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    vocab_mask = vocab_offset < vocab_size
    target_prob = tl.load(target_probs_ptr + token_idx * vocab_size + vocab_offset,
                          mask=vocab_mask, other=0.0)
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
        prob = tl.where(vocab_offset == draft_token_id, 0.0, target_prob)
    else:
        draft_prob = tl.load(draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                             mask=vocab_mask, other=0.0)
        prob = tl.maximum(target_prob - draft_prob, 0.0)

    uniform = _rejection_uniform(seed, position, vocab_offset, 2, tl.float32)
    score = tl.where(vocab_mask, prob * (-1.0 / libdevice.log1p(-uniform)), float('-inf'))
    value, index = tl.max(score, axis=0, return_indices=True)
    tl.store(partial_scores_ptr + req_idx * num_vocab_blocks + block_idx, value)
    tl.store(partial_tokens_ptr + req_idx * num_vocab_blocks + block_idx, block_idx * BLOCK_SIZE + index)


@triton.jit
def _store_recovered_tokens_kernel(
    output_token_ids_ptr,
    last_token_ids_ptr,
    partial_scores_ptr,
    partial_tokens_ptr,
    num_rejected_tokens_ptr,
    is_greedy_ptr,
    num_spec_tokens,
    num_vocab_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
    if num_rejected == 0:
        return
    if is_greedy_ptr is not None:
        if tl.load(is_greedy_ptr + req_idx):
            return
    blocks = tl.arange(0, BLOCK_SIZE)
    scores = tl.load(partial_scores_ptr + req_idx * num_vocab_blocks + blocks,
                     mask=blocks < num_vocab_blocks, other=float('-inf'))
    block_idx = tl.argmax(scores, axis=0)
    token_id = tl.load(partial_tokens_ptr + req_idx * num_vocab_blocks + block_idx)
    pos = num_spec_tokens - num_rejected
    tl.store(output_token_ids_ptr + req_idx * (num_spec_tokens + 1) + pos, token_id)
    tl.store(last_token_ids_ptr + req_idx, token_id)


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
    random_seeds: torch.Tensor,
    random_offsets: torch.Tensor,
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
        random_seeds,
        random_offsets,
        random_seeds.stride(0),
        random_offsets.stride(0),
        is_greedy,
        num_spec_tokens,
        vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )


def sample_recovered_tokens(
    output_token_ids: torch.LongTensor,
    num_rejected_tokens: torch.LongTensor,
    last_token_ids: torch.LongTensor,
    draft_token_ids: torch.LongTensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    random_seeds: torch.Tensor,
    random_offsets: torch.Tensor,
    is_greedy: torch.Tensor | None,
):
    """Fill the first rejected token after prefix verification."""
    batch_size, num_spec_tokens = draft_token_ids.shape
    vocab_size = target_probs.shape[-1]
    block_size = 1024
    num_vocab_blocks = triton.cdiv(vocab_size, block_size)
    partial_scores = target_probs.new_empty((batch_size, num_vocab_blocks))
    partial_tokens = draft_token_ids.new_empty((batch_size, num_vocab_blocks))
    _sample_recovered_tokens_kernel[(batch_size, num_vocab_blocks)](
        partial_scores,
        partial_tokens,
        num_rejected_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        random_seeds,
        random_offsets,
        is_greedy,
        random_seeds.stride(0),
        random_offsets.stride(0),
        num_spec_tokens,
        vocab_size,
        num_vocab_blocks,
        BLOCK_SIZE=block_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    _store_recovered_tokens_kernel[(batch_size,)](
        output_token_ids,
        last_token_ids,
        partial_scores,
        partial_tokens,
        num_rejected_tokens,
        is_greedy,
        num_spec_tokens,
        num_vocab_blocks,
        BLOCK_SIZE=triton.next_power_of_2(num_vocab_blocks),
    )
