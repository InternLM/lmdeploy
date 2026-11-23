# SPDX-License-Identifier: Apache-2.0
#
# Rejection sampling for speculative decoding on Ascend NPU.
#
# Triton kernels adapted from:
#   vllm-ascend: vllm_ascend/ops/triton/reject_sample.py
#   (Apache-2.0, Copyright (c) 2025 Huawei Technologies Co., Ltd.)
#
# PyTorch fallbacks adapted from:
#   vllm-ascend: vllm_ascend/sample/rejection_sampler.py
#   (Apache-2.0, Copyright (c) 2025 Huawei Technologies Co., Ltd.)
#
# Interface wrapper designed for lmdeploy's rejection_sample signature.
# mypy: ignore-errors

import os

import torch

PLACEHOLDER_TOKEN_ID = -1

# ---------------------------------------------------------------------------
# Backend selection.
#
# Which implementation runs (Triton kernels vs. PyTorch) is chosen by an
# environment variable, NOT by whether Triton happens to be importable:
#
#   DLINFER_ASCEND_REJECT_SAMPLE_USE_TRITON = "1"  (default) -> Triton kernels
#                                           = "0"            -> PyTorch
#
# The PyTorch path is the accuracy-verified reference implementation ported
# from the lmdeploy_ext device patch (see __init__torch_sample.py).
# ---------------------------------------------------------------------------
_USE_TRITON_ENV = 'DLINFER_ASCEND_REJECT_SAMPLE_USE_TRITON'


def _reject_sample_use_triton() -> bool:
    """Return True iff the Triton backend is selected via env var."""
    return os.environ.get(_USE_TRITON_ENV, '1') == '1'


# ---------------------------------------------------------------------------
# Try to load Triton and the Ascend CANN get_element extension.
# The `get_element` op is required to extract scalars from Triton vectors
# inside per-request loops on Ascend hardware.
# ---------------------------------------------------------------------------
_TRITON_AVAILABLE = False
_get_element = None

try:
    import triton
    import triton.language as tl

    try:
        import triton.language.extra.cann.extension as _cann_ext

        _get_element = getattr(_cann_ext, 'get_element', None)
    except ImportError:
        _get_element = getattr(tl, 'get_element', None)

    _TRITON_AVAILABLE = _get_element is not None
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Grid / block-size helper
# ---------------------------------------------------------------------------


def _cal_grid_block_size(batch_size: int):
    """Return (grid, block_size) tuned to the NPU's vector-core count."""
    try:
        from .triton_utils import get_vectorcore_num, init_device_properties_triton

        init_device_properties_triton()
        vectorcore_num = get_vectorcore_num()
    except Exception:
        vectorcore_num = max(1, batch_size)

    if batch_size <= vectorcore_num:
        return batch_size, 1
    grid = vectorcore_num
    block_size = triton.next_power_of_2(triton.cdiv(batch_size, grid))
    return grid, block_size


# ---------------------------------------------------------------------------
# Triton kernels  (compiled only when _TRITON_AVAILABLE is True)
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    get_element = _get_element

    @triton.jit(do_not_specialize=['max_spec_len'])
    def _bonus_store_kernel(
        bonus_token_ids_ptr,
        position,
        output_token_ids_ptr,
        max_spec_len,
        num_accepted,
    ):
        bonus = tl.load(bonus_token_ids_ptr + position)
        tl.store(output_token_ids_ptr + position * (max_spec_len + 1) + num_accepted, bonus)

    @triton.jit(do_not_specialize=['vec_len', 'max_spec_len'])
    def _rejection_greedy_kernel(
        output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens_ptr,  # [batch_size]
        draft_token_ids_ptr,  # [num_tokens]
        target_argmax_ptr,  # [num_tokens]
        bonus_token_ids_ptr,  # [batch_size]
        is_greedy_ptr,  # [batch_size] bool, or None (means all greedy)
        vec_len,
        max_spec_len,
        BLOCK_SIZE: tl.constexpr,
    ):
        block_idx = tl.program_id(0)
        offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < vec_len

        if is_greedy_ptr is None:
            is_greedy_mask = mask
        else:
            is_greedy = tl.load(is_greedy_ptr + offset, mask=mask, other=0)
            is_greedy_mask = mask & (is_greedy != 0)

        # The first request starts at zero and has no preceding prefix sum.
        start_idx = tl.load(
            cu_num_draft_tokens_ptr + offset - 1,
            mask=is_greedy_mask & (offset > 0),
            other=0,
        )
        end_idx = tl.load(cu_num_draft_tokens_ptr + offset, mask=is_greedy_mask, other=0)
        num_draft_tokens = end_idx - start_idx

        for pos in tl.range(0, BLOCK_SIZE):
            n = get_element(num_draft_tokens, (pos,))
            s = get_element(start_idx, (pos,))
            is_g = get_element(is_greedy_mask, (pos,))
            req_idx = block_idx * BLOCK_SIZE + pos
            if is_g:
                rejected = False
                for i in range(n):
                    if not rejected:
                        draft_id = tl.load(draft_token_ids_ptr + s + i)
                        target_id = tl.load(target_argmax_ptr + s + i)
                        tl.store(
                            output_token_ids_ptr + req_idx * (max_spec_len + 1) + i,
                            target_id,
                        )
                        if draft_id != target_id:
                            rejected = True
                if not rejected:
                    _bonus_store_kernel(
                        bonus_token_ids_ptr,
                        req_idx,
                        output_token_ids_ptr,
                        max_spec_len,
                        n,
                    )

    @triton.jit(do_not_specialize=['max_spec_len'])
    def _rejection_random_kernel(
        output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens_ptr,  # [batch_size]
        draft_token_ids_ptr,  # [num_tokens]
        draft_probs_ptr,  # [num_tokens, vocab_size] or None
        target_probs_ptr,  # [num_tokens, vocab_size]
        bonus_token_ids_ptr,  # [batch_size]
        recovered_token_ids_ptr,  # [num_tokens]
        uniform_probs_ptr,  # [num_tokens]
        is_greedy_ptr,  # [batch_size] bool
        max_spec_len,
        vocab_size,
        vec_len,
        NO_DRAFT_PROBS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        block_idx = tl.program_id(0)
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vec_len

        is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
        not_greedy_mask = is_greedy == 0

        start_idxs = tl.load(
            cu_num_draft_tokens_ptr + offsets - 1,
            mask=not_greedy_mask & (offsets > 0),
            other=0,
        )
        end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, mask=not_greedy_mask, other=0)
        n_draft_tokens = end_idxs - start_idxs

        for req_i in range(BLOCK_SIZE):
            not_greedy = get_element(not_greedy_mask, (req_i,))
            if not_greedy:
                rejected = False
                s = get_element(start_idxs, (req_i,))
                req_idx = block_idx * BLOCK_SIZE + req_i
                n = get_element(n_draft_tokens, (req_i,))
                for pos in range(n):
                    if not rejected:
                        draft_id = tl.load(draft_token_ids_ptr + s + pos)
                        if NO_DRAFT_PROBS:
                            draft_prob = 1.0
                        else:
                            draft_prob = tl.load(draft_probs_ptr + (s + pos) * vocab_size + draft_id)
                        target_prob = tl.load(target_probs_ptr + (s + pos) * vocab_size + draft_id)
                        uniform_prob = tl.load(uniform_probs_ptr + s + pos)
                        if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                            token_id = draft_id
                        else:
                            rejected = True
                            token_id = tl.load(recovered_token_ids_ptr + s + pos)
                        tl.store(
                            output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                            token_id,
                        )
                if not rejected:
                    bonus = tl.load(bonus_token_ids_ptr + req_idx)
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + n, bonus)

    @triton.jit
    def _sample_recovered_tokens_kernel(
        output_token_ids_ptr,  # [num_tokens]
        cu_num_draft_tokens_ptr,  # [batch_size]
        draft_token_ids_ptr,  # [num_tokens]
        draft_probs_ptr,  # [num_tokens, vocab_size] or None
        target_probs_ptr,  # [num_tokens, vocab_size]  read-only
        q_ptr,  # [batch_size, vocab_size]  exponential samples
        vocab_size,
        PADDED_VOCAB_SIZE: tl.constexpr,
        NO_DRAFT_PROBS: tl.constexpr,
        SUB_BLOCK: tl.constexpr,
    ):
        req_idx = tl.program_id(0)
        start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
        num_draft_tokens = end_idx - start_idx

        pos = tl.program_id(1)
        if pos >= num_draft_tokens:
            return

        loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
        global_recovered_id = -1
        global_max_score = -1.0

        if NO_DRAFT_PROBS:
            # Load draft token id once; mask it out per sub-block via tl.where
            # to avoid in-place modification of target_probs (which could cause
            # memory pipeline hazards on Ascend with multibuffer enabled).
            draft_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            for loop_i in range(loop):
                vocab_start = loop_i * SUB_BLOCK
                vocab_off = vocab_start + tl.arange(0, SUB_BLOCK)
                prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + vocab_off,
                    mask=vocab_off < vocab_size,
                    other=0.0,
                )
                # Zero out the draft-token position without modifying the buffer.
                prob = tl.where(vocab_off == draft_id, 0.0, prob)
                q = tl.load(
                    q_ptr + req_idx * vocab_size + vocab_off,
                    mask=vocab_off < vocab_size,
                    other=float('-inf'),
                )
                score = prob / q
                local_id = tl.argmax(score, axis=-1)
                local_max = get_element(score, (local_id,))
                if local_max > global_max_score:
                    global_max_score = local_max
                    global_recovered_id = vocab_start + local_id
        else:
            for loop_i in range(loop):
                vocab_start = loop_i * SUB_BLOCK
                vocab_off = vocab_start + tl.arange(0, SUB_BLOCK)
                draft_p = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_off,
                    mask=vocab_off < vocab_size,
                    other=0.0,
                )
                target_p = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + vocab_off,
                    mask=vocab_off < vocab_size,
                    other=0.0,
                )
                prob = tl.maximum(target_p - draft_p, 0.0)
                q = tl.load(
                    q_ptr + req_idx * vocab_size + vocab_off,
                    mask=vocab_off < vocab_size,
                    other=float('-inf'),
                )
                score = prob / q
                local_id = tl.argmax(score, axis=-1)
                local_max = get_element(score, (local_id,))
                if local_max > global_max_score:
                    global_max_score = local_max
                    global_recovered_id = vocab_start + local_id

        tl.store(output_token_ids_ptr + start_idx + pos, global_recovered_id)


# ---------------------------------------------------------------------------
# PyTorch implementation  (accuracy-verified reference)
#
# Ported from the lmdeploy_ext device patch (__init__torch_sample.py), which
# was evaluated for correctness. It operates directly on the
# [batch, spec, vocab] tensors (no flattening) and handles the greedy, random
# and mixed cases in a single per-request loop, so it is fully self-contained
# and never falls back to the NVIDIA-Triton rejection_sample.
# ---------------------------------------------------------------------------


def _rejection_sample_torch(
    target_logits,  # [batch_size, num_spec_tokens, vocab_size]
    draft_token_ids,  # [batch_size, num_spec_tokens]
    bonus_token_ids,  # [batch_size]
    sampling_inputs,  # lmdeploy SamplingInputs
    draft_probs=None,  # [batch_size, num_spec_tokens, vocab_size] or None
):
    assert draft_probs is None or draft_probs.is_contiguous()
    if not draft_token_ids.is_contiguous():
        draft_token_ids = draft_token_ids.contiguous()
    if not target_logits.is_contiguous():
        target_logits = target_logits.contiguous()

    batch_size, num_spec_tokens = draft_token_ids.shape
    device = target_logits.device

    output_token_ids = torch.full(
        (batch_size, num_spec_tokens + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )

    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    if sampling_inputs.top_k is not None:
        is_greedy = sampling_inputs.top_k == 1
        if not torch.is_tensor(is_greedy):
            is_greedy = torch.full((batch_size,), bool(is_greedy), dtype=torch.bool, device=device)
        else:
            is_greedy = is_greedy.to(device=device, dtype=torch.bool)
    else:
        is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)

    target_argmax = target_probs.argmax(dim=-1)
    uniform_probs = torch.rand((batch_size, num_spec_tokens), dtype=torch.float64, device=device)
    # Precompute the reciprocal of the exponential samples and multiply below:
    # NPU in-kernel division uses a low-precision reciprocal, so performing the
    # division here in torch keeps the recovered-token argmax accurate.
    inv_q = torch.empty((batch_size, target_probs.shape[-1]), dtype=torch.float32, device=device)
    inv_q.exponential_()
    inv_q = inv_q.reciprocal()

    recovered_token_ids = torch.empty((batch_size, num_spec_tokens), dtype=torch.long, device=device)
    zero = target_probs.new_tensor(0.0)
    for batch_idx in range(batch_size):
        if bool(is_greedy[batch_idx].item()):
            continue
        batch_inv_q = inv_q[batch_idx]
        for pos in range(num_spec_tokens):
            draft_token_id = draft_token_ids[batch_idx, pos]
            if draft_probs is None:
                prob = target_probs[batch_idx, pos].clone()
                prob[draft_token_id] = 0.0
            else:
                prob = torch.maximum(
                    target_probs[batch_idx, pos] - draft_probs[batch_idx, pos],
                    zero,
                )
            recovered_token_ids[batch_idx, pos] = torch.argmax(prob * batch_inv_q)

    for batch_idx in range(batch_size):
        rejected = False
        if bool(is_greedy[batch_idx].item()):
            for pos in range(num_spec_tokens):
                token_id = target_argmax[batch_idx, pos]
                output_token_ids[batch_idx, pos] = token_id
                if draft_token_ids[batch_idx, pos] != token_id:
                    rejected = True
                    break
        else:
            for pos in range(num_spec_tokens):
                draft_token_id = draft_token_ids[batch_idx, pos]
                if draft_probs is None:
                    draft_prob = 1.0
                else:
                    draft_prob = float(draft_probs[batch_idx, pos, draft_token_id].item())
                target_prob = float(target_probs[batch_idx, pos, draft_token_id].item())
                uniform_prob = float(uniform_probs[batch_idx, pos].item())
                if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                    token_id = draft_token_id
                else:
                    token_id = recovered_token_ids[batch_idx, pos]
                    rejected = True
                output_token_ids[batch_idx, pos] = token_id
                if rejected:
                    break

        if not rejected:
            output_token_ids[batch_idx, num_spec_tokens] = bonus_token_ids[batch_idx]

    return _extract_outputs(output_token_ids, num_spec_tokens)


# ---------------------------------------------------------------------------
# Internal driver  (mirrors vllm-ascend rejection_sample logic)
# ---------------------------------------------------------------------------


def _rejection_sample_impl(
    draft_token_ids,  # [num_tokens]
    num_draft_tokens,  # list[int], all equal in lmdeploy
    max_spec_len,  # int
    cu_num_draft_tokens,  # [batch_size]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]  (float32)
    bonus_token_ids,  # [batch_size, 1]
    is_greedy,  # [batch_size] bool or None
    all_greedy,  # bool
    all_random,  # bool
):
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device

    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    grid, block_size = _cal_grid_block_size(batch_size)

    # ------------------------------------------------------------------
    # Greedy rejection pass
    # ------------------------------------------------------------------
    if not all_random:
        target_argmax = target_probs.argmax(dim=-1)
        _rejection_greedy_kernel[(grid,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids.squeeze(1),
            is_greedy,  # None means all-greedy
            batch_size,
            max_spec_len,
            BLOCK_SIZE=block_size,
        )

        if all_greedy:
            return output_token_ids

    # ------------------------------------------------------------------
    # Random rejection pass — sample recovered tokens, then accept/reject
    # ------------------------------------------------------------------
    uniform_probs = torch.rand(num_tokens, dtype=torch.float64, device=device).to(torch.float32)

    # q ~ Exp(1); recovered token = argmax((target - draft) / q)
    q = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
    q.exponential_()

    recovered_token_ids = torch.empty(num_tokens, dtype=torch.int32, device=device)
    padded_vocab = triton.next_power_of_2(vocab_size)
    _sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        padded_vocab,
        NO_DRAFT_PROBS=draft_probs is None,
        SUB_BLOCK=4096,
        multibuffer=False,
    )

    _rejection_random_kernel[(grid,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids.squeeze(1),
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        batch_size,
        NO_DRAFT_PROBS=draft_probs is None,
        BLOCK_SIZE=block_size,
    )

    return output_token_ids


# ---------------------------------------------------------------------------
# Public entry point — lmdeploy rejection_sample interface
# ---------------------------------------------------------------------------


def _extract_outputs(output_token_ids, num_spec_tokens):
    """Compute num_rejected_tokens and last_token_ids from output_token_ids.

    output_token_ids: [batch_size, num_spec_tokens + 1]  (int32, PLACEHOLDER=-1)
    Returns: (output_token_ids, num_rejected_tokens, last_token_ids)
    """
    batch_size = output_token_ids.size(0)
    valid_mask = output_token_ids != PLACEHOLDER_TOKEN_ID
    num_accepted = valid_mask.sum(dim=1)
    num_rejected = num_spec_tokens + 1 - num_accepted
    last_ids = output_token_ids[
        torch.arange(batch_size, device=output_token_ids.device),
        num_accepted - 1,
    ]
    return output_token_ids, num_rejected, last_ids


def rejection_sample(
    target_logits,  # [batch_size, num_spec_tokens, vocab_size]
    draft_token_ids,  # [batch_size, num_spec_tokens]
    bonus_token_ids,  # [batch_size]
    sampling_inputs,  # lmdeploy SamplingInputs
    draft_probs=None,  # [batch_size, num_spec_tokens, vocab_size] or None
):
    """Rejection sampler for lmdeploy speculative decoding on Ascend NPU.

    Replaces the NVIDIA-Triton-based rejection_sample with kernels and
    vectorised PyTorch ops that run correctly on Ascend.

    Returns: (output_token_ids, num_rejected_tokens, last_token_ids)
    """
    # ---- Backend selection via env var (NOT Triton availability) ----------
    if not _reject_sample_use_triton():
        return _rejection_sample_torch(
            target_logits,
            draft_token_ids,
            bonus_token_ids,
            sampling_inputs,
            draft_probs=draft_probs,
        )
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            f"{_USE_TRITON_ENV} selects the Triton rejection-sampling backend, "
            'but Triton (or the Ascend CANN get_element extension) is unavailable. '
            f"Set {_USE_TRITON_ENV}=0 to use the PyTorch backend."
        )

    batch_size, num_spec_tokens, vocab_size = target_logits.shape
    device = target_logits.device

    # ---- Flatten [batch, spec, vocab] → [num_tokens, vocab] ---------------
    num_tokens = batch_size * num_spec_tokens
    target_logits_flat = target_logits.reshape(num_tokens, vocab_size)
    draft_token_ids_flat = draft_token_ids.reshape(num_tokens).to(torch.int32)
    draft_probs_flat = draft_probs.reshape(num_tokens, vocab_size) if draft_probs is not None else None

    # Build cumulative draft-token offsets (uniform: each request has num_spec_tokens).
    cu_num_draft_tokens = torch.arange(1, batch_size + 1, device=device, dtype=torch.int32) * num_spec_tokens
    num_draft_tokens_list = [num_spec_tokens] * batch_size

    # ---- Determine greedy/random policy ------------------------------------
    all_greedy = sampling_inputs.max_top_k == 1
    if all_greedy:
        is_greedy = None  # signal: all requests are greedy
        all_random = False
    elif sampling_inputs.top_k is not None:
        is_greedy = (sampling_inputs.top_k == 1).to(torch.bool).to(device)
        all_random = not is_greedy.any().item()
    else:
        is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)
        all_random = True

    # ---- Convert logits → probs for the non-greedy path -------------------
    target_probs = target_logits_flat.softmax(dim=-1, dtype=torch.float32)

    # ---- Run rejection sampling -------------------------------------------
    output_token_ids = _rejection_sample_impl(
        draft_token_ids=draft_token_ids_flat,
        num_draft_tokens=num_draft_tokens_list,
        max_spec_len=num_spec_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        draft_probs=draft_probs_flat,
        target_probs=target_probs,
        bonus_token_ids=bonus_token_ids.unsqueeze(1),
        is_greedy=is_greedy,
        all_greedy=all_greedy,
        all_random=all_random,
    )

    return _extract_outputs(output_token_ids.long(), num_spec_tokens)
