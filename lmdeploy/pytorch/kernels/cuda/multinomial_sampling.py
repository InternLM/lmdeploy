# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


@triton.jit
def _multinomial_sampling_kernel(Scores, Seeds, Offsets, Indices, Outputs, stride_sb, stride_st, stride_ib, stride_it,
                                 num_tokens, BLOCK_N: tl.constexpr):
    """Kernel."""
    batch_id = tl.program_id(0)
    n_off = tl.arange(0, BLOCK_N)

    # sampling random seed
    seed = tl.load(Seeds + batch_id)
    offset = tl.load(Offsets + batch_id).to(tl.int32)
    samp = tl.rand(seed, offset)

    # initialize
    acc = 0.0
    score_ptr = Scores + batch_id * stride_sb + n_off * stride_st
    indice_ptr = Indices + batch_id * stride_ib
    output = tl.load(indice_ptr)

    found_mask = False
    for b_idx in tl.range(0, num_tokens, BLOCK_N):
        # triton does not have break statement, use mask to skip computation
        if not found_mask:
            s_off = b_idx + n_off
            s_mask = (s_off < num_tokens)
            scores = tl.load(score_ptr, mask=s_mask, other=0.0).to(tl.float32)
            c_scores = tl.cumsum(scores, 0)
            cum_scores = acc + c_scores
            acc += tl.max(c_scores, 0)

            pre_cum_scores = cum_scores - scores
            valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
            found_mask = tl.sum(valid_mask, 0) > 0

            if found_mask:
                valid_pos = tl.argmax(valid_mask.to(tl.int32), 0)
                indice = tl.load(indice_ptr + valid_pos * stride_it)
                output = indice
        score_ptr += stride_st * BLOCK_N
        indice_ptr += stride_it * BLOCK_N

    tl.store(Outputs + batch_id, output)


def multinomial_sampling(scores: torch.Tensor,
                         seeds: torch.LongTensor,
                         offsets: torch.LongTensor,
                         indices: torch.Tensor = None):
    """Multinomial sampling.

    Note that this kernel assumes the input scores are already sorted in descending order.

    scores: [batch_size, num_tokens], sorted softmax scores
    seeds: [batch_size]
    offsets: [batch_size]
    indices: [batch_size, num_tokens], original token indices before sorting
    """
    assert scores.dim() == 2
    batch_size, num_tokens = scores.size()
    device = scores.device

    if num_tokens == 1:
        return torch.zeros_like(scores, dtype=torch.long)

    if indices is None:
        indices = torch.arange(num_tokens, device=device)
        indices = indices.expand_as(scores)

    assert indices.dim() == 2
    assert indices.size() == scores.size()

    outputs = indices[:, 0].clone()

    BLOCK_N = 128

    grid = [batch_size]
    _multinomial_sampling_kernel[grid](scores,
                                       seeds,
                                       offsets,
                                       indices,
                                       outputs,
                                       stride_sb=scores.stride(0),
                                       stride_st=scores.stride(1),
                                       stride_ib=indices.stride(0),
                                       stride_it=indices.stride(1),
                                       num_tokens=num_tokens,
                                       BLOCK_N=BLOCK_N,
                                       num_warps=1)

    return outputs
