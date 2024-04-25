# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor

from .triton_utils import get_kernel_meta, wrap_jit_func


@wrap_jit_func
@triton.jit
def _multinomial_sampling_kernel(Scores: 'Tensor', Seeds: 'Tensor',
                                 Offsets: 'Tensor', Indices: 'Tensor',
                                 Outputs: 'Tensor', stride_sb: int,
                                 stride_st: int, stride_ib: int,
                                 stride_it: int, num_batchs: int, num_tokens,
                                 BLOCK: tl.constexpr, BLOCK_N: tl.constexpr):
    """Kernel."""
    batch_block_id = tl.program_id(0)

    off = batch_block_id * BLOCK + tl.arange(0, BLOCK)
    n_off = tl.arange(0, BLOCK_N)

    off_mask = off < num_batchs
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)

    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    output = tl.load(Indices + off * stride_ib, mask=off_mask)

    for b_idx in range(0, num_tokens, BLOCK_N):
        s_off = b_idx + n_off
        s_mask = off_mask[:, None] & (s_off[None, :] < num_tokens)
        scores = tl.load(Scores + off[:, None] * stride_sb +
                         s_off[None, :] * stride_st,
                         mask=s_mask,
                         other=0.0).to(tl.float32)
        c_scores = tl.cumsum(scores, 1)
        cum_scores = acc[:, None] + c_scores
        acc += tl.max(c_scores, 1)

        pre_cum_scores = cum_scores - scores
        valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
        found_mask = tl.sum(valid_mask, 1) > 0

        valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
        indices = tl.load(Indices + off * stride_ib + valid_pos * stride_it,
                          mask=found_mask & off_mask,
                          other=-1)
        output = tl.where(found_mask, indices, output)

    tl.store(Outputs + off, output, mask=off_mask)


def multinomial_sampling(scores: torch.Tensor,
                         seeds: torch.LongTensor,
                         offsets: torch.LongTensor,
                         indices: torch.Tensor = None):
    """multinomial sampling."""

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

    BLOCK = 32
    BLOCK_N = 64

    grid = [triton.cdiv(batch_size, BLOCK)]
    kernel_meta = get_kernel_meta(scores)
    _multinomial_sampling_kernel[grid](scores,
                                       seeds,
                                       offsets,
                                       indices,
                                       outputs,
                                       stride_sb=scores.stride(0),
                                       stride_st=scores.stride(1),
                                       stride_ib=indices.stride(0),
                                       stride_it=indices.stride(1),
                                       num_batchs=batch_size,
                                       num_tokens=num_tokens,
                                       BLOCK=BLOCK,
                                       BLOCK_N=BLOCK_N,
                                       **kernel_meta)

    return outputs
