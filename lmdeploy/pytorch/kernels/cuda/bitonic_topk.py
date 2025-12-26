# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from triton.language import core
from triton.language.standard import _log2


@triton.jit
def _indicator(n_dims: core.constexpr, j: core.constexpr):
    ar = core.arange(0, 2)
    ar = core.reshape(ar, [1] * (n_dims - j - 1) + [2] + [1] * j)
    return ar


@triton.jit
def _flip_along_middle(x, n_dims, i):
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ix = x.to(idtype, bitcast=True)
    iy = ix ^ tl.xor_sum(ix, n_dims - 1 - i, True)
    y = iy.to(x.dtype, bitcast=True)
    return y


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr):
    # compare-and-swap on the ith *innermost* dimension
    n_dims: core.constexpr = _log2(x.numel)

    # determines whether we are in the right (rather than left) position along the axis:
    is_right = _indicator(n_dims, i)

    # flip along middle dimension (the bitwise XORs will be optimised away):
    y = _flip_along_middle(x, n_dims, i)
    ids_y = _flip_along_middle(ids, n_dims, i)

    # conditional swap:
    mask = (x > y) != (flip ^ is_right)
    ret_x = core.where(mask, y, x)
    ret_ids = core.where(mask, ids_y, ids)
    return ret_x, ret_ids


@triton.jit
def _bitonic_merge_hypercube(x, ids, stage: core.constexpr, order: core.constexpr):
    """order_type 0 == ascending order_type 1 == descending order_type 2 ==
    alternating."""
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        flip = _indicator(_log2(x.numel), stage)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, stage - 1 - i)
    return x, ids


@triton.jit
def _bitonic_merge(x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    """order_type 0 == ascending order_type 1 == descending order_type 2 ==
    alternating."""
    h = core.reshape(x, [2] * _log2(x.numel))
    h_ids = core.reshape(ids, [2] * _log2(x.numel))
    h, h_ids = _bitonic_merge_hypercube(h, h_ids, stage, order)
    x = core.reshape(h, x.shape)
    ids = core.reshape(h_ids, ids.shape)
    return x, ids


@triton.jit
def argsort(x, ids, dim: tl.constexpr = None, descending: tl.constexpr = core.CONSTEXPR_0):
    # handle default dimension or check that it is the most minor dim
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(_dim == len(x.shape) - 1, 'only minor dimension is currently supported')
    # iteratively run bitonic merge-sort steps
    n_dims: tl.constexpr = _log2(x.shape[_dim])

    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


@triton.jit
def _bitonic_topk_kernel0(score_ptr,
                          seqlen_ptr,
                          out_ptr,
                          ids_ptr,
                          stride_m,
                          K: tl.constexpr,
                          fill: tl.constexpr,
                          descending: tl.constexpr = core.CONSTEXPR_0,
                          sorted: tl.constexpr = True):
    """kernel0."""
    batch_id = tl.program_id(0).to(tl.int64)
    block_id = tl.program_id(1).to(tl.int64)

    seqlen = tl.load(seqlen_ptr + batch_id)

    if block_id * K >= seqlen:
        return

    offs_k = tl.arange(0, K)
    origin_ids = block_id * K + offs_k
    # num scores should less than max(int32), I guess
    origin_ids = origin_ids.to(tl.int32)
    mask = (origin_ids < seqlen)
    score_ptrs = score_ptr + batch_id * stride_m + origin_ids
    scores = tl.load(score_ptrs, mask=mask, other=-1e6)
    ids = tl.where(mask, origin_ids, fill)
    ids = origin_ids

    if sorted or (seqlen > K):
        scores, ids = argsort(scores, ids, 0, descending)

    tl.store(out_ptr + batch_id * stride_m + origin_ids, scores, mask=mask)
    tl.store(ids_ptr + batch_id * stride_m + origin_ids, ids, mask=mask)


@triton.jit
def _concate(a, b):
    """concate."""
    c = tl.join(a, b)  # [k, 2]
    c = c.trans()  # [2, k]
    # there are bugs in `tr.ravel` when triton<=3.2.0
    c = tl.reshape(c, (a.numel + b.numel, ))
    return c


@triton.jit
def _split(a, k):
    """split."""
    a = a.reshape(2, k)
    a = a.trans()
    return tl.split(a)


@triton.jit
def _bitonic_topk_kernel1(score_ptr,
                          ids_ptr,
                          seqlen_ptr,
                          out_ptr,
                          stride_m,
                          K: tl.constexpr,
                          fill: tl.constexpr,
                          threshold: tl.constexpr,
                          descending: tl.constexpr = core.CONSTEXPR_0):
    """kernel1."""
    batch_id = tl.program_id(0).to(tl.int64)

    seqlen = tl.load(seqlen_ptr + batch_id)
    offs_k = tl.arange(0, K)
    score_ptrs = score_ptr + batch_id * stride_m + offs_k
    ids_ptrs = ids_ptr + batch_id * stride_m + offs_k

    # initialize
    pos = offs_k
    mask = pos < seqlen
    scores = tl.load(score_ptrs, mask=mask, other=threshold)
    ids = tl.load(ids_ptrs, mask=mask, other=fill)

    pos = 2 * K - 1 - offs_k
    score_ptrs = score_ptr + batch_id * stride_m + pos
    ids_ptrs = ids_ptr + batch_id * stride_m + pos

    stage: tl.constexpr = _log2(2 * K)
    for k in tl.range(K, seqlen, K, num_stages=3):
        mask = pos < seqlen
        new_scores = tl.load(score_ptrs, mask=mask, other=threshold)
        new_ids = tl.load(ids_ptrs, mask=mask, other=fill)

        merged_scores = _concate(scores, new_scores)
        merged_ids = _concate(ids, new_ids)

        merged_scores, merged_ids = _bitonic_merge(merged_scores, merged_ids, stage, descending, stage)

        scores, _ = _split(merged_scores, K)
        ids, _ = _split(merged_ids, K)
        score_ptrs += K
        ids_ptrs += K
        pos += K

    out_ptrs = out_ptr + batch_id * K + offs_k
    ids = tl.where(scores <= threshold, fill, ids)
    tl.store(out_ptrs, ids)


def bitonic_topk(scores: torch.Tensor,
                 q_seqlens: torch.Tensor,
                 kv_seqlens: torch.Tensor,
                 k: int,
                 fill: int = -1,
                 descending: bool = True,
                 sorted: bool = True,
                 threshold: float = -1e6):
    """Bitnoic topk."""
    num_tokens = scores.size(0)
    max_kv_len = scores.size(-1)
    assert max_kv_len < (1 << 31)

    if num_tokens != kv_seqlens.size(0):
        repeat_kv_seqlens = torch.repeat_interleave(kv_seqlens, q_seqlens, output_size=num_tokens)
    else:
        repeat_kv_seqlens = kv_seqlens
    tmp_scores = torch.empty_like(scores)
    tmp_ids = torch.empty_like(scores, dtype=torch.int32)
    num_warps = triton.cdiv(k, 4096)
    grid = (num_tokens, triton.cdiv(max_kv_len, k))
    _bitonic_topk_kernel0[grid](scores,
                                repeat_kv_seqlens,
                                tmp_scores,
                                tmp_ids,
                                stride_m=scores.stride(0),
                                K=k,
                                fill=fill,
                                descending=1 if descending else 0,
                                sorted=sorted,
                                num_warps=num_warps)

    out = kv_seqlens.new_empty((num_tokens, k), dtype=torch.int32)
    _bitonic_topk_kernel1[(num_tokens, )](tmp_scores,
                                          tmp_ids,
                                          repeat_kv_seqlens,
                                          out,
                                          stride_m=tmp_scores.stride(0),
                                          K=k,
                                          fill=fill,
                                          descending=1 if descending else 0,
                                          threshold=threshold,
                                          num_warps=num_warps * 2)
    return out
