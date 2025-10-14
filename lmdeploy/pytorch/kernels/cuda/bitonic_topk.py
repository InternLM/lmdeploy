# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from triton.language import core
from triton.language.standard import _log2


@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)

    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape)
    right_idx = tl.reshape(right_idx, x.shape)

    # actual compare-and-swap
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip
    cond = cond.to(tl.int1)

    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))

    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))

    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    """order_type 0 == ascending order_type 1 == descending order_type 2 ==
    alternating."""
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
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
                          stride_m: tl.constexpr,
                          K: tl.constexpr,
                          fill: tl.constexpr,
                          descending: tl.constexpr = core.CONSTEXPR_0):
    """kernel0."""
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    seqlen = tl.load(seqlen_ptr + batch_id)

    if block_id * K >= seqlen:
        return

    offs_k = tl.arange(0, K)
    origin_ids = block_id * K + offs_k
    mask = (origin_ids < seqlen)
    score_ptrs = score_ptr + batch_id * stride_m + origin_ids
    scores = tl.load(score_ptrs, mask=mask, other=-1e6)
    ids = tl.where(mask, origin_ids, fill)
    ids = origin_ids

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
                          stride_m: tl.constexpr,
                          K: tl.constexpr,
                          fill: tl.constexpr,
                          descending: tl.constexpr = core.CONSTEXPR_0):
    """kernel1."""
    batch_id = tl.program_id(0)

    seqlen = tl.load(seqlen_ptr + batch_id)
    offs_k = tl.arange(0, K)
    score_ptrs = score_ptr + batch_id * stride_m + offs_k
    ids_ptrs = ids_ptr + batch_id * stride_m + offs_k

    # initialize
    pos = offs_k
    mask = pos < seqlen
    scores = tl.load(score_ptrs, mask=mask, other=-1e6)
    ids = tl.load(ids_ptrs, mask=mask, other=fill)

    pos = 2 * K - 1 - offs_k
    score_ptrs = score_ptr + batch_id * stride_m + pos
    ids_ptrs = ids_ptr + batch_id * stride_m + pos

    stage: tl.constexpr = _log2(2 * K)
    for k in tl.range(K, seqlen, K, num_stages=3):
        mask = pos < seqlen
        new_scores = tl.load(score_ptrs, mask=mask, other=-1e6)
        new_ids = tl.load(ids_ptrs, mask=mask, other=fill)

        merged_scores = _concate(scores, new_scores)
        merged_ids = _concate(ids, new_ids)

        merged_scores, merged_ids = _bitonic_merge(merged_scores, merged_ids, stage, descending, stage)
        # merged_scores, merged_ids = argsort(merged_scores, merged_ids, 0, descending)

        scores, _ = _split(merged_scores, K)
        ids, _ = _split(merged_ids, K)
        score_ptrs += K
        ids_ptrs += K
        pos += K

    out_ptrs = out_ptr + batch_id * K + offs_k
    tl.store(out_ptrs, ids)


def bitonic_topk(scores: torch.Tensor,
                 q_seqlens: torch.Tensor,
                 kv_seqlens: torch.Tensor,
                 k: int,
                 fill: int = -1,
                 descending: bool = True):
    """Bitnoic topk."""
    num_tokens = scores.size(0)
    max_kv_len = scores.size(-1)

    if num_tokens != kv_seqlens.size(0):
        repeat_kv_seqlens = torch.repeat_interleave(kv_seqlens, q_seqlens, output_size=num_tokens)
    else:
        repeat_kv_seqlens = kv_seqlens
    tmp_scores = torch.empty_like(scores)
    tmp_ids = torch.empty_like(scores, dtype=torch.int32)
    grid = (num_tokens, triton.cdiv(max_kv_len, k))
    _bitonic_topk_kernel0[grid](scores,
                                repeat_kv_seqlens,
                                tmp_scores,
                                tmp_ids,
                                stride_m=scores.stride(0),
                                K=k,
                                fill=fill,
                                descending=1 if descending else 0,
                                num_warps=4)

    out = kv_seqlens.new_empty((num_tokens, k), dtype=torch.int32)
    _bitonic_topk_kernel1[(num_tokens, )](tmp_scores,
                                          tmp_ids,
                                          repeat_kv_seqlens,
                                          out,
                                          stride_m=tmp_scores.stride(0),
                                          K=k,
                                          fill=fill,
                                          descending=1 if descending else 0,
                                          num_warps=4)
    return out
