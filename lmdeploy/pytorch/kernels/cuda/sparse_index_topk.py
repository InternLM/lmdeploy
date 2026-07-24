# Copyright (c) OpenMMLab. All rights reserved.
"""Shared sparse-index top-k kernels for MLA/NSA indexers.

The CUDA graph constraint is the important bit here: score width is a padded
graph bucket, but the real per-row length is carried by ``kv_seqlens``.  This
module therefore specializes only on model-config ``K`` and keeps the row
length as device data.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T
import torch

tilelang.set_log_level('WARNING')

_SUPPORTED_TOPK = (512, 2048)
_FILL = -1
_THREADS = 1024
_RADIX_BITS = 8
_RADIX_SIZE = 1 << _RADIX_BITS
_STATE_SELECTED_BIN = 0
_STATE_COUNT_ABOVE = 1
_STATE_EMIT_GT_COUNT = 0
_STATE_EMIT_EQ_COUNT = 1

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


def _ordered_fp32_key(score):
    """Map fp32 to an integer key whose unsigned order matches fp32 order."""
    bits = T.reinterpret(score, T.uint32)
    sign_mask = T.cast(2147483648, T.uint32)
    all_ones = T.cast(4294967295, T.uint32)
    return T.if_then_else(
        T.bitwise_and(bits, sign_mask) == T.cast(0, T.uint32),
        T.bitwise_xor(bits, sign_mask), T.bitwise_xor(bits, all_ones))


def is_sparse_index_topk_supported(k: int) -> bool:
    """Return whether the TileLang byte-radix path has a compiled
    specialization."""
    return k in _SUPPORTED_TOPK


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _sparse_index_topk_byte_radix_kernel(top_k: int,
                                         fill: int = _FILL,
                                         threads: int = _THREADS):
    num_tokens = T.dynamic('num_tokens')
    score_width = T.dynamic('score_width')
    score_stride = T.dynamic('score_stride')

    @T.prim_func
    def sparse_index_topk_byte_radix_kernel_(
        Scores: T.StridedTensor[(num_tokens, score_width), (score_stride, 1),
                                T.float32],
        Seqlens: T.Tensor[(num_tokens, ), T.int32],
        Out: T.Tensor[(num_tokens, top_k), T.int32],
    ):
        _ = score_stride
        with T.Kernel(num_tokens, threads=threads) as row:
            tidx = T.get_thread_binding(0)
            histogram = T.alloc_shared((_RADIX_SIZE, ), T.int32)
            shared_state = T.alloc_shared((2, ), T.int32)

            raw_seqlen = Seqlens[row]
            seqlen = T.if_then_else(raw_seqlen < score_width, raw_seqlen,
                                    score_width)

            # If the real row length fits inside top_k, every valid position
            # is selected.
            if seqlen <= top_k:
                for i in T.Parallel(top_k):
                    Out[row, i] = T.if_then_else(i < seqlen, i, fill)
            else:
                # Byte-radix threshold search. Each round fixes one byte of the
                # fp32-order key and narrows the selected prefix until the final
                # threshold key is known.
                prefix_key = T.alloc_var(T.uint32)
                prefix_mask = T.alloc_var(T.uint32)
                rank = T.alloc_var(T.int32)
                prefix_key = T.cast(0, T.uint32)
                prefix_mask = T.cast(0, T.uint32)
                rank = top_k

                for round_idx in T.Unroll(4):
                    shift = 24 - round_idx * _RADIX_BITS
                    byte_mask = T.cast(255, T.uint32) << shift

                    for bin_idx in T.Parallel(_RADIX_SIZE):
                        histogram[bin_idx] = 0

                    pos = T.alloc_var(T.int32)
                    pos = tidx
                    while pos < seqlen:
                        key = _ordered_fp32_key(Scores[row, pos])
                        if T.bitwise_and(key, prefix_mask) == prefix_key:
                            bin_u32 = T.bitwise_and(key >> shift,
                                                    T.cast(255, T.uint32))
                            T.atomic_add(histogram[T.cast(bin_u32, T.int32)],
                                         1)
                        pos += threads

                    T.sync_threads()

                    if tidx < _RADIX_SIZE:
                        above_count = T.alloc_var(T.int32)
                        above_count = 0
                        for other_bin in T.serial(0, _RADIX_SIZE):
                            if other_bin > tidx:
                                above_count += histogram[other_bin]
                        bin_count = histogram[tidx]
                        if above_count < rank and above_count + bin_count >= rank:
                            shared_state[_STATE_SELECTED_BIN] = tidx
                            shared_state[_STATE_COUNT_ABOVE] = above_count

                    T.sync_threads()

                    threshold_bin = shared_state[_STATE_SELECTED_BIN]
                    prefix_key = T.bitwise_or(
                        prefix_key,
                        T.cast(threshold_bin, T.uint32) << shift)
                    prefix_mask = T.bitwise_or(prefix_mask, byte_mask)
                    rank -= shared_state[_STATE_COUNT_ABOVE]

                threshold_key = prefix_key

                # Reuse shared_state as output counters after threshold search:
                # [0] counts scores above threshold, [1] counts scores equal to it.
                if tidx == 0:
                    shared_state[_STATE_EMIT_GT_COUNT] = 0
                    shared_state[_STATE_EMIT_EQ_COUNT] = 0
                T.sync_threads()

                out_pos_buf = T.alloc_local((1, ), T.int32)
                # First emit all scores strictly greater than the threshold.
                pos_emit_gt = T.alloc_var(T.int32)
                pos_emit_gt = tidx
                while pos_emit_gt < seqlen:
                    key = _ordered_fp32_key(Scores[row, pos_emit_gt])
                    if key > threshold_key:
                        out_pos_buf[0] = T.atomic_add(
                            shared_state[_STATE_EMIT_GT_COUNT],
                            1,
                            return_prev=True)
                        if out_pos_buf[0] < top_k:
                            Out[row, out_pos_buf[0]] = pos_emit_gt
                    pos_emit_gt += threads

                T.sync_threads()

                gt_count = shared_state[_STATE_EMIT_GT_COUNT]

                # Then fill remaining slots from scores equal to the threshold.
                # Tie order is intentionally unspecified; sparse attention needs
                # a valid top-k set, not score-sorted ids.
                pos_emit_eq = T.alloc_var(T.int32)
                pos_emit_eq = tidx
                while pos_emit_eq < seqlen:
                    key = _ordered_fp32_key(Scores[row, pos_emit_eq])
                    if key == threshold_key:
                        out_pos_buf[0] = gt_count + T.atomic_add(
                            shared_state[_STATE_EMIT_EQ_COUNT],
                            1,
                            return_prev=True)
                        if out_pos_buf[0] < top_k:
                            Out[row, out_pos_buf[0]] = pos_emit_eq
                    pos_emit_eq += threads

    return sparse_index_topk_byte_radix_kernel_


def sparse_index_topk(scores: torch.Tensor,
                      q_seqlens: torch.Tensor,
                      kv_seqlens: torch.Tensor,
                      k: int,
                      fill: int = _FILL,
                      descending: bool = True,
                      sorted: bool = False) -> torch.Tensor:
    """Return top-k score indices for padded sparse-index score rows.

    The returned ids are packed but not score-sorted. Sparse attention
    consumes them as a set of valid KV positions; avoiding final sorting is the
    point of this selector. Rows shorter than ``k`` are padded with ``fill``.
    """
    if not descending:
        raise ValueError('sparse_index_topk only supports descending=True.')
    if sorted:
        raise ValueError('sparse_index_topk does not support sorted output.')
    if not is_sparse_index_topk_supported(k):
        raise ValueError(
            f'sparse_index_topk supports k in {_SUPPORTED_TOPK}, got {k}.')
    assert scores.is_cuda and q_seqlens.is_cuda and kv_seqlens.is_cuda
    assert scores.dtype == torch.float32
    assert q_seqlens.dtype in (torch.int32, torch.int64)
    assert kv_seqlens.dtype in (torch.int32, torch.int64)
    assert scores.stride(-1) == 1

    kv_seqlens = kv_seqlens.to(torch.int32)
    num_tokens = scores.size(0)
    if num_tokens != kv_seqlens.size(0):
        kv_seqlens = torch.repeat_interleave(kv_seqlens,
                                             q_seqlens,
                                             output_size=num_tokens)
    else:
        kv_seqlens = kv_seqlens.contiguous()

    out = torch.empty((num_tokens, k), device=scores.device, dtype=torch.int32)
    _sparse_index_topk_byte_radix_kernel(k, fill, _THREADS)(scores, kv_seqlens,
                                                            out)
    return out
