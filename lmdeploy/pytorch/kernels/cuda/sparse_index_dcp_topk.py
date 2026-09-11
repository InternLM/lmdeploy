# Copyright (c) OpenMMLab. All rights reserved.
"""DCP top-k kernels for token-sharded sparse indexers."""

from __future__ import annotations

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl

from .sparse_index_topk import _sparse_index_topk

tilelang.set_log_level('WARNING')

_FILL = -1
_THREADS = 1024
_RADIX_BITS = 8
_RADIX_SIZE = 1 << _RADIX_BITS
_STATE_SELECTED_BIN = 0
_STATE_COUNT_PRIOR = 1
_STATE_VALID_COUNT = 2

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
    return T.if_then_else(T.bitwise_and(bits, sign_mask) == T.cast(0, T.uint32),
                          T.bitwise_xor(bits, sign_mask),
                          T.bitwise_xor(bits, all_ones))


def sparse_dcp_local_topk(scores: torch.Tensor,
                          q_seqlens: torch.Tensor,
                          kv_seqlens: torch.Tensor,
                          k: int,
                          fill: int = _FILL,
                          descending: bool = True,
                          sorted: bool = False) -> torch.Tensor:
    """Select deterministic local candidates for an exact DCP merge."""
    return _sparse_index_topk(scores,
                              q_seqlens,
                              kv_seqlens,
                              k,
                              fill=fill,
                              descending=descending,
                              sorted=sorted,
                              stable_ties=True)


@triton.jit
def _pack_dcp_topk_candidates_kernel(
    Scores,
    LocalIndices,
    Packed,
    score_width,
    stride_sr,
    stride_sc,
    stride_ir,
    stride_ic,
    stride_pr,
    stride_pc,
    stride_pp,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Decode graph score buffers can exceed 2**31 elements with a large KV pool.
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    column_mask = columns < top_k
    local_indices = tl.load(LocalIndices + row * stride_ir +
                            columns * stride_ic,
                            mask=column_mask,
                            other=-1).to(tl.int32)
    valid = local_indices >= 0
    safe_indices = tl.maximum(local_indices, 0)
    safe_indices = tl.minimum(safe_indices, score_width - 1)
    scores = tl.load(Scores + row * stride_sr + safe_indices * stride_sc,
                     mask=column_mask & valid,
                     other=-float('inf')).to(tl.float32)
    global_indices = safe_indices * dcp_size + dcp_rank
    global_indices = tl.where(valid, global_indices, -1).to(tl.int32)

    packed = Packed + row * stride_pr + columns * stride_pc
    tl.store(packed, scores, mask=column_mask)
    tl.store(packed + stride_pp,
             tl.cast(global_indices, tl.float32, bitcast=True),
             mask=column_mask)


def pack_dcp_topk_candidates(scores: torch.Tensor, local_indices: torch.Tensor,
                             *, dcp_world_rank: tuple[int, int]) -> torch.Tensor:
    """Pack FP32 scores and bit-preserved INT32 global ids together."""
    assert scores.dim() == 2 and local_indices.dim() == 2
    assert scores.size(0) == local_indices.size(0)
    assert scores.dtype == torch.float32
    assert local_indices.dtype == torch.int32
    dcp_size, dcp_rank = dcp_world_rank
    packed = torch.empty((*local_indices.shape, 2),
                         dtype=torch.float32,
                         device=local_indices.device)
    block = 512
    grid = (local_indices.size(0), triton.cdiv(local_indices.size(1), block))
    _pack_dcp_topk_candidates_kernel[grid](
        scores,
        local_indices,
        packed,
        scores.size(1),
        *scores.stride(),
        *local_indices.stride(),
        *packed.stride(),
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        top_k=local_indices.size(1),
        BLOCK=block,
        num_warps=8,
    )
    return packed


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _sparse_dcp_global_topk_kernel(top_k: int,
                                      dcp_size: int,
                                      fill: int = _FILL,
                                      threads: int = _THREADS):
    """Select exact stable top-k global ids from rank-major candidates."""
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def sparse_dcp_global_topk_kernel_(
        Candidates: T.Tensor[(dcp_size, num_tokens, top_k, 2), T.float32],
        Out: T.Tensor[(num_tokens, top_k), T.int32],
    ):
        num_candidates = dcp_size * top_k
        with T.Kernel(num_tokens, threads=threads) as row:
            tidx = T.get_thread_binding(0)
            histogram = T.alloc_shared((_RADIX_SIZE, ), T.int32)
            state = T.alloc_shared((3, ), T.int32)
            rank_counts = T.alloc_shared((dcp_size, ), T.int32)
            rank_offsets = T.alloc_shared((dcp_size, ), T.int32)
            emit_offsets = T.alloc_shared((threads, ), T.int32)

            for output_idx in T.Parallel(top_k):
                Out[row, output_idx] = fill
            if tidx < 3:
                state[tidx] = 0
            if tidx < dcp_size:
                rank_counts[tidx] = 0
            T.sync_threads()

            candidate = T.alloc_var(T.int32)
            candidate = tidx
            while candidate < num_candidates:
                rank = candidate // top_k
                slot = candidate % top_k
                global_id = T.reinterpret(Candidates[rank, row, slot, 1],
                                          T.int32)
                if global_id >= 0:
                    T.atomic_add(state[_STATE_VALID_COUNT], 1)
                    T.atomic_add(rank_counts[rank], 1)
                candidate += threads
            T.sync_threads()

            valid_count = state[_STATE_VALID_COUNT]
            if valid_count <= top_k:
                # Candidates are packed within every rank. Prefixing the
                # per-rank counts gives each valid id a deterministic slot.
                if tidx == 0:
                    rank_offsets[0] = 0
                    for rank_id in T.serial(1, dcp_size):
                        rank_offsets[rank_id] = (
                            rank_offsets[rank_id - 1]
                            + rank_counts[rank_id - 1])
                T.sync_threads()
                candidate = tidx
                while candidate < num_candidates:
                    rank = candidate // top_k
                    slot = candidate % top_k
                    global_id = T.reinterpret(Candidates[rank, row, slot, 1],
                                              T.int32)
                    if global_id >= 0:
                        Out[row, rank_offsets[rank] + slot] = global_id
                    candidate += threads
            else:
                # The composite (score, -global_id) key makes ties exact.
                score_prefix = T.alloc_var(T.uint32)
                score_mask = T.alloc_var(T.uint32)
                id_prefix = T.alloc_var(T.uint32)
                id_mask = T.alloc_var(T.uint32)
                selected_rank = T.alloc_var(T.int32)
                key = T.alloc_var(T.uint32)
                matches_prefix = T.alloc_var(T.bool)
                score_prefix = T.cast(0, T.uint32)
                score_mask = T.cast(0, T.uint32)
                id_prefix = T.cast(0, T.uint32)
                id_mask = T.cast(0, T.uint32)
                selected_rank = top_k

                for round_idx in T.Unroll(8):
                    key_round = T.if_then_else(round_idx < 4, round_idx,
                                               round_idx - 4)
                    shift = 24 - key_round * _RADIX_BITS
                    byte_mask = T.cast(255, T.uint32) << shift
                    for bin_idx in T.Parallel(_RADIX_SIZE):
                        histogram[bin_idx] = 0

                    candidate = tidx
                    while candidate < num_candidates:
                        rank = candidate // top_k
                        slot = candidate % top_k
                        score = Candidates[rank, row, slot, 0]
                        global_id = T.reinterpret(
                            Candidates[rank, row, slot, 1], T.int32)
                        score_key = _ordered_fp32_key(score)
                        id_key = T.bitwise_xor(
                            T.cast(global_id, T.uint32),
                            T.cast(4294967295, T.uint32))
                        if round_idx < 4:
                            key = score_key
                            matches_prefix = T.bitwise_and(
                                score_key, score_mask) == score_prefix
                        else:
                            key = id_key
                            matches_prefix = (
                                score_key == score_prefix
                                and T.bitwise_and(id_key, id_mask) == id_prefix)
                        if global_id >= 0 and matches_prefix:
                            bin_u32 = T.bitwise_and(key >> shift,
                                                    T.cast(255, T.uint32))
                            T.atomic_add(histogram[T.cast(bin_u32, T.int32)],
                                         1)
                        candidate += threads
                    T.sync_threads()

                    if tidx < _RADIX_SIZE:
                        greater_count = T.alloc_var(T.int32)
                        greater_count = 0
                        for other_bin in T.serial(0, _RADIX_SIZE):
                            if other_bin > tidx:
                                greater_count += histogram[other_bin]
                        bin_count = histogram[tidx]
                        if (greater_count < selected_rank and
                                greater_count + bin_count >= selected_rank):
                            state[_STATE_SELECTED_BIN] = tidx
                            state[_STATE_COUNT_PRIOR] = greater_count
                    T.sync_threads()

                    selected_bin = state[_STATE_SELECTED_BIN]
                    if round_idx < 4:
                        score_prefix = T.bitwise_or(
                            score_prefix,
                            T.cast(selected_bin, T.uint32) << shift)
                        score_mask = T.bitwise_or(score_mask, byte_mask)
                    else:
                        id_prefix = T.bitwise_or(
                            id_prefix,
                            T.cast(selected_bin, T.uint32) << shift)
                        id_mask = T.bitwise_or(id_mask, byte_mask)
                    selected_rank -= state[_STATE_COUNT_PRIOR]

                # Emit in candidate order so equal input rows remain identical.
                segment_start = num_candidates * tidx // threads
                segment_end = num_candidates * (tidx + 1) // threads
                local_win_count = T.alloc_var(T.int32)
                local_win_count = 0
                candidate = segment_start
                while candidate < segment_end:
                    rank = candidate // top_k
                    slot = candidate % top_k
                    score = Candidates[rank, row, slot, 0]
                    global_id = T.reinterpret(Candidates[rank, row, slot, 1],
                                              T.int32)
                    score_key = _ordered_fp32_key(score)
                    id_key = T.bitwise_xor(
                        T.cast(global_id, T.uint32),
                        T.cast(4294967295, T.uint32))
                    wins = (score_key > score_prefix or
                            (score_key == score_prefix
                             and id_key >= id_prefix))
                    if global_id >= 0 and wins:
                        local_win_count += 1
                    candidate += 1
                emit_offsets[tidx] = local_win_count
                T.sync_threads()

                if tidx == 0:
                    running_win_count = T.alloc_var(T.int32)
                    running_win_count = 0
                    for thread_id in T.serial(0, threads):
                        thread_win_count = emit_offsets[thread_id]
                        emit_offsets[thread_id] = running_win_count
                        running_win_count += thread_win_count
                T.sync_threads()

                local_win_count = 0
                candidate = segment_start
                while candidate < segment_end:
                    rank = candidate // top_k
                    slot = candidate % top_k
                    score = Candidates[rank, row, slot, 0]
                    global_id = T.reinterpret(Candidates[rank, row, slot, 1],
                                              T.int32)
                    score_key = _ordered_fp32_key(score)
                    id_key = T.bitwise_xor(
                        T.cast(global_id, T.uint32),
                        T.cast(4294967295, T.uint32))
                    wins = (score_key > score_prefix or
                            (score_key == score_prefix
                             and id_key >= id_prefix))
                    if global_id >= 0 and wins:
                        output_idx = emit_offsets[tidx] + local_win_count
                        Out[row, output_idx] = global_id
                        local_win_count += 1
                    candidate += 1

    return sparse_dcp_global_topk_kernel_


def sparse_dcp_global_topk(gathered_candidates: torch.Tensor,
                              k: int,
                              fill: int = _FILL) -> torch.Tensor:
    """Select stable global top-k ids from packed DCP candidates."""
    assert gathered_candidates.dim() == 4
    dcp_size, num_tokens, local_k, pair_width = gathered_candidates.shape
    assert local_k == k and pair_width == 2
    assert gathered_candidates.dtype == torch.float32
    output = torch.empty((num_tokens, k),
                         dtype=torch.int32,
                         device=gathered_candidates.device)
    _sparse_dcp_global_topk_kernel(k, dcp_size, fill,
                                      _THREADS)(gathered_candidates, output)
    return output
