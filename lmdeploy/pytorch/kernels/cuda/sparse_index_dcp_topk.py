# Copyright (c) OpenMMLab. All rights reserved.
"""Merge token-sharded sparse-indexer candidates without gathering full scores.

With DCP, ordinary ``sparse_index_topk`` sees only one rank's KV shard. Each
rank must contribute up to K candidates, not K / DCP: the best tokens may all
belong to one shard. A locally discarded token has at least K local tokens
with no lower score, so the candidate union suffices for a valid global top-k.

The caller packs local scores and global token ids, all-gathers these pairs,
then selects K ids from DCP * K candidates here. Collectives stay in the
backend; these kernels handle packing and selection only.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl

from .sparse_index_topk import _PASS_CONFIGS, _ordered_fp32_key

tilelang.set_log_level('WARNING')

_FILL = -1
_THREADS = 512
_STATE_SELECTED_BIN = 0
_STATE_COUNT_PRIOR = 1
_STATE_BIN_COUNT = 2


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
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Decode graph score buffers can exceed 2**31 elements with a large KV pool.
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    column_mask = columns < TOP_K
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
    # Undo token-interleaved ownership before candidates cross rank boundaries.
    global_indices = safe_indices * DCP_SIZE + DCP_RANK
    global_indices = tl.where(valid, global_indices, -1).to(tl.int32)

    packed = Packed + row * stride_pr + columns * stride_pc
    tl.store(packed, scores, mask=column_mask)
    # Share one FP32 collective buffer without rounding INT32 ids above 2**24.
    # The receiver reinterprets the id lane; it must not numerically cast it.
    tl.store(packed + stride_pp,
             tl.cast(global_indices, tl.float32, bitcast=True),
             mask=column_mask)


def pack_dcp_topk_candidates(scores: torch.Tensor, local_indices: torch.Tensor,
                             *, dcp_world_rank: tuple[int, int]) -> torch.Tensor:
    """Pack FP32 scores and bit-preserved INT32 global ids for one all-gather.

    Local ids are request-relative positions in this rank's interleaved shard.
    Return ``[rows, K, 2]`` FP32 storage; invalid candidates carry ``(-inf, -1)``
    with the id lane stored as INT32 bits, not as a floating-point number.
    """
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
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        TOP_K=local_indices.size(1),
        BLOCK=block,
        num_warps=8,
    )
    return packed


@T.macro
def _block_prefix_sum(value, warp_totals, threads):
    """Block-wide inclusive scan; shared scratch also holds cumulative warp
    totals."""
    tidx = T.get_thread_binding(0)
    lane = tidx % 32
    prefix = T.alloc_var(T.int32)
    prefix = value
    for step in T.Unroll(5):
        other = T.shfl_sync(prefix, T.max(lane - (1 << step), 0))
        if lane >= (1 << step):
            prefix += other
    if lane == 31:
        warp_totals[tidx // 32] = prefix
    T.sync_threads()
    if tidx < 32:
        total = T.alloc_var(T.int32)
        total = T.if_then_else(lane < threads // 32, warp_totals[lane], 0)
        for step in T.Unroll(5):
            other = T.shfl_sync(total, T.max(lane - (1 << step), 0))
            if lane >= (1 << step):
                total += other
        if lane < threads // 32:
            warp_totals[lane] = total
    T.sync_threads()
    if tidx >= 32:
        prefix += warp_totals[tidx // 32 - 1]
    T.sync_threads()
    return prefix


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _sparse_dcp_global_topk_kernel(top_k: int,
                                 dcp_size: int,
                                 fill: int = _FILL,
                                 threads: int = _THREADS):
    """Select stable top-k global ids among the gathered candidates."""
    num_tokens = T.dynamic('num_tokens')
    # Eleven bits per pass cover the composite (score, -id) key in six passes.
    radix_bits = 11
    radix_size = 1 << radix_bits

    @T.prim_func
    def sparse_dcp_global_topk_kernel_(
        Candidates: T.Tensor[(dcp_size, num_tokens, top_k, 2), T.float32],
        Out: T.Tensor[(num_tokens, top_k), T.int32],
    ):
        num_candidates = dcp_size * top_k
        keys_per_thread = num_candidates // threads
        with T.Kernel(num_tokens, threads=threads) as row:
            tidx = T.get_thread_binding(0)
            keys = T.alloc_local((keys_per_thread,), T.uint64)
            histogram = T.alloc_shared((radix_size,), T.int32)
            histogram_prefix = T.alloc_shared((radix_size,), T.int32)
            state = T.alloc_shared((3,), T.int32)
            warp_totals = T.alloc_shared((threads // 32,), T.int32)

            for output_idx in T.Parallel(top_k):
                Out[row, output_idx] = fill
            # Each warp owns a contiguous segment; adjacent lanes load adjacent
            # candidates. Keep the keys live across radix passes.
            local_count = T.alloc_var(T.int32)
            local_count = 0
            for i in T.Unroll(keys_per_thread):
                candidate = (tidx // 32) * (32 * keys_per_thread) + i * 32 + tidx % 32
                rank = candidate // top_k
                slot = candidate % top_k
                global_id = T.reinterpret(Candidates[rank, row, slot, 1], T.int32)
                score_key = _ordered_fp32_key(Candidates[rank, row, slot, 0])
                id_key = T.bitwise_xor(T.cast(global_id, T.uint32), T.cast(0xFFFFFFFF, T.uint32))
                keys[i] = T.if_then_else(
                    global_id >= 0,
                    (T.cast(score_key, T.uint64) << 32) | T.cast(id_key, T.uint64),
                    T.cast(0, T.uint64))
                local_count += T.cast(global_id >= 0, T.int32)
            output_end = T.alloc_var(T.int32)
            output_end = _block_prefix_sum(local_count, warp_totals, threads)
            # Preserve the branch predicate before histogram scans reuse the scratch.
            valid_count = T.alloc_var(T.int32)
            valid_count = warp_totals[threads // 32 - 1]

            prefix = T.alloc_var(T.uint64)
            prefix_mask = T.alloc_var(T.uint64)
            selected_rank = T.alloc_var(T.int32)
            prefix = T.cast(0, T.uint64)
            prefix_mask = T.cast(0, T.uint64)
            selected_rank = top_k
            if valid_count > top_k:
                round_idx = T.alloc_var(T.int32)
                round_idx = 0
                while round_idx < 6:
                    shift = T.max(64 - (round_idx + 1) * radix_bits, 0)
                    for bin_idx in T.Parallel(radix_size):
                        histogram[bin_idx] = 0
                    T.sync_threads()
                    for i in T.Unroll(keys_per_thread):
                        key = keys[i]
                        if key != 0 and (key & prefix_mask) == prefix:
                            bin_idx = T.cast((key >> shift) & T.cast(radix_size - 1, T.uint64), T.int32)
                            T.atomic_add(histogram[bin_idx], 1)
                    T.sync_threads()
                    bin_total = T.alloc_var(T.int32)
                    bin_total = 0
                    for i in T.Unroll(radix_size // threads):
                        bin_total += histogram[tidx * (radix_size // threads) + i]
                    bin_end = _block_prefix_sum(bin_total, warp_totals, threads)
                    bin_prefix = T.alloc_var(T.int32)
                    bin_prefix = bin_end - bin_total
                    for i in T.Unroll(radix_size // threads):
                        bin_idx = tidx * (radix_size // threads) + i
                        bin_prefix += histogram[bin_idx]
                        histogram_prefix[bin_idx] = bin_prefix
                    T.sync_threads()
                    for bin_idx in T.Parallel(radix_size):
                        greater_count = histogram_prefix[radix_size - 1] - histogram_prefix[bin_idx]
                        bin_count = histogram[bin_idx]
                        if greater_count < selected_rank and greater_count + bin_count >= selected_rank:
                            state[_STATE_SELECTED_BIN] = bin_idx
                            state[_STATE_COUNT_PRIOR] = greater_count
                            state[_STATE_BIN_COUNT] = bin_count
                    T.sync_threads()
                    prefix = prefix | (T.cast(state[_STATE_SELECTED_BIN], T.uint64) << shift)
                    prefix_mask = T.bitwise_not(T.cast(0, T.uint64)) << shift
                    selected_rank -= state[_STATE_COUNT_PRIOR]
                    # All keys in the threshold bin fit; lower bits cannot change the set.
                    if selected_rank == state[_STATE_BIN_COUNT]:
                        round_idx = 6
                    else:
                        round_idx += 1

                local_count = 0
                for i in T.Unroll(keys_per_thread):
                    local_count += T.cast(keys[i] != 0 and keys[i] >= prefix, T.int32)
                output_end = _block_prefix_sum(local_count, warp_totals, threads)

            # Prefix warps, then selected lanes in each chunk, to retain
            # candidate order without a contended atomic output counter.
            output_pos = T.alloc_var(T.int32)
            output_pos = T.shfl_sync(output_end, 31) - T.warp_reduce_sum(local_count)
            lane_mask = (T.cast(1, T.uint32) << (tidx % 32)) - T.cast(1, T.uint32)
            for i in T.Unroll(keys_per_thread):
                wins = keys[i] != 0 and keys[i] >= prefix
                selected_lanes = T.cast(T.ballot_sync(wins), T.uint32)
                if wins:
                    output_idx = output_pos + T.cast(T.popcount(selected_lanes & lane_mask), T.int32)
                    Out[row, output_idx] = T.reinterpret(
                        T.bitwise_xor(T.cast(keys[i], T.uint32), T.cast(0xFFFFFFFF, T.uint32)), T.int32)
                output_pos += T.cast(T.popcount(selected_lanes), T.int32)

    return sparse_dcp_global_topk_kernel_


def sparse_dcp_global_topk(gathered_candidates: torch.Tensor,
                           k: int,
                           fill: int = _FILL) -> torch.Tensor:
    """Select stable global top-k ids from packed DCP candidates.

    Input is rank-major ``[dcp_size, rows, K, 2]``, with each rank's valid
    candidates packed before its invalid tail. Output is ``[rows, K]`` INT32
    global ids in candidate order, padded with ``fill`` when fewer than K exist.

    Score ties prefer smaller global ids among these candidates. Local
    selection may already have discarded other tokens with the same score.
    """
    assert gathered_candidates.dim() == 4
    dcp_size, num_tokens, local_k, pair_width = gathered_candidates.shape
    assert local_k == k and pair_width == 2
    assert gathered_candidates.dtype == torch.float32
    assert dcp_size * k % _THREADS == 0, (
        'DCP candidate count must be divisible by the thread count')
    output = torch.empty((num_tokens, k),
                         dtype=torch.int32,
                         device=gathered_candidates.device)
    _sparse_dcp_global_topk_kernel(k, dcp_size, fill, _THREADS)(gathered_candidates, output)
    return output
