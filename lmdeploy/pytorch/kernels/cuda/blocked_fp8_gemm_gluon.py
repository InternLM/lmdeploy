# Copyright (c) OpenMMLab. All rights reserved.
"""Shape-specialized blocked-FP8 GEMM for NVIDIA Hopper GPUs."""

import functools

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from .utils import get_device_props

try:
    from gluon import aggregate
except ImportError:
    from triton.language.core import _aggregate as aggregate

# Triton 3.7 renamed the CTA-wide thread_barrier builtin to barrier.
_cta_barrier = getattr(gl, 'barrier', None) or gl.thread_barrier


# One WGMMA K tile covers exactly one quantization block, so every raw
# accumulator can be promoted before the next K tile is accumulated.
SCALE_BLOCK_K = 128
SCALE_BLOCK_N = 128
SMALL_M_THRESHOLD = 128
MID_M_THRESHOLD = 256

# The three schedules keep the same blocked-scaling math but optimize different
# bottlenecks. The single-partition schedule keeps loading and compute together
# to minimize overhead. Middle M otherwise separates a TMA producer from the
# compute warpgroup, while larger grids additionally persist CTAs and split
# each 256-row accumulator into waves.


@gluon.constexpr_function
def get_warps_per_cta(block_m, block_n, num_warps):
    warps_per_cta = [4, 1]
    instr_m = 16
    # Tile the atom until we have enough warps.
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        # Tile along M only if it would not cause broadcasting.
        if block_m > instr_m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta


@gluon.constexpr_function
def get_instr_shape_n(block_m, block_n, num_warps):
    instr_m = 16
    m_repetitions = triton.cdiv(block_m, instr_m)
    n_repetitions = triton.cdiv(num_warps, m_repetitions)
    max_instr_n = max(block_n // n_repetitions, 8)
    instr_n = 256
    while instr_n > max_instr_n or block_n % instr_n != 0:
        instr_n -= 8
    assert instr_n >= 8, 'expected to find a valid N instruction shape'
    return instr_n


@gluon.constexpr_function
def pick_wgmma_layout(dtype, block_m, block_n, num_warps):
    instr_m = 16
    instr_k = 256 // dtype.primitive_bitwidth
    instr_n = get_instr_shape_n(block_m, block_n, num_warps)
    warps_per_cta = get_warps_per_cta(block_m, block_n, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[instr_m, instr_n, instr_k],
    )


@gluon.jit
def _load_block_scales(
    a_scale_ptrs,
    b_scale_ptrs,
    a_scale_mask,
    block_valid,
    stride_ask: gl.constexpr,
    stride_bsk: gl.constexpr,
):
    """Fetch one scale slot, using identity past K, and advance the streams."""
    a_scale = gl.load(a_scale_ptrs, mask=a_scale_mask & block_valid, other=1.0)
    b_scale = gl.load(b_scale_ptrs, mask=block_valid, other=1.0)
    return a_scale, b_scale, a_scale_ptrs + stride_ask, b_scale_ptrs + stride_bsk


@gluon.jit
def _fp8_gemm_nt_single_partition_kernel(
    a_desc,
    a_scale_ptr,
    b_desc,
    b_scale_ptr,
    d_desc,
    M,
    K: gl.constexpr,
    stride_asm: gl.constexpr,
    stride_ask: gl.constexpr,
    stride_bsn: gl.constexpr,
    stride_bsk: gl.constexpr,
    transpose_output: gl.constexpr,
    scale_block_n: gl.constexpr,
    num_buffers: gl.constexpr,
    num_warps: gl.constexpr,
):
    # These are compile-time contract checks: they improve compiler failures
    # without adding instructions or latency to the generated kernel.
    gl.static_assert(a_desc.dtype == gl.float8e4nv, 'A must use FP8 E4M3')
    gl.static_assert(b_desc.dtype == gl.float8e4nv, 'B must use FP8 E4M3')
    gl.static_assert(d_desc.dtype == gl.bfloat16, 'output must use BF16')
    gl.static_assert(a_scale_ptr.dtype.element_ty == gl.float32, 'A scales must use FP32')
    gl.static_assert(b_scale_ptr.dtype.element_ty == gl.float32, 'B scales must use FP32')
    gl.static_assert(isinstance(a_desc.layout, gl.NVMMASharedLayout), 'A descriptor must use an NVMMA layout')
    gl.static_assert(isinstance(b_desc.layout, gl.NVMMASharedLayout), 'B descriptor must use an NVMMA layout')
    gl.static_assert(isinstance(d_desc.layout, gl.NVMMASharedLayout), 'output descriptor must use an NVMMA layout')

    block_m: gl.constexpr = a_desc.block_type.shape[0]
    block_n: gl.constexpr = b_desc.block_type.shape[0]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    gl.static_assert(b_desc.block_type.shape[1] == block_k, 'A and B tile K must match')
    gl.static_assert(K % block_k == 0, 'K must contain whole scale blocks')
    gl.static_assert(stride_asm == 1, 'A scales must be column-major')

    if transpose_output:
        # Compute C^T = B @ A^T so WGMMA's flexible N dimension represents
        # the tiny logical M dimension of the original output.
        off_m = gl.program_id(axis=1) * block_m
        off_n = gl.program_id(axis=0) * block_n
    else:
        off_m = gl.program_id(axis=0) * block_m
        off_n = gl.program_id(axis=1) * block_n

    gl.static_assert(num_buffers >= 2)
    a_smem = gl.allocate_shared_memory(
        a_desc.dtype,
        [num_buffers] + a_desc.block_type.shape,
        a_desc.layout,
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype,
        [num_buffers] + b_desc.block_type.shape,
        b_desc.layout,
    )
    gl.static_assert(isinstance(a_smem.type.layout, gl.NVMMASharedLayout))
    gl.static_assert(isinstance(b_smem.type.layout, gl.NVMMASharedLayout))

    acc_layout: gl.constexpr = pick_wgmma_layout(a_desc.dtype, block_m, block_n, num_warps)
    # Normally A scales index WGMMA M and broadcast across WGMMA N, so dim 1
    # is dropped. After transposition they index WGMMA N and broadcast across
    # WGMMA M, so dim 0 is dropped instead.
    if transpose_output:
        scale_layout: gl.constexpr = gl.SliceLayout(0, acc_layout)
        scale_offsets = off_n + gl.arange(0, block_n, layout=scale_layout)
        scale_mask = scale_offsets < M
        a_scale_ptrs = a_scale_ptr + scale_offsets * stride_asm
        b_scale_ptrs = b_scale_ptr + (off_m // scale_block_n) * stride_bsn
    else:
        scale_layout: gl.constexpr = gl.SliceLayout(1, acc_layout)
        scale_offsets = off_m + gl.arange(0, block_m, layout=scale_layout)
        scale_mask = scale_offsets < M
        a_scale_ptrs = a_scale_ptr + scale_offsets * stride_asm
        b_scale_ptrs = b_scale_ptr + (off_n // scale_block_n) * stride_bsn

    # This schedule has no producer worker: the same warpgroup issues TMA and
    # WGMMA in a fixed order. A stage is not wrapped until an intervening
    # warpgroup_mma_wait has retired its consumer, so ready barriers suffice;
    # the warp-specialized schedules below also need explicit empty barriers.
    ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for stage_idx in gl.static_range(0, num_buffers):
        mbarrier.init(ready_bars.index(stage_idx), count=1)

    # final_acc is the long-lived, scaled FP32 result. raw_acc contains only one
    # unscaled K-block partial so it can be promoted before entering final_acc.
    final_acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=acc_layout)
    raw_acc = warpgroup_mma_init(gl.zeros_like(final_acc))
    if transpose_output:
        a_scale = gl.zeros((block_n, ), dtype=gl.float32, layout=scale_layout) + 1.0
    else:
        a_scale = gl.zeros((block_m, ), dtype=gl.float32, layout=scale_layout) + 1.0
    b_scale = 1.0
    # Queue raw scale values one block beyond the scale paired with raw_acc.
    # Do not multiply here: that would immediately serialize on these LDGs.
    next_a_scale, next_b_scale, a_scale_ptrs, b_scale_ptrs = _load_block_scales(
        a_scale_ptrs,
        b_scale_ptrs,
        scale_mask,
        True,
        stride_ask,
        stride_bsk,
    )

    num_k_blocks: gl.constexpr = K // block_k
    num_preloads: gl.constexpr = min(num_buffers - 2, num_k_blocks)
    for k_block_idx in gl.static_range(0, num_preloads):
        stage_idx = k_block_idx % num_buffers
        ready_barrier = ready_bars.index(stage_idx)
        mbarrier.expect(ready_barrier, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(
            a_desc,
            [off_m, k_block_idx * block_k],
            ready_barrier,
            a_smem.index(stage_idx),
        )
        tma.async_copy_global_to_shared(
            b_desc,
            [off_n, k_block_idx * block_k],
            ready_barrier,
            b_smem.index(stage_idx),
        )

    preferred_unroll: gl.constexpr = 8 if transpose_output or block_n == 32 else 4
    num_steady: gl.constexpr = num_k_blocks - num_preloads
    num_prefix: gl.constexpr = num_steady % preferred_unroll

    # Prefix, steady state, and drain preserve the same one-block lag:
    # raw_acc/current scales describe the previous block, while next_*_scale
    # describes the block about to be issued. The final promotion closes the
    # lag after the last WGMMA.
    # Consume a short prefix first so the remaining runtime loop is cleanly
    # partially unrolled by the shape-specific factor.
    for producer_idx in gl.static_range(num_preloads, num_preloads + num_prefix):
        producer_stage = producer_idx % num_buffers
        producer_barrier = ready_bars.index(producer_stage)
        mbarrier.expect(producer_barrier, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(
            a_desc,
            [off_m, producer_idx * block_k],
            producer_barrier,
            a_smem.index(producer_stage),
        )
        tma.async_copy_global_to_shared(
            b_desc,
            [off_n, producer_idx * block_k],
            producer_barrier,
            b_smem.index(producer_stage),
        )

        consumer_idx = producer_idx - num_preloads
        consumer_stage = consumer_idx % num_buffers
        consumer_phase = (consumer_idx // num_buffers) & 1
        a_tile = a_smem.index(consumer_stage)
        b_tile = b_smem.index(consumer_stage).permute((1, 0))
        mbarrier.wait(ready_bars.index(consumer_stage), phase=consumer_phase)

        raw_acc = warpgroup_mma_wait(num_outstanding=0, deps=(raw_acc, ))
        if transpose_output:
            final_acc += raw_acc * (a_scale * b_scale)[None, :]
        else:
            final_acc += raw_acc * (a_scale * b_scale)[:, None]
        raw_acc = gl.zeros_like(final_acc)
        raw_acc = warpgroup_mma(a_tile, b_tile, raw_acc, is_async=True, use_acc=False)
        a_scale = next_a_scale
        b_scale = next_b_scale
        next_a_scale, next_b_scale, a_scale_ptrs, b_scale_ptrs = _load_block_scales(
            a_scale_ptrs,
            b_scale_ptrs,
            scale_mask,
            consumer_idx + 1 < num_k_blocks,
            stride_ask,
            stride_bsk,
        )

    for producer_group in range(num_preloads + num_prefix, num_k_blocks, preferred_unroll):
        for producer_offset in gl.static_range(0, preferred_unroll):
            group_producer_idx = producer_group + producer_offset
            group_producer_stage = group_producer_idx % num_buffers
            group_producer_barrier = ready_bars.index(group_producer_stage)
            mbarrier.expect(group_producer_barrier, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(
                a_desc,
                [off_m, group_producer_idx * block_k],
                group_producer_barrier,
                a_smem.index(group_producer_stage),
            )
            tma.async_copy_global_to_shared(
                b_desc,
                [off_n, group_producer_idx * block_k],
                group_producer_barrier,
                b_smem.index(group_producer_stage),
            )

            group_consumer_idx = group_producer_idx - num_preloads
            group_consumer_stage = group_consumer_idx % num_buffers
            group_consumer_phase = (group_consumer_idx // num_buffers) & 1
            group_a_tile = a_smem.index(group_consumer_stage)
            group_b_tile = b_smem.index(group_consumer_stage).permute((1, 0))
            mbarrier.wait(ready_bars.index(group_consumer_stage), phase=group_consumer_phase)

            raw_acc = warpgroup_mma_wait(num_outstanding=0, deps=(raw_acc, ))
            if transpose_output:
                final_acc += raw_acc * (a_scale * b_scale)[None, :]
            else:
                final_acc += raw_acc * (a_scale * b_scale)[:, None]
            raw_acc = gl.zeros_like(final_acc)
            raw_acc = warpgroup_mma(group_a_tile, group_b_tile, raw_acc, is_async=True, use_acc=False)
            a_scale = next_a_scale
            b_scale = next_b_scale
            next_a_scale, next_b_scale, a_scale_ptrs, b_scale_ptrs = _load_block_scales(
                a_scale_ptrs,
                b_scale_ptrs,
                scale_mask,
                group_consumer_idx + 1 < num_k_blocks,
                stride_ask,
                stride_bsk,
            )

    for drain_offset in gl.static_range(0, num_preloads):
        consumer_idx = num_steady + drain_offset
        consumer_stage = consumer_idx % num_buffers
        consumer_phase = (consumer_idx // num_buffers) & 1
        a_tile = a_smem.index(consumer_stage)
        b_tile = b_smem.index(consumer_stage).permute((1, 0))
        mbarrier.wait(ready_bars.index(consumer_stage), phase=consumer_phase)

        raw_acc = warpgroup_mma_wait(num_outstanding=0, deps=(raw_acc, ))
        if transpose_output:
            final_acc += raw_acc * (a_scale * b_scale)[None, :]
        else:
            final_acc += raw_acc * (a_scale * b_scale)[:, None]
        raw_acc = gl.zeros_like(final_acc)
        raw_acc = warpgroup_mma(a_tile, b_tile, raw_acc, is_async=True, use_acc=False)
        a_scale = next_a_scale
        b_scale = next_b_scale
        next_a_scale, next_b_scale, a_scale_ptrs, b_scale_ptrs = _load_block_scales(
            a_scale_ptrs,
            b_scale_ptrs,
            scale_mask,
            consumer_idx + 1 < num_k_blocks,
            stride_ask,
            stride_bsk,
        )

    raw_acc = warpgroup_mma_wait(num_outstanding=0, deps=(raw_acc, ))
    if transpose_output:
        final_acc += raw_acc * (a_scale * b_scale)[None, :]
    else:
        final_acc += raw_acc * (a_scale * b_scale)[:, None]

    for stage_idx in gl.static_range(0, num_buffers):
        mbarrier.invalidate(ready_bars.index(stage_idx))
    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
    if transpose_output:
        d_smem.store(final_acc.permute(1, 0).to(d_desc.dtype))
    else:
        d_smem.store(final_acc.to(d_desc.dtype))
    fence_async_shared()
    if transpose_output:
        tma.async_copy_shared_to_global(d_desc, [off_n, off_m], d_smem)
    else:
        tma.async_copy_shared_to_global(d_desc, [off_m, off_n], d_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def _load_mid_m_tile(
    descs,
    bars,
    buffers,
    offsets,
    block_k: gl.constexpr,
    num_buffers: gl.constexpr,
    num_k_blocks: gl.constexpr,
):
    a_desc, b_desc = descs
    ready_bars, empty_bars = bars
    a_smem, b_smem = buffers
    off_m, off_n = offsets

    # The low-register producer worker owns empty -> ready for every stage.
    # Splitting out the short prefix leaves the runtime loop two-way unrolled
    # without changing the stage/phase sequence.
    unroll: gl.constexpr = 2
    prefix: gl.constexpr = num_k_blocks % unroll
    for prefix_k_block_idx in gl.static_range(0, prefix):
        prefix_stage_idx = prefix_k_block_idx % num_buffers
        prefix_stage_phase = (prefix_k_block_idx // num_buffers) & 1
        ready_barrier = ready_bars.index(prefix_stage_idx)
        empty_barrier = empty_bars.index(prefix_stage_idx)

        mbarrier.wait(empty_barrier, phase=prefix_stage_phase ^ 1)
        mbarrier.expect(ready_barrier, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(
            a_desc,
            [off_m, prefix_k_block_idx * block_k],
            ready_barrier,
            a_smem.index(prefix_stage_idx),
        )
        tma.async_copy_global_to_shared(
            b_desc,
            [off_n, prefix_k_block_idx * block_k],
            ready_barrier,
            b_smem.index(prefix_stage_idx),
        )

    for k_block_group in range(prefix, num_k_blocks, unroll):
        for k_block_offset in gl.static_range(0, unroll):
            group_k_block_idx = k_block_group + k_block_offset
            group_stage_idx = group_k_block_idx % num_buffers
            group_stage_phase = (group_k_block_idx // num_buffers) & 1
            ready_barrier = ready_bars.index(group_stage_idx)
            empty_barrier = empty_bars.index(group_stage_idx)

            mbarrier.wait(empty_barrier, phase=group_stage_phase ^ 1)
            mbarrier.expect(ready_barrier, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(
                a_desc,
                [off_m, group_k_block_idx * block_k],
                ready_barrier,
                a_smem.index(group_stage_idx),
            )
            tma.async_copy_global_to_shared(
                b_desc,
                [off_n, group_k_block_idx * block_k],
                ready_barrier,
                b_smem.index(group_stage_idx),
            )


@gluon.jit
def _compute_mid_m_tile(
    bars,
    buffers,
    scale_ptrs,
    acc_layout,
    offsets,
    M,
    stride_asm: gl.constexpr,
    stride_ask: gl.constexpr,
    stride_bsn: gl.constexpr,
    stride_bsk: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    scale_block_n: gl.constexpr,
    num_buffers: gl.constexpr,
    num_k_blocks: gl.constexpr,
):
    ready_bars, empty_bars = bars
    a_smem, b_smem = buffers
    a_scale_ptr, b_scale_ptr = scale_ptrs
    off_m, off_n = offsets

    # Drop accumulator N: one A scale is distributed per output row and then
    # broadcast across all BLOCK_N accumulator columns.
    scale_layout: gl.constexpr = gl.SliceLayout(1, acc_layout)
    row_offsets = off_m + gl.arange(0, block_m, layout=scale_layout)
    a_scale_ptrs = a_scale_ptr + row_offsets * stride_asm
    b_scale_ptrs = b_scale_ptr + (off_n // scale_block_n) * stride_bsn

    final_acc = gl.zeros((block_m, block_n), dtype=gl.float32, layout=acc_layout)
    raw_acc = gl.zeros_like(final_acc)

    # Keep the next block's raw scale values live without consuming them yet.
    # Their loads can overlap two WGMMA intervals; multiplying immediately
    # would instead stall this warpgroup on the scale loads at the load site.
    row_mask = row_offsets < M
    a_scale, b_scale, a_scale_ptrs, b_scale_ptrs = _load_block_scales(
        a_scale_ptrs,
        b_scale_ptrs,
        row_mask,
        True,
        stride_ask,
        stride_bsk,
    )

    next_a_scale = gl.zeros((block_m, ), dtype=gl.float32, layout=scale_layout) + 1.0
    next_b_scale = 1.0
    if num_k_blocks > 1:
        next_a_scale, next_b_scale, a_scale_ptrs, b_scale_ptrs = _load_block_scales(
            a_scale_ptrs,
            b_scale_ptrs,
            row_mask,
            True,
            stride_ask,
            stride_bsk,
        )

    # Prime K block zero without waiting on or promoting a sentinel.
    prime_stage: gl.constexpr = 0
    mbarrier.wait(ready_bars.index(prime_stage), phase=0)
    a_tile = a_smem.index(prime_stage)
    b_tile = b_smem.index(prime_stage).permute((1, 0))
    raw_acc = warpgroup_mma(a_tile, b_tile, raw_acc, is_async=True, use_acc=False)

    # After priming block zero, each iteration waits for the next ready stage,
    # retires the previous WGMMA, releases that previous stage, promotes its
    # partial with register-resident scales, and issues the next WGMMA. The
    # WGMMA wait—not the following proxy fence—is what makes release safe.
    unroll: gl.constexpr = 2
    prefix: gl.constexpr = (num_k_blocks - 1) % unroll
    for prefix_k_block_idx in gl.static_range(1, 1 + prefix):
        prefix_stage_idx = prefix_k_block_idx % num_buffers
        prefix_stage_phase = (prefix_k_block_idx // num_buffers) & 1
        mbarrier.wait(ready_bars.index(prefix_stage_idx), phase=prefix_stage_phase)

        prefix_a_tile = a_smem.index(prefix_stage_idx)
        prefix_b_tile = b_smem.index(prefix_stage_idx).permute((1, 0))

        raw_acc = warpgroup_mma_wait(num_outstanding=0, deps=(raw_acc, ))
        previous_stage_idx = (prefix_k_block_idx - 1) % num_buffers
        fence_async_shared()
        mbarrier.arrive(empty_bars.index(previous_stage_idx), count=1)
        final_acc += raw_acc * (a_scale * b_scale)[:, None]

        raw_acc = gl.zeros_like(final_acc)
        raw_acc = warpgroup_mma(prefix_a_tile, prefix_b_tile, raw_acc, is_async=True, use_acc=False)

        a_scale = next_a_scale
        b_scale = next_b_scale
        prefix_has_future_scale: gl.constexpr = prefix_k_block_idx + 1 < num_k_blocks
        next_a_scale, next_b_scale, a_scale_ptrs, b_scale_ptrs = _load_block_scales(
            a_scale_ptrs,
            b_scale_ptrs,
            row_mask,
            prefix_has_future_scale,
            stride_ask,
            stride_bsk,
        )

    for k_block_group in range(1 + prefix, num_k_blocks, unroll):
        for k_block_offset in gl.static_range(0, unroll):
            group_k_block_idx = k_block_group + k_block_offset
            group_stage_idx = group_k_block_idx % num_buffers
            group_stage_phase = (group_k_block_idx // num_buffers) & 1
            mbarrier.wait(ready_bars.index(group_stage_idx), phase=group_stage_phase)

            group_a_tile = a_smem.index(group_stage_idx)
            group_b_tile = b_smem.index(group_stage_idx).permute((1, 0))

            raw_acc = warpgroup_mma_wait(num_outstanding=0, deps=(raw_acc, ))
            group_previous_stage_idx = (group_k_block_idx - 1) % num_buffers
            fence_async_shared()
            mbarrier.arrive(empty_bars.index(group_previous_stage_idx), count=1)
            final_acc += raw_acc * (a_scale * b_scale)[:, None]

            raw_acc = gl.zeros_like(final_acc)
            raw_acc = warpgroup_mma(group_a_tile, group_b_tile, raw_acc, is_async=True, use_acc=False)
            a_scale = next_a_scale
            b_scale = next_b_scale
            group_has_future_scale = group_k_block_idx + 1 < num_k_blocks
            next_a_scale, next_b_scale, a_scale_ptrs, b_scale_ptrs = _load_block_scales(
                a_scale_ptrs,
                b_scale_ptrs,
                row_mask,
                group_has_future_scale,
                stride_ask,
                stride_bsk,
            )

    raw_acc = warpgroup_mma_wait(num_outstanding=0, deps=(raw_acc, ))
    fence_async_shared()
    mbarrier.arrive(empty_bars.index((num_k_blocks - 1) % num_buffers), count=1)
    final_acc += raw_acc * (a_scale * b_scale)[:, None]
    return (final_acc, )


@gluon.jit
def _fp8_gemm_nt_warp_specialized_kernel(
    a_desc,
    a_scale_ptr,
    b_desc,
    b_scale_ptr,
    d_desc,
    M,
    K: gl.constexpr,
    stride_asm: gl.constexpr,
    stride_ask: gl.constexpr,
    stride_bsn: gl.constexpr,
    stride_bsk: gl.constexpr,
    scale_block_n: gl.constexpr,
    num_buffers: gl.constexpr,
    num_warps: gl.constexpr,
):
    # Compile-time only: descriptor/pointer types and layouts are specialization
    # properties, so these assertions do not execute on the GPU hot path.
    gl.static_assert(a_desc.dtype == gl.float8e4nv, 'A must use FP8 E4M3')
    gl.static_assert(b_desc.dtype == gl.float8e4nv, 'B must use FP8 E4M3')
    gl.static_assert(d_desc.dtype == gl.bfloat16, 'output must use BF16')
    gl.static_assert(a_scale_ptr.dtype.element_ty == gl.float32, 'A scales must use FP32')
    gl.static_assert(b_scale_ptr.dtype.element_ty == gl.float32, 'B scales must use FP32')
    gl.static_assert(isinstance(a_desc.layout, gl.NVMMASharedLayout), 'A descriptor must use an NVMMA layout')
    gl.static_assert(isinstance(b_desc.layout, gl.NVMMASharedLayout), 'B descriptor must use an NVMMA layout')
    gl.static_assert(isinstance(d_desc.layout, gl.NVMMASharedLayout), 'output descriptor must use an NVMMA layout')

    block_m: gl.constexpr = d_desc.block_type.shape[0]
    block_n: gl.constexpr = d_desc.block_type.shape[1]
    block_k: gl.constexpr = a_desc.block_type.shape[1]
    gl.static_assert(b_desc.block_type.shape[1] == block_k, 'A and B tile K must match')
    gl.static_assert(K % block_k == 0, 'K must contain whole scale blocks')
    gl.static_assert(stride_asm == 1, 'A scales must be column-major')
    num_k_blocks: gl.constexpr = K // block_k

    off_m = gl.program_id(axis=0) * block_m
    off_n = gl.program_id(axis=1) * block_n

    a_smem = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    gl.static_assert(isinstance(a_smem.type.layout, gl.NVMMASharedLayout))
    gl.static_assert(isinstance(b_smem.type.layout, gl.NVMMASharedLayout))

    acc_layout: gl.constexpr = pick_wgmma_layout(a_desc.dtype, block_m, block_n, num_warps)

    ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for stage_idx in gl.static_range(0, num_buffers):
        mbarrier.init(ready_bars.index(stage_idx), count=1)
        mbarrier.init(empty_bars.index(stage_idx), count=1)

    bars = (ready_bars, empty_bars)
    buffers = (a_smem, b_smem)
    offsets = (off_m, off_n)
    # The default partition owns WGMMA, scale promotion, and the result. One
    # additional worker warp owns the TMA producer side of the stage ring.
    final_acc, = gl.warp_specialize(
        [
            (
                _compute_mid_m_tile,
                (
                    bars,
                    buffers,
                    (a_scale_ptr, b_scale_ptr),
                    acc_layout,
                    offsets,
                    M,
                    stride_asm,
                    stride_ask,
                    stride_bsn,
                    stride_bsk,
                    block_m,
                    block_n,
                    scale_block_n,
                    num_buffers,
                    num_k_blocks,
                ),
            ),
            (
                _load_mid_m_tile,
                (
                    (a_desc, b_desc),
                    bars,
                    buffers,
                    offsets,
                    block_k,
                    num_buffers,
                    num_k_blocks,
                ),
            ),
        ],
        worker_num_warps=[1],
        worker_num_regs=[40],
    )

    for stage_idx in gl.static_range(0, num_buffers):
        mbarrier.invalidate(ready_bars.index(stage_idx))
        mbarrier.invalidate(empty_bars.index(stage_idx))
    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
    d_smem.store(final_acc.to(d_desc.dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(d_desc, [off_m, off_n], d_smem)
    tma.store_wait(pendings=0)


@aggregate
class PersistentTileScheduler:
    first_tile: gl.tensor
    total_tiles: gl.tensor
    num_tiles_m: gl.tensor
    num_tiles_n: gl.tensor
    num_programs: gl.tensor

    @gluon.constexpr_function
    def __init__(self, first_tile, total_tiles, num_tiles_m, num_tiles_n, num_programs):
        self.first_tile = first_tile
        self.total_tiles = total_tiles
        self.num_tiles_m = num_tiles_m
        self.num_tiles_n = num_tiles_n
        self.num_programs = num_programs

    @gluon.jit
    def initialize(M, N: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
        first_tile = gl.program_id(axis=0)
        num_programs = gl.num_programs(axis=0)
        num_tiles_m = gl.cdiv(M, BLOCK_M)
        num_tiles_n = gl.cdiv(N, BLOCK_N)
        total_tiles = num_tiles_m * num_tiles_n
        return PersistentTileScheduler(first_tile, total_tiles, num_tiles_m, num_tiles_n, num_programs)

    @gluon.jit
    def get_num_tiles(self):
        return gl.cdiv(self.total_tiles - self.first_tile, self.num_programs)

    @gluon.jit
    def get_tile(self, idx):
        # Each persistent program advances by the resident grid size. Grouping
        # nearby M tiles under the same N range improves B-tile L2 locality.
        tile_idx = self.first_tile + idx * self.num_programs

        group_size_m = 16
        tiles_per_group = self.num_tiles_n * group_size_m
        group_idx = tile_idx // tiles_per_group
        first_tile_m = group_idx * group_size_m
        tile_idx_in_group = tile_idx % tiles_per_group
        tiles_m_in_group = min(group_size_m, self.num_tiles_m - first_tile_m)

        tile_m = first_tile_m + tile_idx_in_group % tiles_m_in_group
        tile_n = tile_idx_in_group // tiles_m_in_group
        return tile_m, tile_n


@gluon.jit
def _load_persistent_tiles(
    descs,
    bars,
    buffers,
    M, N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_K_BLOCKS: gl.constexpr,
):
    a_desc, b_desc, a_scale_desc = descs
    ready_bars, empty_bars = bars
    a_smem, b_smem, a_scale_smem = buffers

    scheduler = PersistentTileScheduler.initialize(M, N, BLOCK_M, BLOCK_N)

    # This worker owns empty -> ready across every persistent output tile. The
    # flattened iteration keeps stage phase continuous at tile boundaries;
    # resetting phase for each tile would eventually consume stale stages.
    for tile_iter in range(scheduler.get_num_tiles()):
        tile_m, tile_n = scheduler.get_tile(tile_iter)
        off_m = tile_m * BLOCK_M
        off_n = tile_n * BLOCK_N

        for k_block_idx in range(0, NUM_K_BLOCKS):
            off_k = k_block_idx * BLOCK_K
            pipeline_iter = tile_iter * NUM_K_BLOCKS + k_block_idx
            stage_idx = pipeline_iter % NUM_BUFFERS
            stage_phase = (pipeline_iter // NUM_BUFFERS) & 1

            a_stage = a_smem.index(stage_idx)
            b_stage = b_smem.index(stage_idx)
            a_scale_stage = a_scale_smem.index(stage_idx)
            ready_barrier = ready_bars.index(stage_idx)
            empty_barrier = empty_bars.index(stage_idx)

            mbarrier.wait(empty_barrier, phase=stage_phase ^ 1)

            mbarrier.expect(
                ready_barrier,
                a_desc.block_type.nbytes + b_desc.block_type.nbytes + a_scale_desc.block_type.nbytes,
            )
            tma.async_copy_global_to_shared(a_desc, [off_m, off_k], ready_barrier, a_stage)
            tma.async_copy_global_to_shared(b_desc, [off_n, off_k], ready_barrier, b_stage)
            tma.async_copy_global_to_shared(a_scale_desc, [k_block_idx, off_m], ready_barrier, a_scale_stage)


@gluon.jit
def _compute_persistent_tiles(
    dtype,
    d_desc,
    bars,
    buffers,
    b_scale_ptr,
    M,
    N: gl.constexpr,
    stride_bsn: gl.constexpr,
    stride_bsk: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SCALE_BLOCK_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_K_BLOCKS: gl.constexpr,
    num_warps: gl.constexpr,
):
    ready_bars, empty_bars = bars
    a_smem, b_smem, a_scale_smem = buffers

    # BLOCK_M=256 is computed as two 128-row waves to keep the temporary
    # WGMMA accumulator small enough to avoid register spills.
    WAVE_M: gl.constexpr = min(BLOCK_M, 128)
    NUM_M_WAVES: gl.constexpr = BLOCK_M // WAVE_M
    wave_acc_layout: gl.constexpr = pick_wgmma_layout(dtype, WAVE_M, BLOCK_N, num_warps)
    # Each M wave has one A scale per accumulator row, broadcast across N.
    scale_layout: gl.constexpr = gl.SliceLayout(1, wave_acc_layout)

    scheduler = PersistentTileScheduler.initialize(M, N, BLOCK_M, BLOCK_N)
    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)

    NUM_COMPUTE_WARPS: gl.constexpr = gl.num_warps()
    NUM_B_SCALE_SLOTS: gl.constexpr = triton.next_power_of_2(max(NUM_K_BLOCKS, NUM_COMPUTE_WARPS * 32))

    b_scale_load_layout: gl.constexpr = gl.BlockedLayout(
        [1],
        [32],
        [NUM_COMPUTE_WARPS],
        [0],
    )
    b_scale_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0],
    )
    b_scale_smem = gl.allocate_shared_memory(gl.float32, (NUM_B_SCALE_SLOTS, 1), b_scale_smem_layout)
    b_scale_smem_flat = b_scale_smem._reinterpret(gl.float32, (NUM_B_SCALE_SLOTS, ), b_scale_smem_layout)

    for tile_iter in range(scheduler.get_num_tiles()):
        tile_m, tile_n = scheduler.get_tile(tile_iter)
        off_m = tile_m * BLOCK_M
        off_n = tile_n * BLOCK_N

        # Cooperatively cache all B scales for this output-column tile. Every
        # M wave reuses this cache throughout the K loop.
        n_scale_group = off_n // SCALE_BLOCK_N
        b_scale_tile_ptr = b_scale_ptr + n_scale_group * stride_bsn
        _cta_barrier()
        b_scale_offs = gl.arange(0, NUM_B_SCALE_SLOTS, layout=b_scale_load_layout)
        b_scale_mask = b_scale_offs < NUM_K_BLOCKS
        b_scales = gl.load(b_scale_tile_ptr + b_scale_offs * stride_bsk, mask=b_scale_mask, other=0.0)
        b_scale_smem_flat.store(b_scales)
        _cta_barrier()

        wave_0_acc = gl.zeros(
            (WAVE_M, BLOCK_N),
            gl.float32,
            wave_acc_layout,
        )
        if NUM_M_WAVES == 2:
            wave_1_acc = gl.zeros_like(wave_0_acc)

        # The consumer reconstructs the same flattened stage/phase sequence as
        # the producer. A and its scales arrive together through TMA; B scales
        # come from the per-output-tile shared cache above.
        for k_block_idx in range(0, NUM_K_BLOCKS):
            pipeline_iter = tile_iter * NUM_K_BLOCKS + k_block_idx
            stage_idx = pipeline_iter % NUM_BUFFERS
            stage_phase = (pipeline_iter // NUM_BUFFERS) & 1
            ready_barrier = ready_bars.index(stage_idx)
            mbarrier.wait(ready_barrier, phase=stage_phase)

            a_stage = a_smem.index(stage_idx)
            b_stage = b_smem.index(stage_idx).permute((1, 0))
            a_scale_stage = a_scale_smem.index(stage_idx)
            b_scale = b_scale_smem.index(k_block_idx).load(scale_layout)

            a_scale_wave_0 = a_scale_stage.slice(0, WAVE_M, dim=1).load(gl.AutoLayout())
            a_scale_wave_0 = a_scale_wave_0.reshape((WAVE_M,))
            a_scale_wave_0 = gl.set_auto_layout(a_scale_wave_0, scale_layout)
            if NUM_M_WAVES == 2:
                a_scale_wave_1 = a_scale_stage.slice(WAVE_M, WAVE_M, dim=1).load(gl.AutoLayout())
                a_scale_wave_1 = a_scale_wave_1.reshape((WAVE_M,))
                a_scale_wave_1 = gl.set_auto_layout(a_scale_wave_1, scale_layout)

            a_wave_0 = a_stage.slice(0, WAVE_M, dim=0)
            if NUM_M_WAVES == 2:
                a_wave_1 = a_stage.slice(WAVE_M, WAVE_M, dim=0)

            partial_acc = gl.zeros_like(wave_0_acc)
            partial_acc = warpgroup_mma(a_wave_0, b_stage, partial_acc, is_async=True, use_acc=False)
            partial_acc = warpgroup_mma_wait(num_outstanding=0, deps=(partial_acc, ))
            block_scale = a_scale_wave_0 * b_scale
            wave_0_acc += partial_acc * block_scale[:, None]

            if NUM_M_WAVES == 2:
                partial_acc = warpgroup_mma(a_wave_1, b_stage, partial_acc, is_async=True, use_acc=False)
                partial_acc = warpgroup_mma_wait(num_outstanding=0, deps=(partial_acc, ))
                block_scale = a_scale_wave_1 * b_scale
                wave_1_acc += partial_acc * block_scale[:, None]

            # Both M waves have finished reading this stage. Publish that it is
            # safe for the loader warp to overwrite on the next phase.
            fence_async_shared()
            empty_barrier = empty_bars.index(stage_idx)
            mbarrier.arrive(empty_barrier, count=1)

        # The output shared-memory buffer is reused across persistent tiles.
        tma.store_wait(pendings=0)

        d_smem.slice(0, WAVE_M, dim=0).store(wave_0_acc.to(d_desc.dtype))
        if NUM_M_WAVES == 2:
            d_smem.slice(WAVE_M, WAVE_M, dim=0).store(wave_1_acc.to(d_desc.dtype))
        fence_async_shared()

        tma.async_copy_shared_to_global(d_desc, [off_m, off_n], d_smem)

    tma.store_wait(pendings=0)


@gluon.jit
def _fp8_gemm_nt_persistent_kernel(
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_ptr,
    d_desc,
    M,
    N: gl.constexpr,
    K: gl.constexpr,
    stride_bsn: gl.constexpr,
    stride_bsk: gl.constexpr,
    SCALE_BLOCK_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    num_warps: gl.constexpr,
):
    # Compile-time only: the persistent path additionally requires its staged
    # A-scale descriptor to carry FP32 elements in an NVMMA shared layout.
    gl.static_assert(a_desc.dtype == gl.float8e4nv, 'A must use FP8 E4M3')
    gl.static_assert(b_desc.dtype == gl.float8e4nv, 'B must use FP8 E4M3')
    gl.static_assert(d_desc.dtype == gl.bfloat16, 'output must use BF16')
    gl.static_assert(a_scale_desc.dtype == gl.float32, 'A scales must use FP32')
    gl.static_assert(b_scale_ptr.dtype.element_ty == gl.float32, 'B scales must use FP32')
    gl.static_assert(isinstance(a_desc.layout, gl.NVMMASharedLayout), 'A descriptor must use an NVMMA layout')
    gl.static_assert(isinstance(b_desc.layout, gl.NVMMASharedLayout), 'B descriptor must use an NVMMA layout')
    gl.static_assert(
        isinstance(a_scale_desc.layout, gl.NVMMASharedLayout),
        'A-scale descriptor must use an NVMMA layout',
    )
    gl.static_assert(isinstance(d_desc.layout, gl.NVMMASharedLayout), 'output descriptor must use an NVMMA layout')

    BLOCK_M: gl.constexpr = d_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = d_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    gl.static_assert(b_desc.block_type.shape[1] == BLOCK_K, 'A and B tile K must match')
    gl.static_assert(K % BLOCK_K == 0, 'K must contain whole scale blocks')
    gl.static_assert(a_scale_desc.block_type.shape == [1, BLOCK_M], 'A-scale TMA tile must cover one M tile')
    NUM_K_BLOCKS: gl.constexpr = K // BLOCK_K

    a_smem = gl.allocate_shared_memory(
        a_desc.dtype,
        [NUM_BUFFERS] + a_desc.block_type.shape,
        a_desc.layout,
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype,
        [NUM_BUFFERS] + b_desc.block_type.shape,
        b_desc.layout,
    )
    a_scale_smem = gl.allocate_shared_memory(
        a_scale_desc.dtype,
        [NUM_BUFFERS] + a_scale_desc.block_type.shape,
        a_scale_desc.layout,
    )
    gl.static_assert(isinstance(a_smem.type.layout, gl.NVMMASharedLayout))
    gl.static_assert(isinstance(b_smem.type.layout, gl.NVMMASharedLayout))

    dtype = a_desc.dtype

    ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
    empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout())
    for stage_idx in gl.static_range(0, NUM_BUFFERS):
        ready_barrier = ready_bars.index(stage_idx)
        empty_barrier = empty_bars.index(stage_idx)
        mbarrier.init(ready_barrier, count=1)
        mbarrier.init(empty_barrier, count=1)

    descs = (a_desc, b_desc, a_scale_desc)
    bars = (ready_bars, empty_bars)
    buffers = (a_smem, b_smem, a_scale_smem)

    # The default partition uses the launcher's compute warps. One additional
    # low-register worker warp owns the TMA producer pipeline.
    gl.warp_specialize(
        [
            (
                _compute_persistent_tiles,
                (
                    dtype,
                    d_desc,
                    bars,
                    buffers,
                    b_scale_ptr,
                    M,
                    N,
                    stride_bsn,
                    stride_bsk,
                    BLOCK_M,
                    BLOCK_N,
                    SCALE_BLOCK_N,
                    NUM_BUFFERS,
                    NUM_K_BLOCKS,
                    num_warps,
                ),
            ),
            (
                _load_persistent_tiles,
                (
                    descs,
                    bars,
                    buffers,
                    M,
                    N,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    NUM_BUFFERS,
                    NUM_K_BLOCKS,
                ),
            ),
        ],
        worker_num_warps=[1],
        worker_num_regs=[40],
    )

    for stage_idx in gl.static_range(0, NUM_BUFFERS):
        ready_barrier = ready_bars.index(stage_idx)
        empty_barrier = empty_bars.index(stage_idx)
        mbarrier.invalidate(ready_barrier)
        mbarrier.invalidate(empty_barrier)


def _make_matrix_descriptors(a_quant, b_quant, output, block_m, block_n):
    """Build the A/B/output descriptors shared by all schedule families."""
    a_block_shape = [block_m, SCALE_BLOCK_K]
    b_block_shape = [block_n, SCALE_BLOCK_K]
    d_block_shape = [block_m, block_n]
    a_layout = gl.NVMMASharedLayout.get_default_for(a_block_shape, gl.float8e4nv)
    b_layout = gl.NVMMASharedLayout.get_default_for(b_block_shape, gl.float8e4nv)
    d_layout = gl.NVMMASharedLayout.get_default_for(d_block_shape, gl.bfloat16)

    a_desc = TensorDescriptor.from_tensor(a_quant, a_block_shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(b_quant, b_block_shape, b_layout)
    d_desc = TensorDescriptor.from_tensor(output, d_block_shape, d_layout)
    return a_desc, b_desc, d_desc


def _make_transposed_small_descriptors(a_quant, b_quant, output, block_m, block_n):
    """Build descriptors for C^T = B @ A^T while storing C in-place."""
    lhs_block_shape = [block_m, SCALE_BLOCK_K]
    rhs_block_shape = [block_n, SCALE_BLOCK_K]
    d_block_shape = [block_n, block_m]
    lhs_layout = gl.NVMMASharedLayout.get_default_for(lhs_block_shape, gl.float8e4nv)
    rhs_layout = gl.NVMMASharedLayout.get_default_for(rhs_block_shape, gl.float8e4nv)
    d_layout = gl.NVMMASharedLayout.get_default_for(d_block_shape, gl.bfloat16)

    lhs_desc = TensorDescriptor.from_tensor(b_quant, lhs_block_shape, lhs_layout)
    rhs_desc = TensorDescriptor.from_tensor(a_quant, rhs_block_shape, rhs_layout)
    d_desc = TensorDescriptor.from_tensor(output, d_block_shape, d_layout)
    return lhs_desc, rhs_desc, d_desc


def _get_transpose_m_limit(k):
    """Choose where avoiding padded WGMMA rows outweighs reloading B."""
    if k <= 5120:
        return 32
    if k < 8192:
        return 24
    if k < 16384:
        return 16
    return 8


def _launch_single_partition(a_quant, a_scale, b_quant, b_scale, output):
    m, n = output.shape
    k = a_quant.size(-1)
    transpose_output = m <= _get_transpose_m_limit(k) and k >= 4096
    block_m = 64
    # Transposing the GEMM makes WGMMA's flexible N dimension represent the
    # small logical M dimension. Use BN8 for slices and BN16 at exactly M=32;
    # the K-dependent limit balances avoided padded promotion against reloading
    # each B tile for another M slice. Short K does not amortize the smaller
    # 64-CTA grid used by M <= 8.
    if transpose_output:
        block_n = 16 if m == 32 else 8
        num_buffers = 16 if k >= 8192 else 8
    elif m <= 64:
        block_n = 32
        num_buffers = 8
    else:
        block_n = 64
        num_buffers = 6
    num_warps = 4

    if transpose_output:
        a_desc, b_desc, d_desc = _make_transposed_small_descriptors(
            a_quant,
            b_quant,
            output,
            block_m,
            block_n,
        )
    else:
        a_desc, b_desc, d_desc = _make_matrix_descriptors(a_quant, b_quant, output, block_m, block_n)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    if transpose_output:
        grid = (triton.cdiv(m, block_n), triton.cdiv(n, block_m))
    _fp8_gemm_nt_single_partition_kernel[grid](
        a_desc,
        a_scale,
        b_desc,
        b_scale,
        d_desc,
        m,
        k,
        *a_scale.stride(),
        *b_scale.stride(),
        transpose_output=transpose_output,
        scale_block_n=SCALE_BLOCK_N,
        num_buffers=num_buffers,
        num_warps=num_warps,
    )


def _launch_warp_specialized(a_quant, a_scale, b_quant, b_scale, output):
    m, n = output.shape
    k = a_quant.size(-1)
    block_m = 64
    block_n = 128
    num_buffers = 4
    num_warps = 4

    a_desc, b_desc, d_desc = _make_matrix_descriptors(a_quant, b_quant, output, block_m, block_n)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fp8_gemm_nt_warp_specialized_kernel[grid](
        a_desc,
        a_scale,
        b_desc,
        b_scale,
        d_desc,
        m,
        k,
        *a_scale.stride(),
        *b_scale.stride(),
        scale_block_n=SCALE_BLOCK_N,
        num_buffers=num_buffers,
        num_warps=num_warps,
        maxnreg=128,
    )


def _launch_persistent(a_quant, a_scale, b_quant, b_scale, output):
    m, n = output.shape
    k = a_quant.size(-1)
    block_m = 256
    block_n = 128
    num_buffers = 3
    num_warps = 8

    a_desc, b_desc, d_desc = _make_matrix_descriptors(a_quant, b_quant, output, block_m, block_n)

    # A scales are physically contiguous along M. The transposed descriptor
    # exposes [K blocks, M], letting one TMA transaction stage all row scales
    # needed by a 256-row output tile and one K block.
    a_scale_transposed = a_scale.T
    a_scale_block_shape = [1, block_m]
    a_scale_layout = gl.NVMMASharedLayout.get_default_for(a_scale_block_shape, gl.float32)
    a_scale_desc = TensorDescriptor.from_tensor(a_scale_transposed, a_scale_block_shape, a_scale_layout)

    num_tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    num_sms = get_device_props(a_quant.device.index)['multi_processor_count']
    # Cap the persistent grid at the SM count; each program obtains further
    # logical output tiles from PersistentTileScheduler.
    grid = (min(num_sms, num_tiles), )
    _fp8_gemm_nt_persistent_kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale,
        d_desc,
        m,
        n,
        k,
        *b_scale.stride(),
        SCALE_BLOCK_N=SCALE_BLOCK_N,
        NUM_BUFFERS=num_buffers,
        num_warps=num_warps,
        maxnreg=168,
    )


def _prefer_single_partition(m, n, num_sms):
    """Select the single-partition schedule for small or compact grids."""
    if m <= SMALL_M_THRESHOLD:
        return True
    if m > MID_M_THRESHOLD:
        return False

    num_tiles_m = triton.cdiv(m, 64)
    single_partition_tiles = num_tiles_m * triton.cdiv(n, 64)
    if single_partition_tiles <= num_sms:
        return True

    # BN64 removes the producer/consumer barrier but doubles the grid relative
    # to the warp-specialized BN128 schedule. It remains profitable for up to
    # two dense CTA waves. Sparse final M tiles lose that tradeoff, so require
    # at least 48 of their 64 rows to be useful in the second case.
    warp_specialized_tiles = num_tiles_m * triton.cdiv(n, 128)
    final_tile_rows = m - (num_tiles_m - 1) * 64
    return warp_specialized_tiles <= num_sms and final_tile_rows >= 48


@functools.lru_cache
def _use_single_partition(m, n, device_index):
    num_sms = get_device_props(device_index)['multi_processor_count']
    return _prefer_single_partition(m, n, num_sms)


@functools.lru_cache
def _use_warp_specialized(m, n, device_index):
    """Use warp specialization while its grid fits one resident CTA wave."""
    if m <= MID_M_THRESHOLD:
        return True

    # The warp-specialized kernel can keep two CTAs resident per Hopper SM,
    # whereas the register-heavy persistent kernel is limited to one. Prefer
    # warp specialization when all 64x128 tiles fit in that resident capacity;
    # this avoids an underfilled persistent grid, notably at M=512, N=4096.
    warp_specialized_tiles = triton.cdiv(m, 64) * triton.cdiv(n, 128)
    num_sms = get_device_props(device_index)['multi_processor_count']
    return warp_specialized_tiles <= 2 * num_sms


def fp8_gemm_nt(a, b, d, c):
    """Compute a blocked-scaled FP8 GEMM with an NT operand convention.

    ``a`` contains contiguous FP8 values shaped ``[M, K]`` and per-row scales
    shaped ``[M, K / 128]``. The A scales must have stride 1 along M; their
    physical K-block stride must be 16-byte aligned for the persistent path.
    ``b`` contains contiguous FP8 values shaped ``[N, K]`` and contiguous
    scales shaped ``[ceil(N / 128), K / 128]``. Both K and N scale groups are
    128 elements. The result is written to contiguous BF16 ``d``; ``c`` is
    accepted for DeepGEMM-style signature compatibility and is not used.

    M up to 128 uses a single-partition multistage schedule. Within that body,
    small M with K at least 4096 computes the equivalent transposed GEMM to
    avoid padding the tensor-core result to 64 rows. Its tuned M limit shrinks
    from 32 to 8 as K grows because each additional M slice reloads B. For M up
    to 256, dense 64x64 grids that fit at most two CTA waves reuse the
    single-partition body to avoid warp-specialization barriers. Other middle
    shapes use a primed warp-specialized 64x128 schedule. Larger grids use a
    persistent two-wave schedule that reuses B data and scales while
    controlling accumulator register pressure.
    """
    a_quant, a_scale = a
    b_quant, b_scale = b

    if _use_single_partition(d.size(0), d.size(1), d.device.index):
        _launch_single_partition(a_quant, a_scale, b_quant, b_scale, d)
    elif _use_warp_specialized(d.size(0), d.size(1), d.device.index):
        _launch_warp_specialized(a_quant, a_scale, b_quant, b_scale, d)
    else:
        _launch_persistent(a_quant, a_scale, b_quant, b_scale, d)
