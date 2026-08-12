# Copyright (c) OpenMMLab. All rights reserved.
"""Gluon GEMM for small routed blocked-FP8 MoE tiles on Hopper."""

import torch
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

from .blocked_fp8_gemm_gluon import pick_wgmma_layout

SCALE_BLOCK_K = 128
SCALE_BLOCK_N = 128
SMALL_M_BLOCK = 8
MULTISTAGE_MIN_K_BLOCKS = 8
# At small M, CTA-level concurrency cannot hide the long-K B-TMA and gathered-A
# latency. H200 sweeps at E=256 and E=384 select these boundaries; above them,
# extra shared-memory stages reduce occupancy without improving latency.
THREE_STAGE_MAX_M = 32
TWO_STAGE_MAX_M = 128


def _select_pipeline_stages(m: int, k: int) -> int:
    """Select the pipeline from the logical GEMM M and K."""
    if k // SCALE_BLOCK_K < MULTISTAGE_MIN_K_BLOCKS:
        return 1
    if m <= THREE_STAGE_MAX_M:
        return 3
    if m <= TWO_STAGE_MAX_M:
        return 2
    return 1


@gluon.jit
def _fused_moe_blocked_fp8_small_m_kernel(
    a_ptr,
    a_scale_ptr,
    b_desc,
    b_scale_ptr,
    bias_ptr,
    c_ptr,
    sorted_idx_ptr,
    exp_end_ptr,
    block_end_ptr,
    block_expert_ids_ptr,
    block_offsets_ptr,
    N: gl.constexpr,
    K: gl.constexpr,
    stride_am: gl.constexpr,
    stride_ak: gl.constexpr,
    stride_asm: gl.constexpr,
    stride_ask: gl.constexpr,
    stride_bse: gl.constexpr,
    stride_bsn: gl.constexpr,
    stride_bsk: gl.constexpr,
    stride_bie: gl.constexpr,
    stride_bin: gl.constexpr,
    stride_cm: gl.constexpr,
    stride_cn: gl.constexpr,
    top_k: gl.constexpr,
    expert_offset: gl.constexpr,
    num_local_experts: gl.constexpr,
    reindex_a: gl.constexpr,
    reindex_c: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SCALE_N: gl.constexpr,
    NUM_PIPELINE_STAGES: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Compute C^T = B @ A^T for one expert M block and N tile."""
    gl.static_assert(a_ptr.dtype.element_ty == gl.float8e4nv,
                     'A must use FP8 E4M3')
    gl.static_assert(b_desc.dtype == gl.float8e4nv, 'B must use FP8 E4M3')
    gl.static_assert(isinstance(b_desc.layout, gl.NVMMASharedLayout),
                     'B must use an NVMMA descriptor')
    gl.static_assert(b_desc.block_type.shape == [BLOCK_N, BLOCK_K],
                     'B descriptor shape must match BLOCK_N x BLOCK_K')
    gl.static_assert(a_scale_ptr.dtype.element_ty == gl.float32,
                     'A scales must use FP32')
    gl.static_assert(b_scale_ptr.dtype.element_ty == gl.float32,
                     'B scales must use FP32')
    gl.static_assert(BLOCK_M == 8, 'small-M kernel requires BLOCK_M=8')
    gl.static_assert(BLOCK_N == 64 or BLOCK_N == 128,
                     'BLOCK_N must be 64 or 128')
    gl.static_assert(BLOCK_K == 128,
                     'BLOCK_K must match blocked-FP8 K scaling')
    gl.static_assert(SCALE_N == 128,
                     'SCALE_N must match blocked-FP8 N scaling')
    gl.static_assert(NUM_WARPS == 4, 'A load layout requires four warps')
    gl.static_assert(K % BLOCK_K == 0, 'K must contain whole scale blocks')

    m_block_id = gl.program_id(0)
    n_block_id = gl.program_id(1)
    num_m_blocks = gl.load(block_end_ptr + num_local_experts - 1)
    if m_block_id >= num_m_blocks:
        return

    local_expert = gl.load(block_expert_ids_ptr + m_block_id)
    global_expert = local_expert + expert_offset
    sorted_m_start = gl.load(block_offsets_ptr + m_block_id)
    expert_end = gl.load(exp_end_ptr + global_expert)

    # The four compute warps gather eight independently routed rows while each
    # row retains contiguous 128-byte K-block accesses.
    a_load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [1, 4],
                                                   [1, 0])
    load_m = gl.arange(0,
                       BLOCK_M,
                       layout=gl.SliceLayout(1, a_load_layout))
    load_k = gl.arange(0,
                       BLOCK_K,
                       layout=gl.SliceLayout(0, a_load_layout))
    load_sorted_m = sorted_m_start + load_m
    load_mask_m = load_sorted_m < expert_end
    load_source_m = gl.load(sorted_idx_ptr + load_sorted_m,
                            mask=load_mask_m,
                            other=0)
    if reindex_a:
        load_a_row = load_source_m // top_k
    else:
        load_a_row = load_sorted_m

    a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_K], gl.float8e4nv)
    a_smem = gl.allocate_shared_memory(
        gl.float8e4nv,
        [NUM_PIPELINE_STAGES, BLOCK_M, BLOCK_K],
        a_smem_layout,
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype,
        [NUM_PIPELINE_STAGES] + b_desc.block_type.shape,
        b_desc.layout,
    )
    b_ready = gl.allocate_shared_memory(
        gl.int64,
        [NUM_PIPELINE_STAGES, 1],
        mbarrier.MBarrierLayout(),
    )
    for init_stage in gl.static_range(0, NUM_PIPELINE_STAGES):
        mbarrier.init(b_ready.index(init_stage), count=1)

    # Logical C[M, N] = A[M, K] @ B[N, K]^T. We issue the transposed GEMM
    # C^T = B @ A^T, so WGMMA M is logical N and WGMMA N is logical M.
    accumulator_layout: gl.constexpr = pick_wgmma_layout(
        gl.float8e4nv, BLOCK_N, BLOCK_M, NUM_WARPS)
    accumulator = gl.zeros((BLOCK_N, BLOCK_M),
                           dtype=gl.float32,
                           layout=accumulator_layout)
    m_layout: gl.constexpr = gl.SliceLayout(0, accumulator_layout)
    n_layout: gl.constexpr = gl.SliceLayout(1, accumulator_layout)

    sorted_m = sorted_m_start + gl.arange(0, BLOCK_M, layout=m_layout)
    mask_m = sorted_m < expert_end
    # Reload M indices in the accumulator layout. Reusing load_source_m would
    # require a cross-layout conversion from a_load_layout.
    source_m = gl.load(sorted_idx_ptr + sorted_m, mask=mask_m, other=0)
    if reindex_a:
        a_scale_m = source_m // top_k
    else:
        a_scale_m = sorted_m

    local_expert_i64 = local_expert.to(gl.int64)
    b_row = (local_expert * N + n_block_id * BLOCK_N).to(gl.int32)
    num_k_blocks: gl.constexpr = K // BLOCK_K
    if NUM_PIPELINE_STAGES == 1:
        # With no independent stage to prepare, retire each WGMMA in the same
        # iteration. This keeps the async accumulator and gathered A tile
        # from becoming simultaneous loop-carried register state.
        phase = 0
        stage_b_ready = b_ready.index(0)
        stage_b = b_smem.index(0)
        stage_a = a_smem.index(0)
        for k_block in range(0, num_k_blocks):
            mbarrier.expect(stage_b_ready, b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(
                b_desc,
                [b_row, k_block * BLOCK_K],
                stage_b_ready,
                stage_b,
            )

            a_scale = gl.load(
                a_scale_ptr + a_scale_m * stride_asm + k_block * stride_ask,
                mask=mask_m,
                other=1.0,
            )
            b_scale = gl.load(
                b_scale_ptr + local_expert_i64 * stride_bse +
                (n_block_id * BLOCK_N // SCALE_N) * stride_bsn +
                k_block * stride_bsk)
            a_ptrs = (
                a_ptr + load_a_row[:, None] * stride_am +
                (k_block * BLOCK_K + load_k[None, :]) * stride_ak)
            a = gl.load(a_ptrs, mask=load_mask_m[:, None], other=0.0)
            stage_a.store(a)
            gl.barrier()
            mbarrier.wait(stage_b_ready, phase=phase)
            phase ^= 1

            raw_accumulator = warpgroup_mma(
                stage_b,
                stage_a.permute((1, 0)),
                gl.zeros_like(accumulator),
                is_async=True,
                use_acc=False,
            )
            raw_accumulator = warpgroup_mma_wait(num_outstanding=0,
                                                 deps=(raw_accumulator, ))
            accumulator += raw_accumulator * (a_scale * b_scale)[None, :]

        mbarrier.invalidate(stage_b_ready)
    else:
        # Leave one B stage free. The initialized WGMMA token lets the
        # first loop iteration run the normal wait/promote/refill sequence:
        # waiting returns zero, then the free stage receives the final preload.
        num_preload: gl.constexpr = min(NUM_PIPELINE_STAGES - 1, num_k_blocks)
        for preload_block in gl.static_range(0, num_preload):
            preload_stage = preload_block % NUM_PIPELINE_STAGES
            preload_b_ready = b_ready.index(preload_stage)
            preload_b = b_smem.index(preload_stage)
            mbarrier.expect(preload_b_ready, b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(
                b_desc,
                [b_row, preload_block * BLOCK_K],
                preload_b_ready,
                preload_b,
            )

        raw_accumulator = warpgroup_mma_init(gl.zeros_like(accumulator))
        a_scale = gl.zeros((BLOCK_M, ), dtype=gl.float32, layout=m_layout) + 1.0
        b_scale = 1.0
        for k_block in range(0, num_k_blocks):
            pipeline_stage = k_block % NUM_PIPELINE_STAGES
            stage_phase = (k_block // NUM_PIPELINE_STAGES) & 1
            stage_b_ready = b_ready.index(pipeline_stage)
            stage_b = b_smem.index(pipeline_stage)
            stage_a = a_smem.index(pipeline_stage)

            current_a_scale = gl.load(
                a_scale_ptr + a_scale_m * stride_asm + k_block * stride_ask,
                mask=mask_m,
                other=1.0,
            )
            current_b_scale = gl.load(
                b_scale_ptr + local_expert_i64 * stride_bse +
                (n_block_id * BLOCK_N // SCALE_N) * stride_bsn +
                k_block * stride_bsk)
            current_a_ptrs = (
                a_ptr + load_a_row[:, None] * stride_am +
                (k_block * BLOCK_K + load_k[None, :]) * stride_ak)
            current_a = gl.load(current_a_ptrs,
                                mask=load_mask_m[:, None],
                                other=0.0)
            stage_a.store(current_a)
            gl.barrier()

            raw_accumulator = warpgroup_mma_wait(num_outstanding=0,
                                                 deps=(raw_accumulator, ))
            next_b_block = k_block + NUM_PIPELINE_STAGES - 1
            if next_b_block < num_k_blocks:
                # The producer stage is either initially free (block zero) or
                # was consumed by the WGMMA that just completed.
                producer_stage = next_b_block % NUM_PIPELINE_STAGES
                fence_async_shared()
                producer_b_ready = b_ready.index(producer_stage)
                producer_b = b_smem.index(producer_stage)
                mbarrier.expect(producer_b_ready, b_desc.block_type.nbytes)
                tma.async_copy_global_to_shared(
                    b_desc,
                    [b_row, next_b_block * BLOCK_K],
                    producer_b_ready,
                    producer_b,
                )
            accumulator += raw_accumulator * (a_scale * b_scale)[None, :]

            mbarrier.wait(stage_b_ready, phase=stage_phase)
            raw_accumulator = gl.zeros_like(accumulator)
            raw_accumulator = warpgroup_mma(
                stage_b,
                stage_a.permute((1, 0)),
                raw_accumulator,
                is_async=True,
                use_acc=False,
            )
            a_scale = current_a_scale
            b_scale = current_b_scale

        # Epilogue: promote the final unscaled K-block partial.
        raw_accumulator = warpgroup_mma_wait(num_outstanding=0,
                                             deps=(raw_accumulator, ))
        accumulator += raw_accumulator * (a_scale * b_scale)[None, :]

        for final_stage in gl.static_range(0, NUM_PIPELINE_STAGES):
            mbarrier.invalidate(b_ready.index(final_stage))

    offs_n = n_block_id * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    mask_n = offs_n < N
    if bias_ptr is not None:
        bias = gl.load(
            bias_ptr + local_expert_i64 * stride_bie + offs_n * stride_bin,
            mask=mask_n,
            other=0.0,
        )
        accumulator += bias[:, None]

    if reindex_c:
        offs_m = source_m
    else:
        offs_m = sorted_m
    c_ptrs = (c_ptr + offs_m[None, :] * stride_cm +
              offs_n[:, None] * stride_cn)
    mask = mask_n[:, None] & mask_m[None, :]
    gl.store(c_ptrs,
             accumulator.to(c_ptr.dtype.element_ty),
             mask=mask)


def fused_moe_blocked_fp8_kernel_launcher(
    A: torch.Tensor,
    A_scale: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    sorted_idx: torch.Tensor,
    exp_end: torch.Tensor,
    block_end: torch.Tensor,
    block_expert_ids: torch.Tensor,
    block_offsets: torch.Tensor,
    bias: torch.Tensor | None = None,
    top_k: int = 1,
    expert_offset: int = 0,
    reindex_a: bool = True,
    reindex_c: bool = True,
    block_n: int = 128,
    num_stages: int | None = None,
) -> None:
    """Launch the workload-selected transposed Gluon routed GEMM."""
    num_local_experts, n, k = B.shape
    assert block_n in (64, 128), 'block_n must be 64 or 128'
    assert k % SCALE_BLOCK_K == 0 and n % SCALE_BLOCK_N == 0, \
        'K and N must be divisible by 128'
    assert A_scale.shape == (
        A.size(0),
        k // SCALE_BLOCK_K,
    ), 'A_scale must use one scale per 128 A columns'
    assert B_scale.shape == (
        num_local_experts,
        n // SCALE_BLOCK_N,
        k // SCALE_BLOCK_K,
    ), 'B_scale must use one scale per 128x128 B block'

    flat_b = B.view(num_local_experts * n, k)
    b_block_shape = [block_n, SCALE_BLOCK_K]
    b_layout = gl.NVMMASharedLayout.get_default_for(b_block_shape,
                                                    gl.float8e4nv)
    b_desc = TensorDescriptor.from_tensor(flat_b, b_block_shape, b_layout)

    C = C.view(-1, n)
    if num_stages is None:
        m = A.size(0) * top_k if reindex_a else A.size(0)
        num_stages = _select_pipeline_stages(m, k)
    assert num_stages in (1, 2, 3), 'num_stages must be 1, 2, or 3'
    grid = (block_expert_ids.numel(), triton.cdiv(n, block_n))
    _fused_moe_blocked_fp8_small_m_kernel[grid](
        A,
        A_scale,
        b_desc,
        B_scale,
        bias,
        C,
        sorted_idx,
        exp_end,
        block_end,
        block_expert_ids,
        block_offsets,
        N=n,
        K=k,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_asm=A_scale.stride(0),
        stride_ask=A_scale.stride(1),
        stride_bse=B_scale.stride(0),
        stride_bsn=B_scale.stride(1),
        stride_bsk=B_scale.stride(2),
        stride_bie=bias.stride(0) if bias is not None else 0,
        stride_bin=bias.stride(1) if bias is not None else 0,
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        top_k=top_k,
        expert_offset=expert_offset,
        num_local_experts=num_local_experts,
        reindex_a=reindex_a,
        reindex_c=reindex_c,
        BLOCK_M=SMALL_M_BLOCK,
        BLOCK_N=block_n,
        BLOCK_K=SCALE_BLOCK_K,
        SCALE_N=SCALE_BLOCK_N,
        NUM_PIPELINE_STAGES=num_stages,
        NUM_WARPS=4,
        num_warps=4,
    )
