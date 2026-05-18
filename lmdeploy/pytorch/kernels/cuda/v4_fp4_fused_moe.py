# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Callable

import torch
import triton
import triton.language as tl

from .activation import silu_and_mul
from .blocked_gemm_fp8 import quant_fp8
from .fused_moe import _get_sorted_idx, _make_intermediate, _renormalize, moe_reduce


@triton.jit
def fused_moe_v4_fp4_kernel(
    A,
    A_scale,
    B,
    B_scale,
    bias,
    C,
    SortedIdx,
    ExpStart,
    ExpEnd,
    N: tl.constexpr,
    K: tl.constexpr,
    group_ak: tl.constexpr,
    group_bk: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_asm,
    stride_ask: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bie: tl.constexpr,
    stride_bin: tl.constexpr,
    stride_cm,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    M_NP2: tl.constexpr,
    top_k: tl.constexpr,
    expert_offset: tl.constexpr,
    reindex_a: tl.constexpr,
    reindex_c: tl.constexpr,
    B_SCALE_E8M0: tl.constexpr,
):
    """Fused MoE GEMM with FP8 activations and checkpoint-native V4 FP4
    weights.

    Uses int32 load + prmt decode + join for 3x speedup over per-element decode.
    """
    exp_id = tl.program_id(1)
    pid = tl.program_id(0)

    exp_start = tl.load(ExpStart + exp_id + expert_offset)
    exp_end = tl.load(ExpEnd + exp_id + expert_offset)
    M = exp_end - exp_start
    if M <= 0:
        return

    num_pid_m = tl.cdiv(M_NP2, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= M or pid_n * BLOCK_SIZE_N >= N:
        return

    offs_sid = exp_start + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_sid = offs_sid < exp_end
    sid = tl.load(SortedIdx + offs_sid, mask=mask_sid, other=0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if reindex_a:
        offs_am = sid // top_k
    else:
        offs_am = offs_sid
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    exp_id = exp_id.to(tl.int64)
    exp_off = stride_be * exp_id
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    K8_BLOCK: tl.constexpr = BLOCK_SIZE_K // 8
    offs_k8 = tl.arange(0, K8_BLOCK)
    LUT_LO: tl.constexpr = 0x3C383000
    LUT_HI: tl.constexpr = 0x4C484440

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        global_k = k_start + offs_k
        k8_offs = k_start // 8 + offs_k8

        a = tl.load(a_ptrs, mask=mask_sid[:, None] & (global_k[None, :] < K), other=0.0)

        # B is int32*: each int32 = 4 packed uint8 = 8 FP4 values
        # Load as [BLOCK_SIZE_N, K8_BLOCK] to avoid transposing later
        b_i32 = tl.load(B + exp_off + offs_bn[:, None] * stride_bn + k8_offs[None, :] * stride_bk,
                         mask=k8_offs[None, :] < K // 8, other=0)

        # Low nibble decode
        mag_lo = b_i32 & 0x07070707
        sel_lo = (mag_lo & 0x00000007) | ((mag_lo >> 4) & 0x00000070) | \
            ((mag_lo >> 8) & 0x00000700) | ((mag_lo >> 12) & 0x00007000)
        result_lo = tl.inline_asm_elementwise(
            'prmt.b32 $0, $2, $3, $4;', '=r,r,r,r,r',
            args=[sel_lo, LUT_LO, LUT_HI, sel_lo],
            dtype=tl.uint32, is_pure=True, pack=1,
        )
        sign_lo = b_i32 & 0x08080808
        result_lo = result_lo ^ (sign_lo << 4)

        # High nibble decode — shift then mask to prevent cross-byte leakage
        b_shifted = (b_i32 >> 4) & 0x0F0F0F0F
        mag_hi = b_shifted & 0x07070707
        sel_hi = (mag_hi & 0x00000007) | ((mag_hi >> 4) & 0x00000070) | \
            ((mag_hi >> 8) & 0x00000700) | ((mag_hi >> 12) & 0x00007000)
        result_hi = tl.inline_asm_elementwise(
            'prmt.b32 $0, $2, $3, $4;', '=r,r,r,r,r',
            args=[sel_hi, LUT_LO, LUT_HI, sel_hi],
            dtype=tl.uint32, is_pure=True, pack=1,
        )
        sign_hi = b_shifted & 0x08080808
        result_hi = result_hi ^ (sign_hi << 4)

        # Extract 8 fp8 bytes: byte j low→K=k8*8+2j, byte j high→K=k8*8+2j+1
        b0 = (result_lo).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        b1 = (result_lo >> 8).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        b2 = (result_lo >> 16).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        b3 = (result_lo >> 24).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        b4 = (result_hi).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        b5 = (result_hi >> 8).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        b6 = (result_hi >> 16).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        b7 = (result_hi >> 24).to(tl.uint8).to(tl.float8e4nv, bitcast=True)

        # Already [BLOCK_SIZE_N, K8_BLOCK] from load — join with v2 pairing
        j_lo_02 = tl.join(b0, b2)
        j_lo_13 = tl.join(b1, b3)
        j_lo = tl.join(j_lo_02, j_lo_13)
        j_hi_02 = tl.join(b4, b6)
        j_hi_13 = tl.join(b5, b7)
        j_hi = tl.join(j_hi_02, j_hi_13)
        # join(low, high) interleaves → K order [0,4,1,5,2,6,3,7]
        j_all = tl.join(j_lo, j_hi)
        flat = j_all.reshape(BLOCK_SIZE_N, K8_BLOCK * 8)
        b = flat.trans()  # [BLOCK_SIZE_K, BLOCK_SIZE_N]

        offs_ksa = k_start // group_ak
        offs_ksb = k_start // group_bk
        a_scale = tl.load(A_scale + offs_am * stride_asm + offs_ksa * stride_ask, mask=mask_sid, other=1.0)
        b_scale = tl.load(B_scale + stride_bse * exp_id + offs_bn * stride_bsn + offs_ksb * stride_bsk)
        if B_SCALE_E8M0:
            b_scale = (b_scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak

    if bias is not None:
        bias_ptrs = bias + exp_id * stride_bie + offs_bn * stride_bin
        bias_val = tl.load(bias_ptrs).to(accumulator.dtype)
        accumulator += bias_val[None]

    c = accumulator.to(C.dtype.element_ty)

    if reindex_c:
        offs_cm = sid
    else:
        offs_cm = offs_sid
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=mask_sid[:, None])


# Tile configs keyed by per-expert M range: (BM, BN, num_stages, num_warps)
# BN=128 is critical for HBM read coalescing — BN=64 gives ~3x lower B-bandwidth.
# 4 warps at small BM hides latency better than 2 warps (+16% at M<=16).
# BM>=64 triggers WGMMA FP8 path (vs mma.sync FP16), which only wins at large
# per-expert M (>=4) where CTA count isn't the bottleneck.
_TILE_CONFIGS = [
    (16, 128, 4, 4),   # M <= 16
    (64, 128, 4, 4),   # M <= 64
    (128, 128, 4, 4),  # M > 64
]
_TILE_THRESHOLDS = [16, 64]


def _select_tile_config(num_tokens: int, num_experts: int, top_k: int):
    """Select BM/BN based on estimated per-expert token count."""
    m_per_exp = num_tokens * top_k / num_experts
    for i, thresh in enumerate(_TILE_THRESHOLDS):
        if m_per_exp <= thresh:
            return _TILE_CONFIGS[i]
    return _TILE_CONFIGS[-1]


def fused_moe_v4_fp4_kernel_launcher(
    A: torch.Tensor,
    A_scale: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    sorted_idx: torch.Tensor,
    exp_start: torch.Tensor,
    exp_end: torch.Tensor,
    bias: torch.Tensor = None,
    top_k: int = 1,
    num_tokens: int = None,
    expert_offset: int = 0,
    reindex_a: bool = True,
    reindex_c: bool = True,
    routing_topk: int = None,
):
    """Launch the V4 FP8xFP4 fused MoE GEMM kernel.

    routing_topk is used for tile selection to estimate per-expert M. For gate_up, routing_topk=topk (each token routes
    to topk experts). For down, top_k=1 (no reindexing) but routing_topk=topk because each expert still processes
    M*topk/E rows on average.
    """
    if num_tokens is None:
        num_tokens = A.size(0)
    E, N, packed_K = B.shape
    K = packed_K * 2

    assert A.dim() == 2
    assert A_scale.dim() == 2
    assert B.dim() == 3
    assert B_scale.dim() == 3
    assert A.size(1) == K
    assert B_scale.size(1) == N
    assert K % A_scale.size(1) == 0
    assert K % B_scale.size(2) == 0

    # B layout checks for int32 pack + prmt decode
    assert B.dtype in (torch.int8, torch.uint8), (
        f"B must be int8/uint8 (packed FP4), got {B.dtype}")
    assert B.stride(-1) == 1, (
        f"B must have stride[-1]==1 for int32 view, got stride {B.stride()}")
    assert B.shape[-1] % 4 == 0, (
        f"B packed_K (last dim) must be a multiple of 4 for int32 packing, "
        f"got {B.shape[-1]}")

    group_ak = K // A_scale.size(1)
    group_bk = K // B_scale.size(2)
    assert group_bk == 32, 'DeepSeek-V4 FP4 weights use per-32 K scales.'
    b_scale_e8m0 = B_scale.dtype == torch.float8_e8m0fnu
    if b_scale_e8m0:
        B_scale = B_scale.view(torch.uint8)

    # View B as int32: [E, N, packed_K] int8 → [E, N, packed_K//4] int32
    # Kernel uses int32 pointer with strides in int32 elements
    B_i32 = B.view(torch.int32)

    tk = routing_topk if routing_topk is not None else top_k
    BM, BN, num_stages, num_warps = _select_tile_config(num_tokens, E, tk)
    M_NP2 = max(64, triton.next_power_of_2(num_tokens))

    grid = (triton.cdiv(M_NP2, BM) * triton.cdiv(N, BN), E)

    A = A.flatten(0, -2)
    C = C.flatten(0, -2)
    enable_bias = bias is not None

    BLOCK_SIZE_K = group_bk
    GROUP_SIZE_M = 1
    fused_moe_v4_fp4_kernel[grid](
        A,
        A_scale,
        B_i32,
        B_scale,
        bias,
        C,
        sorted_idx,
        exp_start,
        exp_end,
        N=N,
        K=K,
        group_ak=group_ak,
        group_bk=group_bk,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_asm=A_scale.stride(0),
        stride_ask=A_scale.stride(1),
        stride_be=B_i32.stride(0),
        stride_bn=B_i32.stride(1),
        stride_bk=B_i32.stride(2),
        stride_bse=B_scale.stride(0),
        stride_bsn=B_scale.stride(1),
        stride_bsk=B_scale.stride(2),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        stride_bie=bias.stride(0) if enable_bias else 0,
        stride_bin=bias.stride(1) if enable_bias else 0,
        top_k=top_k,
        expert_offset=expert_offset,
        reindex_a=reindex_a,
        reindex_c=reindex_c,
        B_SCALE_E8M0=b_scale_e8m0,
        M_NP2=M_NP2,
        BLOCK_SIZE_M=BM,
        BLOCK_SIZE_N=BN,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=num_stages,
        num_warps=num_warps,
    )


def fused_moe_v4_fp4(input: torch.Tensor,
                     input_scale: torch.Tensor,
                     w1: torch.Tensor,
                     w1_scale: torch.Tensor,
                     w2: torch.Tensor,
                     w2_scale: torch.Tensor,
                     topk_weights: torch.Tensor,
                     topk_ids: torch.Tensor,
                     topk: int,
                     w1_bias: torch.Tensor = None,
                     w2_bias: torch.Tensor = None,
                     out_dtype: torch.dtype = torch.float16,
                     expert_offset: int = 0,
                     num_experts: int = None,
                     renormalize: bool = False,
                     act_func: Callable = None) -> torch.Tensor:
    """Fused MoE for DeepSeek-V4 checkpoint-native packed FP4 expert
    weights."""
    device = input.device
    M = input.size(0)
    E, N, _ = w1.shape
    if num_experts is None:
        num_experts = E
    full_exp = num_experts == E
    group_size = input.size(-1) // input_scale.size(-1)

    topk_weights = _renormalize(topk_weights, renormalize)
    sorted_idx, exp_start, exp_end = _get_sorted_idx(topk_ids, num_experts)

    intermediate_cache1 = _make_intermediate((M, topk, N), dtype=out_dtype, device=device, zeros=not full_exp)
    fused_moe_v4_fp4_kernel_launcher(
        input,
        input_scale,
        w1,
        w1_scale,
        intermediate_cache1,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        bias=w1_bias,
        top_k=topk,
        num_tokens=M,
        expert_offset=expert_offset,
        reindex_a=True,
        reindex_c=False,
    )

    intermediate_cache1 = intermediate_cache1.flatten(0, -2)
    if act_func is None:
        gate_cache = silu_and_mul(intermediate_cache1)
    else:
        gate_cache = act_func(intermediate_cache1)
    del intermediate_cache1
    gate_cache, gate_scale = quant_fp8(gate_cache, group_size, dtype=input.dtype, scale_fmt='ue8m0')

    intermediate_cache2 = _make_intermediate((M, topk, w2.shape[1]), dtype=out_dtype, device=device, zeros=not full_exp)
    fused_moe_v4_fp4_kernel_launcher(
        gate_cache,
        gate_scale,
        w2,
        w2_scale,
        intermediate_cache2,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        bias=w2_bias,
        top_k=1,
        num_tokens=M,
        expert_offset=expert_offset,
        reindex_a=False,
        reindex_c=True,
        routing_topk=topk,
    )

    ret = moe_reduce(intermediate_cache2, topk_weights)
    return ret
