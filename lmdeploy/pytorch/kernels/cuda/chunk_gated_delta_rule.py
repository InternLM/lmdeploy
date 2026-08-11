# Copyright (c) OpenMMLab. All rights reserved.
# Ported forward-only kernels for the chunked gated delta rule.
#
# The implementation is a triton-to-triton port of the forward path from
# flash-linear-attention (fla.ops.gated_delta_rule), stripped of the backend
# dispatch machinery, the autograd wrapper and every backward kernel (lmdeploy
# is inference-only). The inter-chunk state kernel already materializes the
# recurrent state at every chunk boundary; we expose that tensor as
# ``chunk_states`` so downstream prefix-caching can checkpoint per chunk.
#
# Source covered here:
#   fla/ops/utils/index.py        (prepare_chunk_indices / prepare_chunk_offsets)
#   fla/ops/utils/cumsum.py       (chunk_local_cumsum_scalar)
#   fla/ops/gated_delta_rule/chunk_fwd.py   (kkt + solve_tril fused kernel, BT=64)
#   fla/ops/gated_delta_rule/wy_fast.py     (recompute_w_u_fwd)
#   fla/ops/common/chunk_delta_h.py         (fwd inter-chunk state passing)
#   fla/ops/common/chunk_o.py               (chunk_fwd_kernel_o)
#   fla/ops/gated_delta_rule/chunk.py       (fwd orchestration, no autograd)
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

RCP_LN2 = 1.4426950408889634073599246810018921374266459541529859341354494069313

# FLA selects TF32 only on Ampere or newer. This module is CUDA-only and is
# imported after the CUDA backend is selected, so matching that capability
# check here keeps the fused triangular solve portable to older cards.
_IS_TF32_SUPPORTED = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32' if _IS_TF32_SUPPORTED else 'ieee')

if torch.cuda.is_available():
    _DEVICE_PROPERTIES = torch.cuda.get_device_properties(torch.cuda.current_device())
    _MAX_SHARED_MEMORY = getattr(
        _DEVICE_PROPERTIES,
        'shared_memory_per_block_optin',
        _DEVICE_PROPERTIES.shared_memory_per_block,
    )
    _IS_NVIDIA_BLACKWELL = _DEVICE_PROPERTIES.major in (10, 12)
else:
    _MAX_SHARED_MEMORY = 0
    _IS_NVIDIA_BLACKWELL = False

# Match FLA's guarded state-forward search space. Four warps can race in the
# recurrent tl.dot loop on Blackwell, while wider BV/stage configurations can
# exceed shared memory on smaller GPUs.
_STATE_FWD_NUM_WARPS = [2] if _IS_NVIDIA_BLACKWELL else [2, 4]
_STATE_FWD_NUM_STAGES = [2, 3, 4] if _MAX_SHARED_MEMORY >= 166912 else [2, 1]
_STATE_FWD_BV = [32, 64] if _MAX_SHARED_MEMORY >= 101376 else [32]


@triton.jit
def exp2(x):
    # Matches fla.ops.utils.op.exp2 (default, non-fast path).
    return tl.math.exp2(x.to(tl.float32))


# ---------------------------------------------------------------------------
# chunk index helpers (ported from fla/ops/utils/index.py)
# ---------------------------------------------------------------------------
def _segmented_arange(counts: torch.Tensor):
    """Expand per-segment counts into flat per-slot index tensors."""
    seg_id = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=counts.dtype),
        counts,
    )
    seg_start = F.pad(counts.cumsum(0), (1, 0))[:-1]
    intra_idx = torch.arange(seg_id.shape[0], device=counts.device, dtype=counts.dtype) - seg_start[seg_id]
    return seg_id, intra_idx


def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return torch.diff(cu_seqlens)


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_counts = (prepare_lens(cu_seqlens) + (chunk_size - 1)).div(chunk_size, rounding_mode='floor')
    seg_id, intra_chunk_idx = _segmented_arange(chunk_counts)
    return torch.stack([seg_id, intra_chunk_idx], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)


def _chunk_count_bucket(num_chunks: int) -> int:
    if num_chunks <= 4:
        return 4
    if num_chunks <= 16:
        return 16
    if num_chunks <= 64:
        return 64
    if num_chunks <= 128:
        return 128
    return 129


# ---------------------------------------------------------------------------
# gate chunk-local cumsum (ported from fla/ops/utils/cumsum.py, scalar path)
# ---------------------------------------------------------------------------
@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=['B', 'H', 'BT', 'IS_VARLEN', 'REVERSE', 'NT_BUCKET'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    NT_BUCKET: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if HEAD_FIRST:
        p_s = s + bos * H + i_h * T + o_t
        p_o = o + bos * H + i_h * T + o_t
    else:
        p_s = s + bos * H + i_h + o_t * H
        p_o = o + bos * H + i_h + o_t * H
    b_s = tl.load(p_s, mask=m_t, other=0.0).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t)


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2**(chunk_size.bit_length() - 1), 'chunk_size must be a power of 2'
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    chunk_local_cumsum_scalar_kernel[(NT, B * H)](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        NT_BUCKET=_chunk_count_bucket(NT),
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g


# ---------------------------------------------------------------------------
# intra-chunk: fused beta*K@K^T (lower-tri) + solve_tril (I+A)^{-1}
# ported from fla/ops/gated_delta_rule/chunk_fwd.py (BT=64 fused path only)
# ---------------------------------------------------------------------------
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({'BK': BK}, num_warps=num_warps) for BK in [32, 64] for num_warps in [1, 2, 4]],
    key=['H', 'HV', 'K', 'BC', 'NT_BUCKET'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NT_BUCKET: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k += (bos * H + i_h // (HV // H)) * K
    A += (bos * HV + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    p_b0 = beta + bos * HV + i_h + (i_tc0 + o_i) * HV
    p_b1 = beta + bos * HV + i_h + (i_tc1 + o_i) * HV
    p_b2 = beta + bos * HV + i_h + (i_tc2 + o_i) * HV
    p_b3 = beta + bos * HV + i_h + (i_tc3 + o_i) * HV
    b_b0 = tl.load(p_b0, mask=m_tc0, other=0.0).to(tl.float32)
    b_b1 = tl.load(p_b1, mask=m_tc1, other=0.0).to(tl.float32)
    b_b2 = tl.load(p_b2, mask=m_tc2, other=0.0).to(tl.float32)
    b_b3 = tl.load(p_b3, mask=m_tc3, other=0.0).to(tl.float32)

    if USE_G:
        p_g0 = g + bos * HV + i_h + (i_tc0 + o_i) * HV
        p_g1 = g + bos * HV + i_h + (i_tc1 + o_i) * HV
        p_g2 = g + bos * HV + i_h + (i_tc2 + o_i) * HV
        p_g3 = g + bos * HV + i_h + (i_tc3 + o_i) * HV
        b_g0 = tl.load(p_g0, mask=m_tc0, other=0.0).to(tl.float32)
        b_g1 = tl.load(p_g1, mask=m_tc1, other=0.0).to(tl.float32)
        b_g2 = tl.load(p_g2, mask=m_tc2, other=0.0).to(tl.float32)
        b_g3 = tl.load(p_g3, mask=m_tc3, other=0.0).to(tl.float32)

    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)

    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_k0 = k + (i_tc0 + o_i)[:, None] * (H * K) + o_k[None, :]
        b_k0 = tl.load(p_k0, mask=m_tc0[:, None] & (o_k[None, :] < K), other=0.0)
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))

        if i_tc1 < T:
            p_k1 = k + (i_tc1 + o_i)[:, None] * (H * K) + o_k[None, :]
            b_k1 = tl.load(p_k1, mask=m_tc1[:, None] & (o_k[None, :] < K), other=0.0)
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))

            if i_tc2 < T:
                p_k2 = k + (i_tc2 + o_i)[:, None] * (H * K) + o_k[None, :]
                b_k2 = tl.load(p_k2, mask=m_tc2[:, None] & (o_k[None, :] < K), other=0.0)
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))

                if i_tc3 < T:
                    p_k3 = k + (i_tc3 + o_i)[:, None] * (H * K) + o_k[None, :]
                    b_k3 = tl.load(p_k3, mask=m_tc3[:, None] & (o_k[None, :] < K), other=0.0)
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    if USE_G:
        b_A00 *= tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], exp2(b_g0[:, None] - b_g0[None, :]), 0.)
        b_A11 *= tl.where(m_d & m_tc1[:, None] & m_tc1[None, :], exp2(b_g1[:, None] - b_g1[None, :]), 0.)
        b_A22 *= tl.where(m_d & m_tc2[:, None] & m_tc2[None, :], exp2(b_g2[:, None] - b_g2[None, :]), 0.)
        b_A33 *= tl.where(m_d & m_tc3[:, None] & m_tc3[None, :], exp2(b_g3[:, None] - b_g3[None, :]), 0.)

        b_A10 *= tl.where(m_tc1[:, None] & m_tc0[None, :], exp2(b_g1[:, None] - b_g0[None, :]), 0.)
        b_A20 *= tl.where(m_tc2[:, None] & m_tc0[None, :], exp2(b_g2[:, None] - b_g0[None, :]), 0.)
        b_A21 *= tl.where(m_tc2[:, None] & m_tc1[None, :], exp2(b_g2[:, None] - b_g1[None, :]), 0.)
        b_A30 *= tl.where(m_tc3[:, None] & m_tc0[None, :], exp2(b_g3[:, None] - b_g0[None, :]), 0.)
        b_A31 *= tl.where(m_tc3[:, None] & m_tc1[None, :], exp2(b_g3[:, None] - b_g1[None, :]), 0.)
        b_A32 *= tl.where(m_tc3[:, None] & m_tc2[None, :], exp2(b_g3[:, None] - b_g2[None, :]), 0.)
    else:
        b_A00 = tl.where(m_d, b_A00, 0.)
        b_A11 = tl.where(m_d, b_A11, 0.)
        b_A22 = tl.where(m_d, b_A22, 0.)
        b_A33 = tl.where(m_d, b_A33, 0.)

    b_A00 = b_A00 * b_b0[:, None]
    b_A11 = b_A11 * b_b1[:, None]
    b_A22 = b_A22 * b_b2[:, None]
    b_A33 = b_A33 * b_b3[:, None]

    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
        b_a00 = tl.where(o_i < i, b_a00, 0.)
        b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.), 0)
        b_a11 = tl.where(o_i < i, b_a11, 0.)
        b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.), 0)
        b_a22 = tl.where(o_i < i, b_a22, 0.)
        b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.), 0)
        b_a33 = tl.where(o_i < i, b_a33, 0.)
        b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    p_A00 = A + (i_tc0 + o_i)[:, None] * (HV * BT) + o_i[None, :]
    p_A10 = A + (i_tc1 + o_i)[:, None] * (HV * BT) + o_i[None, :]
    p_A11 = A + (i_tc1 + o_i)[:, None] * (HV * BT) + (BC + o_i)[None, :]
    p_A20 = A + (i_tc2 + o_i)[:, None] * (HV * BT) + o_i[None, :]
    p_A21 = A + (i_tc2 + o_i)[:, None] * (HV * BT) + (BC + o_i)[None, :]
    p_A22 = A + (i_tc2 + o_i)[:, None] * (HV * BT) + (2 * BC + o_i)[None, :]
    p_A30 = A + (i_tc3 + o_i)[:, None] * (HV * BT) + o_i[None, :]
    p_A31 = A + (i_tc3 + o_i)[:, None] * (HV * BT) + (BC + o_i)[None, :]
    p_A32 = A + (i_tc3 + o_i)[:, None] * (HV * BT) + (2 * BC + o_i)[None, :]
    p_A33 = A + (i_tc3 + o_i)[:, None] * (HV * BT) + (3 * BC + o_i)[None, :]

    m_A0 = m_tc0[:, None] & (o_i[None, :] < BT)
    m_A1 = m_tc1[:, None] & (o_i[None, :] < BT)
    m_A2 = m_tc2[:, None] & (o_i[None, :] < BT)
    m_A3 = m_tc3[:, None] & (o_i[None, :] < BT)
    m_A11 = m_tc1[:, None] & ((BC + o_i)[None, :] < BT)
    m_A21 = m_tc2[:, None] & ((BC + o_i)[None, :] < BT)
    m_A22 = m_tc2[:, None] & ((2 * BC + o_i)[None, :] < BT)
    m_A31 = m_tc3[:, None] & ((BC + o_i)[None, :] < BT)
    m_A32 = m_tc3[:, None] & ((2 * BC + o_i)[None, :] < BT)
    m_A33 = m_tc3[:, None] & ((3 * BC + o_i)[None, :] < BT)

    tl.store(p_A00, b_Ai00.to(A.dtype.element_ty), mask=m_A0)
    tl.store(p_A10, b_Ai10.to(A.dtype.element_ty), mask=m_A1)
    tl.store(p_A11, b_Ai11.to(A.dtype.element_ty), mask=m_A11)
    tl.store(p_A20, b_Ai20.to(A.dtype.element_ty), mask=m_A2)
    tl.store(p_A21, b_Ai21.to(A.dtype.element_ty), mask=m_A21)
    tl.store(p_A22, b_Ai22.to(A.dtype.element_ty), mask=m_A22)
    tl.store(p_A30, b_Ai30.to(A.dtype.element_ty), mask=m_A3)
    tl.store(p_A31, b_Ai31.to(A.dtype.element_ty), mask=m_A31)
    tl.store(p_A32, b_Ai32.to(A.dtype.element_ty), mask=m_A32)
    tl.store(p_A33, b_Ai33.to(A.dtype.element_ty), mask=m_A33)


def chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
):
    assert chunk_size == 64, 'only the fused BT=64 kkt+solve path is ported'
    B, T, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size
    BC = 16

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * HV)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        NT_BUCKET=_chunk_count_bucket(NT),
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, A


# ---------------------------------------------------------------------------
# recompute w, u (ported from fla/ops/gated_delta_rule/wy_fast.py, fwd only)
# ---------------------------------------------------------------------------
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [2, 4, 8] for num_stages in [2, 3, 4]],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN', 'NT_BUCKET'],
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_BUCKET: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_b = beta + bos * HV + i_h + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    p_A = A + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos * HV + i_h) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        p_u = u + (bos * HV + i_h) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    if USE_G:
        p_g = g + (bos * HV + i_h) + o_t * HV
        b_g = exp2(tl.load(p_g, mask=m_t, other=0.0))

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos * H + i_h // (HV // H)) * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_w = w + (bos * HV + i_h) * K + o_t[:, None] * (HV * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb *= b_g[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
):
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    recompute_w_u_fwd_kernel[(NT, B * HV)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT_BUCKET=_chunk_count_bucket(NT),
    )
    return w, u


# ---------------------------------------------------------------------------
# inter-chunk state passing (ported from fla/ops/common/chunk_delta_h.py)
# Produces h: [B, NT, HV, V, K] (state_v_first) — the per-chunk boundary states.
# h[:, c] is the recurrent state at the START of chunk c, i.e. the state after
# processing chunks 0..c-1, representing tokens 0..c*BT. final_state (ht) is the
# state after the last chunk, representing all tokens.
# ---------------------------------------------------------------------------
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in _STATE_FWD_NUM_WARPS
            for num_stages in _STATE_FWD_NUM_STAGES
            for BV in _STATE_FWD_BV],
    key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST', 'NT_BUCKET'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT_BUCKET: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = pid % NV, (pid // NV).to(tl.int64)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    h += (boh * HV + i_h).to(tl.int64) * K * V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * HV + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0_1 = h0 + o_v[:, None] * K + o_k1[None, :]
            m_h0_1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
            m_h0_1 = m_k1[:, None] & m_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
        if K > 64:
            if STATE_V_FIRST:
                p_h0_2 = h0 + o_v[:, None] * K + o_k2[None, :]
                m_h0_2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
                m_h0_2 = m_k2[:, None] & m_v[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)
        if K > 128:
            if STATE_V_FIRST:
                p_h0_3 = h0 + o_v[:, None] * K + o_k3[None, :]
                m_h0_3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
                m_h0_3 = m_k3[:, None] & m_v[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h0_3, other=0.0).to(tl.float32)
        if K > 192:
            if STATE_V_FIRST:
                p_h0_4 = h0 + o_v[:, None] * K + o_k4[None, :]
                m_h0_4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
                m_h0_4 = m_k4[:, None] & m_v[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h0_4, other=0.0).to(tl.float32)

    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        if STATE_V_FIRST:
            p_h1 = h + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k1[None, :]
            m_h1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h1 = h + i_t_int64 * HV * K * V + o_k1[:, None] * V + o_v[None, :]
            m_h1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            if STATE_V_FIRST:
                p_h2 = h + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k2[None, :]
                m_h2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h2 = h + i_t_int64 * HV * K * V + o_k2[:, None] * V + o_v[None, :]
                m_h2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            if STATE_V_FIRST:
                p_h3 = h + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k3[None, :]
                m_h3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h3 = h + i_t_int64 * HV * K * V + o_k3[:, None] * V + o_v[None, :]
                m_h3 = m_k3[:, None] & m_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            if STATE_V_FIRST:
                p_h4 = h + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k4[None, :]
                m_h4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h4 = h + i_t_int64 * HV * K * V + o_k4[:, None] * V + o_v[None, :]
                m_h4 = m_k4[:, None] & m_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        p_w = w + o_t[:, None] * (HV * K) + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        if STATE_V_FIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * (HV * K) + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = w + o_t[:, None] * (HV * K) + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = w + o_t[:, None] * (HV * K) + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

        if SAVE_NEW_VALUE:
            p_v = v_new + o_t[:, None] * (HV * V) + o_v[None, :]
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * HV * K + i_h * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp2(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * HV * K + i_h * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp2(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * HV * K + i_h * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp2(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * HV * K + i_h * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp2(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * (H * K)
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = ht + o_v[:, None] * K + o_k1[None, :]
            m_ht = m_v[:, None] & m_k1[None, :]
        else:
            p_ht = ht + o_k1[:, None] * V + o_v[None, :]
            m_ht = m_k1[:, None] & m_v[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 64:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k2[None, :]
                m_ht = m_v[:, None] & m_k2[None, :]
            else:
                p_ht = ht + o_k2[:, None] * V + o_v[None, :]
                m_ht = m_k2[:, None] & m_v[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 128:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k3[None, :]
                m_ht = m_v[:, None] & m_k3[None, :]
            else:
                p_ht = ht + o_k3[:, None] * V + o_v[None, :]
                m_ht = m_k3[:, None] & m_v[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 192:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k4[None, :]
                m_ht = m_v[:, None] & m_k4[None, :]
            else:
                p_ht = ht + o_k4[:, None] * V + o_v[None, :]
                m_ht = m_k4[:, None] & m_v[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht)


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
):
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, 'current kernel does not support head dimension larger than 256.'

    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta): return (triton.cdiv(V, meta['BV']) * N * HV, )

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        NT_BUCKET=_chunk_count_bucket(NT),
        STATE_V_FIRST=state_v_first,
    )
    return h, v_new, final_state


# ---------------------------------------------------------------------------
# output projection (ported from fla/ops/common/chunk_o.py, fwd only)
# ---------------------------------------------------------------------------
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=3),
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=3),
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST', 'NT_BUCKET'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_BUCKET: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    v += (bos * HV + i_h) * V
    o += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        p_k = k + o_k[:, None] + o_t[None, :] * (H * K)
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = (o_v[:, None] < V) & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & (o_v[None, :] < V)
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_k = tl.load(p_k, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_h = tl.load(p_h, mask=m_h, other=0.0)

        if STATE_V_FIRST:
            b_o += tl.dot(b_q, tl.trans(b_h))
        else:
            b_o += tl.dot(b_q, b_h)
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * HV + i_h
        p_g = g + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
    p_o = o + o_t[:, None] * (HV * V) + o_v[None, :]

    b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & (o_v < V)[None, :])


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V, HV = *q.shape, v.shape[-1], v.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)

    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * HV)

    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        NT_BUCKET=_chunk_count_bucket(NT),
        STATE_V_FIRST=state_v_first,
    )
    return o


# ---------------------------------------------------------------------------
# forward orchestration (ported from fla/ops/gated_delta_rule/chunk.py, fwd only)
# ---------------------------------------------------------------------------
def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    state_v_first: bool = True,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
):
    """Chunked gated delta rule forward, local triton port.

    Mirrors ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` for the
    inference-only path lmdeploy uses (``use_gate_in_kernel=False``,
    ``use_qk_l2norm_in_kernel=False``: q/k already l2norm'd, beta already
    sigmoid'd, ``g`` is the per-token log decay). ``chunk_size`` is fixed to 64
    so chunk boundaries align with lmdeploy's 64-token KV blocks.

    Returns:
        o: output of shape ``[B, T, HV, V]``.
        final_state: recurrent state after all chunks (``[N, HV, V, K]`` when
            ``state_v_first=True``), or None if ``output_final_state=False``.
        chunk_states: per-chunk boundary recurrent state ``h`` of shape
            ``[B, NT, HV, V, K]`` (``state_v_first=True``). ``chunk_states[:, c]``
            is the state at the START of chunk c, i.e. the state after processing
            chunks 0..c-1, representing tokens 0..c*chunk_size. Thus the state
            for a block-aligned prefix of length s=c*64 is ``chunk_states[:, c]``
            (c>=1), and the full-sequence state is ``final_state``.
    """
    assert chunk_size == 64, 'only chunk_size=64 is supported (block alignment)'
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    # gate chunk-local cumsum, in log2 space (exp2(g_cumsum) == exp(g_cumsum))
    g = chunk_local_cumsum_scalar(
        g=g,
        chunk_size=BT,
        scale=RCP_LN2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    # intra-chunk WY representation: fused kkt + solve_tril, then recompute w/u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        chunk_indices=chunk_indices,
    )

    # inter-chunk state passing -> per-chunk boundary states + final state
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=BT,
        save_new_value=True,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )

    # output
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        chunk_indices=chunk_indices,
    )
    return o, final_state, h


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    state_v_first: bool,
    chunk_size: int,
    scale: float,
):
    """Validate the dense-layout forward contract before launching Triton."""
    if chunk_size != 64:
        raise ValueError(f'only chunk_size=64 is supported, got {chunk_size}')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f'scale must be finite and positive, got {scale}')

    tensors = {'q': q, 'k': k, 'v': v, 'g': g, 'beta': beta}
    if initial_state is not None:
        tensors['initial_state'] = initial_state
    if cu_seqlens is not None:
        tensors['cu_seqlens'] = cu_seqlens
    if chunk_indices is not None:
        tensors['chunk_indices'] = chunk_indices
    if chunk_offsets is not None:
        tensors['chunk_offsets'] = chunk_offsets
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'{name} must be a torch.Tensor')
        if not tensor.is_cuda:
            raise ValueError(f'{name} must be a CUDA tensor')
        if tensor.device != q.device:
            raise ValueError(f'{name} must be on {q.device}, got {tensor.device}')

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError('q, k and v must have shape [B, T, H, D]')
    B, T, H, K = q.shape
    if k.shape != q.shape:
        raise ValueError(f'q and k must have identical shapes, got {q.shape} and {k.shape}')
    if v.shape[:2] != (B, T):
        raise ValueError(f'v batch/token dimensions must be {(B, T)}, got {v.shape[:2]}')
    HV, V = v.shape[2:]
    if H <= 0 or HV <= 0 or HV % H != 0:
        raise ValueError(f'num value heads ({HV}) must be divisible by num key heads ({H})')
    if K > 256:
        raise ValueError(f'key head dimension must not exceed 256, got {K}')
    expected_gate_shape = (B, T, HV)
    if g.shape != expected_gate_shape or beta.shape != expected_gate_shape:
        raise ValueError(
            f'g and beta must have shape {expected_gate_shape}, got {g.shape} and {beta.shape}')
    if q.dtype != k.dtype or v.dtype != q.dtype:
        raise ValueError(f'q, k and v must have the same dtype, got {q.dtype}, {k.dtype}, {v.dtype}')
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f'unsupported q/k/v dtype: {q.dtype}')
    if not g.is_floating_point() or not beta.is_floating_point():
        raise ValueError('g and beta must be floating-point tensors')

    N = B
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(f'physical batch size must be 1 with cu_seqlens, got {B}')
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError('cu_seqlens must be a one-dimensional int32/int64 tensor')
        if cu_seqlens.numel() < 2:
            raise ValueError('cu_seqlens must contain at least a start and an end offset')
        N = cu_seqlens.numel() - 1
        if chunk_indices is not None:
            if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
                raise ValueError('chunk_indices must have shape [num_chunks, 2]')
            if chunk_indices.dtype not in (torch.int32, torch.int64):
                raise ValueError('chunk_indices must be an int32/int64 tensor')
        if chunk_offsets is not None:
            if chunk_offsets.ndim != 1 or chunk_offsets.numel() != N + 1:
                raise ValueError(f'chunk_offsets must have shape [{N + 1}]')
            if chunk_offsets.dtype not in (torch.int32, torch.int64):
                raise ValueError('chunk_offsets must be an int32/int64 tensor')
    elif chunk_indices is not None or chunk_offsets is not None:
        raise ValueError('chunk metadata requires cu_seqlens')

    if initial_state is not None:
        expected_state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
        if initial_state.shape != expected_state_shape:
            raise ValueError(
                f'initial_state must have shape {expected_state_shape}, got {tuple(initial_state.shape)}')


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    state_v_first: bool = True,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
):
    """Run the local chunked gated-delta forward implementation.

    The Triton kernels use dense row-major pointer arithmetic, as their FLA
    counterparts do. FLA's public API guarantees that precondition through
    ``input_guard``. Qwen supplies q/k/v as views of one fused QKV tensor, so
    reproducing that guard here is required for correctness (v commonly has a
    token stride larger than ``HV * V``).
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    _validate_inputs(q, k, v, g, beta, initial_state, cu_seqlens,
                     chunk_indices, chunk_offsets, state_v_first, chunk_size, scale)

    # Enter the input device before allocations/launches. ``contiguous`` is a
    # no-op for already dense tensors and copies real fused-QKV views exactly as
    # FLA's input_guard does.
    with torch.cuda.device(q.device):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        if initial_state is not None:
            initial_state = initial_state.contiguous()
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.contiguous()
        if chunk_indices is not None:
            chunk_indices = chunk_indices.contiguous()
        if chunk_offsets is not None:
            chunk_offsets = chunk_offsets.contiguous()
        o, final_state, chunk_states = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            state_v_first=state_v_first,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )
    return o, final_state, chunk_states
