# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# fused_recurrent_gated_delta_rule_fwd_kernel      : Triton prefill kernel for gated delta rule.
# fused_sigmoid_gating_delta_rule_update_kernel    : Triton decode kernel for sigmoid-gated delta rule.
# fused_sigmoid_gating_delta_rule_update           : Python wrapper dispatching the decode kernel.
#   Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/fla/sigmoid_gating.py
#   Original source: https://github.com/fla-org/flash-linear-attention (MIT License)
#   Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors

import os

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    div = tldevice.fast_dividef
    exp = tldevice.fast_expf
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:

    @triton.jit
    def div_normal(x, y):
        return x / y

    div = div_normal
    exp = tl.exp
    log = tl.log
    log2 = tl.log2


@triton.heuristics(
    {
        'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
        'IS_CONTINUOUS_BATCHING': lambda args: args['ssm_state_indices'] is not None,
        'IS_SPEC_DECODING': lambda args: args['num_accepted_tokens'] is not None,
    }
)
@triton.jit(do_not_specialize=['N', 'T'])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.constexpr,  # num of sequences
    T: tl.constexpr,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            p_h0 = (
                h0 + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64) * stride_init_state_token
            )
        else:
            p_h0 = h0 + bos * HV * K * V
        p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        p_q = q + (bos * H + i_h) * K + o_k + H * K * i_t
        p_k = k + (bos * H + i_h) * K + o_k + H * K * i_t
        p_v = v + (bos * HV + i_hv) * V + o_v + HV * V * i_t

        if IS_BETA_HEADWISE:
            p_beta = beta + (bos * HV + i_hv) * V + o_v + HV * V * i_t
        else:
            p_beta = beta + bos * HV + i_hv + HV * i_t

        if not IS_KDA:
            p_g = g + bos * HV + i_hv + HV * i_t
        else:
            p_gk = g + (bos * HV + i_hv + HV * i_t) * K + o_k

        p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v + HV * V * i_t

        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BK, BV]
        # b_h *= tl.exp(b_g)
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= exp(b_gk[:, None])
        # [BV]
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BK, BV]
        b_h += b_k[:, None] * b_v[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
            p_ht = (
                ht + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64) * stride_final_state_token
            )
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
        p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.heuristics(
    {
        'USE_INITIAL_STATE': lambda args: args['h0_source'] is not None,
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.jit(do_not_specialize=['T'])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Fused kernel that combines sigmoid gating computation with recurrent
    delta rule update."""
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        # if idx >= 0:
        tmp0 = tl.where(idx < 0, 0, idx)
        p_h0 = h0_source + tmp0 * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        temp1 = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
        temp2 = tl.zeros_like(temp1)
        value0 = tl.where(idx < 0, temp2, temp1)
        b_h += value0  # tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i in range(0, T):
        # Load inputs
        b_q = tl.load(p_q + i * H * K, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k + i * H * K, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v + i * HV * V, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b + i * HV).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a + i * HV).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o + i * HV * V, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # # Update pointers for next timestep
        # p_q += H * K
        # p_k += H * K
        # p_o += HV * V
        # p_v += HV * V
        # p_b += HV
        # p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor = None,
):
    """Fused triton implementation of sigmoid gating delta rule update.

    This function uses a single fused kernel that combines both sigmoid gating computation and the recurrent delta rule
    update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported yet'
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, 'scale must be positive'

    o = q.new_empty(NK, *v.shape)
    grid = (NK, NV, N * HV)

    if not initial_state_indices.is_contiguous():
        initial_state_indices = initial_state_indices.contiguous()
    if not initial_state_source.is_contiguous():
        initial_state_source = initial_state_source.contiguous()
    if not cu_seqlens.is_contiguous():
        cu_seqlens = cu_seqlens.contiguous()

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o
