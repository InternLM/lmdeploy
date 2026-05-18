# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .blocked_gemm_fp8 import quant_fp8
from .v4_fp4_fused_moe import fused_moe_v4_fp4_kernel_launcher


def m_grouped_fp8_fp4_gemm_nt_contiguous(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    *,
    recipe: tuple[int, int, int] = (1, 1, 32),
    disable_ue8m0_cast: bool = True,
):
    """Contiguous grouped GEMM: FP8 activations x FP4 weights.

    API-compatible with ``deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous``
    so that SM100 can swap the function body for the native DeepGEMM path.

    Args:
        a: (fp8_data [M, K], scale [M, K//gran_k]).
        b: (fp4_data [E, N, K//2], scale [E, N, K//32]).
        d: Output [M, N] bf16.
        grouped_layout: [M] int32, expert id per row (-1 = padding).
        recipe: (gran_mn_a, gran_mn_b, gran_k). Ignored on SM90.
        disable_ue8m0_cast: Ignored on SM90 (always uses e8m0fnu scale).
    """
    A, A_scale = a
    B, B_scale = b
    M = A.size(0)
    E = B.size(0)

    # Build exp_start / exp_end from grouped_layout.
    # Input rows are already sorted by expert (after ep_scatter).
    expert_counts = torch.zeros(E, dtype=torch.int64, device=A.device)
    valid = grouped_layout >= 0
    valid_layout = grouped_layout[valid]
    expert_counts.scatter_add_(0, valid_layout.long(), torch.ones_like(valid_layout, dtype=torch.int64))
    exp_end = expert_counts.cumsum(0)
    exp_start = exp_end - expert_counts

    # Tokens are contiguous per expert; sorted_idx is just arange.
    sorted_idx = torch.arange(M, device=A.device, dtype=torch.int64)

    fused_moe_v4_fp4_kernel_launcher(
        A,
        A_scale,
        B,
        B_scale,
        d,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        top_k=1,
        num_tokens=M,
        expert_offset=0,
        reindex_a=False,
        reindex_c=False,
    )


def m_grouped_fp8_fp4_gemm_nt_masked(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    recipe: tuple[int, int, int] = (1, 1, 32),
    disable_ue8m0_cast: bool = True,
):
    """Masked grouped GEMM: FP8 activations x FP4 weights.

    API-compatible with ``deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked``.

    Args:
        a: (fp8_data [E, max_m, K], scale [E, max_m, K//gran_k]).
        b: (fp4_data [E, N, K//2], scale [E, N, K//32]).
        d: Output [E, max_m, N] bf16.
        masked_m: [E] int, valid rows per expert.
        expected_m: Scheduling hint (max m per group).
        recipe: (gran_mn_a, gran_mn_b, gran_k). Ignored on SM90.
        disable_ue8m0_cast: Ignored on SM90.
    """
    A, A_scale = a
    B, B_scale = b
    E = B.size(0)
    max_m = A.size(1)

    # Flatten 3D -> 2D for kernel.
    A = A.reshape(E * max_m, A.size(-1))
    A_scale = A_scale.reshape(E * max_m, A_scale.size(-1))
    d_flat = d.reshape(E * max_m, d.size(-1))

    # exp_start[e] = e * max_m, exp_end[e] = e * max_m + masked_m[e]
    exp_start = torch.arange(E, device=A.device, dtype=torch.int64) * max_m
    exp_end = exp_start + masked_m.to(torch.int64)

    sorted_idx = torch.arange(E * max_m, device=A.device, dtype=torch.int64)

    fused_moe_v4_fp4_kernel_launcher(
        A,
        A_scale,
        B,
        B_scale,
        d_flat,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        top_k=1,
        num_tokens=E * max_m,
        expert_offset=0,
        reindex_a=False,
        reindex_c=False,
    )


def _v4_swiglu(gate_up: torch.Tensor, swiglu_limit: float = 0.0) -> torch.Tensor:
    """SiLU+mul for V4 with optional clamping."""
    hidden = gate_up.size(-1) // 2
    gate = gate_up[..., :hidden].float()
    up = gate_up[..., hidden:].float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    return (torch.nn.functional.silu(gate) * up).to(gate_up.dtype)


def silu_and_mul_moe_ep_v4(
    gate_up: torch.Tensor,
    swiglu_limit: float = 0.0,
    group_size: int = 128,
    scale_fmt: str = 'ue8m0',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SiLU+mul(+clamp) + FP8 requantization for V4 EP MoE.

    Padding rows (in the [E, max_m, ...] layout) are processed alongside
    valid rows — the downstream masked GEMM ignores them anyway.

    Args:
        gate_up: [total_M, 2*N] bf16 (contiguous) or [E, max_m, 2*N] (masked).
        swiglu_limit: Clamp value for V4 SwiGLU (0 = no clamp).
        group_size: Quantization group size.
        scale_fmt: Scale format for quant_fp8.

    Returns:
        (bf16_act, fp8_quant, scale) where bf16_act is the SiLU output,
        fp8_quant and scale are the requantized tensors for the next GEMM.
    """
    act = _v4_swiglu(gate_up, swiglu_limit)
    shape = act.shape
    act_flat = act.reshape(-1, shape[-1])
    act_flat, act_scale = quant_fp8(act_flat, group_size, dtype=torch.float8_e4m3fn, scale_fmt=scale_fmt)
    return act, act_flat.reshape(shape), act_scale.reshape(shape[:-1] + act_scale.shape[-1:])


def fused_moe_v4_fp4_ep_normal(
    recv_x: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    recv_tokens_per_expert: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    num_local_experts: int,
    expert_offset: int = 0,
    swiglu_limit: float = 0.0,
    group_size: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Prefill EP: contiguous GEMM x2 on dispatcher-permuted tokens.

    The caller (DeepEPTokenDispatcher) has already all-to-all dispatched
    and permuted tokens by expert. The returned ``recv_x`` is sorted
    expert-contiguously, so we can directly quantize + GEMM without
    ep_scatter/ep_gather.  The caller's combine() will handle unpermute
    + weight multiply + all-to-all combine.

    Args:
        recv_x: [recv_tokens, K] BF16, already permuted by expert.
        recv_topk_idx: [recv_tokens, top_k] global expert IDs (unused here,
            permute was done by dispatcher).
        recv_topk_weights: [recv_tokens, top_k] float32 (unused here).
        recv_tokens_per_expert: [num_local_experts] int64, token count per
            local expert after permute.
        w1/w1_scale: FP4 gate_up weights [num_local_experts, ...].
        w2/w2_scale: FP4 down weights [num_local_experts, ...].
        num_local_experts: Number of experts on this rank.
        expert_offset: First global expert ID on this rank.
        swiglu_limit: Clamp value for V4 SwiGLU.
        group_size: Quantization group size.
        out_dtype: Output dtype.
    """
    all_tokens = recv_tokens_per_expert.sum().item()
    if all_tokens <= 0:
        return recv_x.new_empty(0, w2.size(1), dtype=out_dtype)

    M, K = recv_x.size()
    N = w1.size(1)

    # Build grouped_layout from recv_tokens_per_expert.
    # Tokens are already contiguous per expert after dispatcher permute.
    expert_ids = torch.arange(num_local_experts, device=recv_x.device, dtype=torch.int32)
    m_indices = torch.repeat_interleave(expert_ids, recv_tokens_per_expert.to(torch.int64))

    # --- Quantize BF16 tokens to FP8 ---
    input_quant, input_scale = quant_fp8(recv_x, group_size, dtype=torch.float8_e4m3fn, scale_fmt='ue8m0')

    # --- Gate-up GEMM (contiguous) ---
    gateup_output = recv_x.new_empty((all_tokens, N), dtype=out_dtype)
    m_grouped_fp8_fp4_gemm_nt_contiguous(
        (input_quant, input_scale),
        (w1, w1_scale),
        gateup_output,
        m_indices,
    )
    del input_quant, input_scale

    # --- SiLU+mul + requantize ---
    _, act_quant, act_scale = silu_and_mul_moe_ep_v4(
        gateup_output,
        swiglu_limit=swiglu_limit,
        group_size=group_size,
    )
    del gateup_output

    # --- Down GEMM (contiguous) ---
    down_output = recv_x.new_empty((all_tokens, w2.size(1)), dtype=out_dtype)
    m_grouped_fp8_fp4_gemm_nt_contiguous(
        (act_quant, act_scale),
        (w2, w2_scale),
        down_output,
        m_indices,
    )
    del act_quant, act_scale

    return down_output


def fused_moe_v4_fp4_ep_low_latency(
    recv_hidden: tuple[torch.Tensor, torch.Tensor],
    masked_m: torch.Tensor,
    expected_m: int,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    swiglu_limit: float = 0.0,
    group_size: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode EP: masked GEMM x2, output BF16 for low_latency_combine.

    Takes the FP8 (data, scale) tuple from DeepEP low_latency_dispatch
    and runs two masked grouped GEMMs with V4 SwiGLU in between.

    Args:
        recv_hidden: (fp8_data [E, max_m, K], scale [E, max_m, K//128]).
        masked_m: [E] int, valid rows per expert.
        expected_m: Scheduling hint.
        w1/w1_scale: FP4 gate_up weights [E, ...].
        w2/w2_scale: FP4 down weights [E, ...].
        swiglu_limit: Clamp value for V4 SwiGLU.
        group_size: Quantization group size.
        out_dtype: Output dtype.

    Returns:
        [E, max_m, hidden_dim] bf16 tensor for low_latency_combine.
    """
    A, A_scale = recv_hidden
    E, max_m = A.shape[:2]
    hidden_dim = w2.size(1)

    # --- Gate-up GEMM (masked) ---
    gateup_output = torch.empty((E, max_m, w1.size(1)), dtype=out_dtype, device=A.device)
    m_grouped_fp8_fp4_gemm_nt_masked(
        (A, A_scale),
        (w1, w1_scale),
        gateup_output,
        masked_m,
        expected_m,
    )

    # --- SiLU+mul + requantize ---
    _, act_quant, act_scale = silu_and_mul_moe_ep_v4(
        gateup_output,
        swiglu_limit=swiglu_limit,
        group_size=group_size,
    )
    del gateup_output

    # --- Down GEMM (masked) ---
    down_output = torch.empty((E, max_m, hidden_dim), dtype=out_dtype, device=A.device)
    m_grouped_fp8_fp4_gemm_nt_masked(
        (act_quant, act_scale),
        (w2, w2_scale),
        down_output,
        masked_m,
        expected_m,
    )
    return down_output
