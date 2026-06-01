# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit(do_not_specialize=['NUM_TOKENS'])
def _gated_delta_preprocess_kernel(
    q,
    k,
    b,
    a,
    dt_bias,
    a_log_exp,
    init_token_mask,
    q_out,
    k_out,
    beta_out,
    g_out,
    q_stride_b,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_b,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    b_stride_b,
    b_stride_t,
    b_stride_src_h,
    b_stride_rep,
    a_stride_b,
    a_stride_t,
    a_stride_src_h,
    a_stride_rep,
    q_out_stride_b,
    q_out_stride_t,
    q_out_stride_h,
    q_out_stride_d,
    k_out_stride_b,
    k_out_stride_t,
    k_out_stride_h,
    k_out_stride_d,
    beta_out_stride_b,
    beta_out_stride_t,
    beta_out_stride_h,
    g_out_stride_b,
    g_out_stride_t,
    g_out_stride_h,
    NUM_TOKENS,
    BATCH_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_RATIO: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
    USE_INIT_TOKEN_MASK: tl.constexpr,
    APPLY_QK_L2NORM: tl.constexpr,
):
    batch_id = tl.program_id(0)
    src_head_id = tl.program_id(1)
    block_t_id = tl.program_id(2)
    offs_t = block_t_id * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)
    mask_t = offs_t < NUM_TOKENS
    mask_d = offs_d < HEAD_DIM

    if BATCH_SIZE == 1:
        q_batch_offset = 0
        k_batch_offset = 0
        b_batch_offset = 0
        a_batch_offset = 0
        q_out_batch_offset = 0
        k_out_batch_offset = 0
        beta_out_batch_offset = 0
        g_out_batch_offset = 0
    else:
        q_batch_offset = batch_id * q_stride_b
        k_batch_offset = batch_id * k_stride_b
        b_batch_offset = batch_id * b_stride_b
        a_batch_offset = batch_id * a_stride_b
        q_out_batch_offset = batch_id * q_out_stride_b
        k_out_batch_offset = batch_id * k_out_stride_b
        beta_out_batch_offset = batch_id * beta_out_stride_b
        g_out_batch_offset = batch_id * g_out_stride_b

    q_ptrs = q + q_batch_offset + offs_t[:, None] * q_stride_t + src_head_id * q_stride_h \
        + offs_d[None, :] * q_stride_d
    k_ptrs = k + k_batch_offset + offs_t[:, None] * k_stride_t + src_head_id * k_stride_h \
        + offs_d[None, :] * k_stride_d
    q_vals = tl.load(q_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    k_vals = tl.load(k_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    if APPLY_QK_L2NORM:
        q_rstd = 1.0 / tl.sqrt(tl.sum(q_vals * q_vals, 1) + EPS)
        k_rstd = 1.0 / tl.sqrt(tl.sum(k_vals * k_vals, 1) + EPS)
        q_vals = q_vals * q_rstd[:, None]
        k_vals = k_vals * k_rstd[:, None]

    reps = tl.arange(0, KV_RATIO)
    dst_heads = src_head_id * KV_RATIO + reps
    mask_t_rep = mask_t[:, None] & (reps[None, :] < KV_RATIO)
    beta = tl.load(
        b + b_batch_offset + offs_t[:, None] * b_stride_t + src_head_id * b_stride_src_h
        + reps[None, :] * b_stride_rep,
        mask=mask_t_rep,
        other=0.0,
    ).to(tl.float32)
    beta = tl.sigmoid(beta)
    a_val = tl.load(
        a + a_batch_offset + offs_t[:, None] * a_stride_t + src_head_id * a_stride_src_h
        + reps[None, :] * a_stride_rep,
        mask=mask_t_rep,
        other=0.0,
    ).to(tl.float32)
    dt = tl.load(dt_bias + dst_heads).to(tl.float32)
    a_scale = tl.load(a_log_exp + dst_heads).to(tl.float32)

    x = a_val + dt[None, :]
    softplus = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
    g_val = a_scale[None, :] * softplus
    if USE_INIT_TOKEN_MASK:
        is_init_token = tl.load(init_token_mask + offs_t, mask=mask_t, other=0)
        g_val = tl.where(is_init_token[:, None], -1.0e6, g_val)

    tl.store(
        beta_out + beta_out_batch_offset + offs_t[:, None] * beta_out_stride_t
        + dst_heads[None, :] * beta_out_stride_h,
        beta,
        mask=mask_t_rep,
    )
    tl.store(
        g_out + g_out_batch_offset + offs_t[:, None] * g_out_stride_t
        + dst_heads[None, :] * g_out_stride_h,
        g_val,
        mask=mask_t_rep,
    )

    for rep in tl.static_range(0, KV_RATIO):
        dst_head_id = src_head_id * KV_RATIO + rep

        q_out_ptrs = q_out + q_out_batch_offset + offs_t[:, None] * q_out_stride_t \
            + dst_head_id * q_out_stride_h + offs_d[None, :] * q_out_stride_d
        k_out_ptrs = k_out + k_out_batch_offset + offs_t[:, None] * k_out_stride_t \
            + dst_head_id * k_out_stride_h + offs_d[None, :] * k_out_stride_d
        tl.store(q_out_ptrs, q_vals, mask=mask_t[:, None] & mask_d[None, :])
        tl.store(k_out_ptrs, k_vals, mask=mask_t[:, None] & mask_d[None, :])


def gated_delta_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log_exp: torch.Tensor,
    kv_ratio: int,
    init_token_mask: torch.Tensor | None = None,
    apply_qk_l2norm: bool = True,
):
    """Prepare q/k/beta/g for gated-delta rule."""
    assert q.dim() == k.dim() == 4
    assert b.dim() == a.dim()
    assert b.dim() in (3, 4)
    assert q.shape == k.shape
    assert b.shape == a.shape
    batch_size, num_tokens, num_k_heads, head_dim = q.shape
    if b.dim() == 3:
        num_v_heads = b.size(2)
        b_stride_src_h = b.stride(2) * kv_ratio
        b_stride_rep = b.stride(2)
        a_stride_src_h = a.stride(2) * kv_ratio
        a_stride_rep = a.stride(2)
    else:
        assert b.size(2) == num_k_heads
        assert b.size(3) == kv_ratio
        num_v_heads = b.size(2) * b.size(3)
        b_stride_src_h = b.stride(2)
        b_stride_rep = b.stride(3)
        a_stride_src_h = a.stride(2)
        a_stride_rep = a.stride(3)
    assert num_v_heads == num_k_heads * kv_ratio
    assert dt_bias.numel() == num_v_heads
    assert a_log_exp.numel() == num_v_heads
    if init_token_mask is not None:
        assert init_token_mask.dim() == 1
        assert init_token_mask.numel() == num_tokens
        assert init_token_mask.is_cuda
    if b.dtype is torch.float16:
        if b.dim() == 4:
            beta_out = b.sigmoid().flatten(-2, -1)
            a = a.float().flatten(-2, -1)
        else:
            beta_out = b.sigmoid()
            a = a.float()
        g_out = a_log_exp * F.softplus(a + dt_bias)
        if init_token_mask is not None:
            g_out = g_out.masked_fill(init_token_mask[None, :, None], -1.0e6)
        if kv_ratio > 1:
            q = q.repeat_interleave(kv_ratio, dim=-2)
            k = k.repeat_interleave(kv_ratio, dim=-2)
        if apply_qk_l2norm:
            from fla.modules.l2norm import l2norm_fwd
            q_shape = q.shape
            k_shape = k.shape
            q, _ = l2norm_fwd(q.reshape(-1, q.shape[-1]))
            k, _ = l2norm_fwd(k.reshape(-1, k.shape[-1]))
            q = q.view(q_shape)
            k = k.view(k_shape)
        return q, k, beta_out, g_out

    q_out = torch.empty((batch_size, num_tokens, num_v_heads, head_dim), dtype=q.dtype, device=q.device)
    k_out = torch.empty_like(q_out)
    beta_out = torch.empty((batch_size, num_tokens, num_v_heads), dtype=b.dtype, device=b.device)
    g_out = torch.empty((batch_size, num_tokens, num_v_heads), dtype=torch.float32, device=a.device)

    block_d = triton.next_power_of_2(head_dim)
    block_t = 8
    grid = (batch_size, num_k_heads, triton.cdiv(num_tokens, block_t))
    _gated_delta_preprocess_kernel[grid](
        q,
        k,
        b,
        a,
        dt_bias,
        a_log_exp,
        init_token_mask if init_token_mask is not None else g_out,
        q_out,
        k_out,
        beta_out,
        g_out,
        *q.stride(),
        *k.stride(),
        b.stride(0),
        b.stride(1),
        b_stride_src_h,
        b_stride_rep,
        a.stride(0),
        a.stride(1),
        a_stride_src_h,
        a_stride_rep,
        *q_out.stride(),
        *k_out.stride(),
        *beta_out.stride(),
        *g_out.stride(),
        NUM_TOKENS=num_tokens,
        BATCH_SIZE=batch_size,
        HEAD_DIM=head_dim,
        KV_RATIO=kv_ratio,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        EPS=1e-6,
        USE_INIT_TOKEN_MASK=init_token_mask is not None,
        APPLY_QK_L2NORM=apply_qk_l2norm,
    )
    return q_out, k_out, beta_out, g_out
