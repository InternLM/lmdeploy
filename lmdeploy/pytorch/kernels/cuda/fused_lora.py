# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


def get_autotune_config():
    """Get autotune config."""
    return [
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 128
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 256,
            'BLOCK_SIZE_K': 128
        }, num_stages=4, num_warps=4),
    ]


@triton.jit
def _atomic_store(ptrs, val, mask):
    """Atomic store values."""
    dtype = ptrs.dtype.element_ty
    if (dtype == torch.float16) | (dtype == torch.float32):
        tl.atomic_add(ptrs, val, mask=mask, sem='relaxed')
    else:
        # bfloat16 does not support atomic add
        origin = tl.load(ptrs, mask=mask)
        val = val.to(origin.dtype)
        val += origin
        tl.store(ptrs, val, mask=mask)


@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'K'],
    restore_value=['c_ptr'],
)
@triton.jit
def _fused_lora_kernel(
    a_ptr,
    lora_a_ptr,
    lora_b_ptr,
    c_ptr,
    scaling_ptr,
    rank_start_ptr,
    ranks_ptr,
    seq_start_ptr,
    seq_lens_ptr,
    adapter_ids_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak: tl.constexpr,
    stride_lar: tl.constexpr,
    stride_lak: tl.constexpr,
    stride_lbr: tl.constexpr,
    stride_lbn: tl.constexpr,
    stride_cm,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    CUM: tl.constexpr,
):
    """Fused lora kernel."""
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    M = tl.load(seq_lens_ptr + bid)
    if M <= 0:
        return

    seq_start = tl.load(seq_start_ptr + bid)
    adapter_id = tl.load(adapter_ids_ptr + bid)
    rank_start = tl.load(rank_start_ptr + adapter_id)
    rank = tl.load(ranks_ptr + adapter_id)

    pid_m = pid

    if pid_m * BLOCK_SIZE_M >= M:
        return

    offs_m = (seq_start + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    mask_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < M
    offs_cm = offs_m
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_n[None, :]

    if rank == 0:
        if not CUM:
            for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
                mask_cn = (offs_n < N - n * BLOCK_SIZE_N)
                c_mask = mask_cm[:, None] * mask_cn[None, :]
                tl.store(c_ptrs, 0.0, mask=c_mask)
                c_ptrs += stride_cn * BLOCK_SIZE_N
    else:

        offs_am = (seq_start + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M)
        offs_r = rank_start + tl.arange(0, BLOCK_SIZE_R) % rank
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        la_ptrs = lora_a_ptr + (offs_k[:, None] * stride_lak + offs_r[None, :] * stride_lar)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_R), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            la = tl.load(la_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.
            accumulator = tl.dot(a, la, acc=accumulator)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            la_ptrs += BLOCK_SIZE_K * stride_lak
        ar = accumulator.to(lora_b_ptr.dtype.element_ty)

        scaling = tl.load(scaling_ptr + adapter_id).to(ar.dtype)
        ar *= scaling
        ar = tl.where(tl.arange(0, BLOCK_SIZE_R)[None, :] < rank, ar, tl.zeros_like(ar))
        lb_ptrs = lora_b_ptr + (offs_r[:, None] * stride_lbr + offs_n[None, :] * stride_lbn)

        for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
            lb = tl.load(lb_ptrs, mask=offs_n[None, :] < N - n * BLOCK_SIZE_N)
            c = tl.dot(ar, lb)

            mask_cn = (offs_n < N - n * BLOCK_SIZE_N)
            c_mask = mask_cm[:, None] * mask_cn[None, :]
            if CUM:
                _atomic_store(c_ptrs, c, mask=c_mask)
            else:
                tl.store(c_ptrs, c, mask=c_mask)
            c_ptrs += stride_cn * BLOCK_SIZE_N
            lb_ptrs += stride_lbn * BLOCK_SIZE_N


def fused_lora(input: torch.Tensor,
               lora_a: torch.Tensor,
               lora_b: torch.Tensor,
               scaling: torch.LongTensor,
               rank_start: torch.LongTensor,
               ranks: torch.LongTensor,
               seq_start: torch.LongTensor,
               seq_lens: torch.LongTensor,
               adapter_ids: torch.LongTensor,
               max_rank: int,
               max_seqlen: int,
               output: torch.Tensor = None,
               cum: bool = False):
    """Fused lora."""

    def grid(META):
        ret = ((triton.cdiv(max_seqlen, META['BLOCK_SIZE_M'])), batch_size)
        return ret

    assert input.dim() == 2
    batch_size = seq_lens.numel()
    M, K = input.shape
    N = lora_b.size(1)

    if output is None:
        output = input.new_empty((M, N))
        cum = False
    else:
        assert output.size(0) == M
        assert output.size(1) == N

    BLOCK_SIZE_R = max(16, max_rank)
    _fused_lora_kernel[grid](
        input,
        lora_a,
        lora_b,
        output,
        scaling,
        rank_start,
        ranks,
        seq_start,
        seq_lens,
        adapter_ids,
        N,
        K,
        stride_am=input.stride(0),
        stride_ak=input.stride(1),
        stride_lar=lora_a.stride(0),
        stride_lak=lora_a.stride(1),
        stride_lbr=lora_b.stride(0),
        stride_lbn=lora_b.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        BLOCK_SIZE_R=BLOCK_SIZE_R,
        CUM=cum,
    )

    return output
