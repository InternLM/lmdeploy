# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


def get_autotune_config():
    """get autotune config."""
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 128
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 16,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 128
            },
            num_stages=4,
            num_warps=4),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'K'],
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
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_lar: tl.constexpr,
    stride_lak: tl.constexpr,
    stride_lbr: tl.constexpr,
    stride_lbn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """fused lora kernel."""
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    M = tl.load(seq_lens_ptr + bid)
    if M <= 0:
        return

    seq_start = tl.load(seq_start_ptr + bid)
    adapter_id = tl.load(adapter_ids_ptr + bid)
    rank_start = tl.load(rank_start_ptr + adapter_id)
    rank = tl.load(ranks_ptr + adapter_id)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    GROUP_SIZE_M: tl.constexpr = 1
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= M:
        return

    offs_m = (seq_start + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    mask_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < M
    if rank == 0:
        offs_cm = offs_m
        offs_cn = offs_n
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
            None, :]
        c_mask = mask_cm[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, 0, mask=c_mask)
        return

    offs_am = (seq_start +
               (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M)
    offs_r = rank_start + tl.arange(0, BLOCK_SIZE_R) % rank
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    la_ptrs = lora_a_ptr + (offs_k[:, None] * stride_lak +
                            offs_r[None, :] * stride_lar)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_R), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs,
                    mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        la = tl.load(la_ptrs,
                     mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                     other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, la)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        la_ptrs += BLOCK_SIZE_K * stride_lak
    ar = accumulator.to(lora_b_ptr.dtype.element_ty)

    offs_lbn = offs_n % N
    lb_ptrs = lora_b_ptr + (offs_r[:, None] * stride_lbr +
                            offs_lbn * stride_lbn)
    lb = tl.load(lb_ptrs, mask=tl.arange(0, BLOCK_SIZE_R)[:, None] < rank)

    c = tl.dot(ar, lb)

    scaling = tl.load(scaling_ptr + adapter_id)
    c *= scaling

    c = c.to(c_ptr.dtype.element_ty)
    offs_cm = offs_m
    offs_cn = offs_n
    c_ptrs = c_ptr + stride_cm * offs_cm[:,
                                         None] + stride_cn * offs_cn[None, :]
    c_mask = mask_cm[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def fused_lora(input: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor,
               scaling: torch.LongTensor, rank_start: torch.LongTensor,
               ranks: torch.LongTensor, seq_start: torch.LongTensor,
               seq_lens: torch.LongTensor, adapter_ids: torch.LongTensor,
               max_rank: int, max_seqlen: int):
    """fused lora."""

    def grid(META):
        ret = ((triton.cdiv(max_seqlen, META['BLOCK_SIZE_M']) *
                triton.cdiv(N, META['BLOCK_SIZE_N'])), batch_size)
        return ret

    assert input.dim() == 2
    batch_size = seq_lens.numel()
    M, K = input.shape
    N = lora_b.size(1)

    output = input.new_empty((M, N))

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
    )

    return output
