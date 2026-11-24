# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
import torch
import triton
import triton.language as tl

from .device import get_npu_vectorcore_num


@triton.jit
def l2norm_fwd_kernel(
    X,
    Y,
    eps,
    M,
    N: tl.constexpr,
    rows_per_prog: tl.constexpr,
    MBLOCK: tl.constexpr,
    NBLOCK: tl.constexpr,
):
    prog_id = tl.program_id(0)

    row_start = prog_id * rows_per_prog
    col_off = tl.arange(0, NBLOCK)
    col_mask = col_off < N
    for row_blk_id in range(0, rows_per_prog, MBLOCK):
        row_blk_id = tl.multiple_of(row_blk_id, MBLOCK)
        row_idx = row_start + row_blk_id + tl.arange(0, MBLOCK)
        row_off = row_idx * N
        row_mask = row_idx < M

        tot_off = row_off[:, None] + col_off[None, :]
        tot_mask = row_mask[:, None] & col_mask[None, :]

        xs = tl.load(X + tot_off, mask=tot_mask).to(tl.float32)

        square = xs * xs
        square_sum = tl.sum(square, 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)

        tl.store(Y + tot_off, xs * rsqrt, mask=tot_mask)


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None
):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    element_size = 4  # Always cast tensor elements to f32

    def get_dyn_size(D):
        size_of_x = D * element_size
        size_of_square = D * element_size
        size_of_rsqrt = D * element_size
        size_of_y = D * element_size
        return size_of_x + size_of_square + size_of_rsqrt + size_of_y

    ub_size = 196608
    core_num = get_npu_vectorcore_num()

    if T <= core_num:
        num_progs = T
        rows_per_prog = 1
        MBLOCK = 1
    else:
        num_progs = core_num
        rows_per_prog = triton.cdiv(T, core_num)
        MBLOCK = min(ub_size // get_dyn_size(BD), rows_per_prog)

    grid = (num_progs,)
    l2norm_fwd_kernel[grid](
        X=x,
        Y=y,
        eps=eps,
        M=T,
        N=D,
        rows_per_prog=rows_per_prog,
        MBLOCK=MBLOCK,
        NBLOCK=BD,
    )

    return y.view(x_shape_og)
