# Copyright (c) OpenMMLab. All rights reserved.
# Ported from DeepSeek-V4 official inference code
# Source: DeepSeek-V4-Flash/inference/kernel.py (hc_split_sinkhorn_kernel + hc_split_sinkhorn)
# Original copyright (c) DeepSeek

import tilelang
import tilelang.language as T
import torch

tilelang.set_log_level('WARNING')

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP32 = 'float32'
INT32 = 'int32'


@tilelang.jit(pass_configs=pass_configs)
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    n = T.symbolic('n')
    mix_hc = (2 + hc) * hc
    threads = 64

    @T.prim_func
    def hc_split_sinkhorn_kernel_(
        mixes: T.Tensor[(n, mix_hc), FP32],
        hc_scale: T.Tensor[(3,), FP32],
        hc_base: T.Tensor[(mix_hc,), FP32],
        pre: T.Tensor[(n, hc), FP32],
        post: T.Tensor[(n, hc), FP32],
        comb: T.Tensor[(n, hc, hc), FP32],
    ):
        with T.Kernel(n, threads=threads) as i:
            mixes_shared = T.alloc_shared(mix_hc, FP32)
            comb_frag = T.alloc_fragment((hc, hc), FP32)
            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = mixes_shared[j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]

            row_sum = T.alloc_fragment(hc, FP32)
            col_sum = T.alloc_fragment(hc, FP32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc, FP32)
            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])

    return hc_split_sinkhorn_kernel_


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    b, s, _ = mixes.size()
    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)
    kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    kernel(
        mixes.view(-1, (2 + hc_mult) * hc_mult),
        hc_scale,
        hc_base,
        pre.view(-1, hc_mult),
        post.view(-1, hc_mult),
        comb.view(-1, hc_mult, hc_mult),
    )
    return pre, post, comb
