# Copyright (c) OpenMMLab. All rights reserved.
import triton
from triton import language as tl

from .triton_utils import get_kernel_meta, wrap_jit_func


def get_cuda_autotune_config():
    return [
        # most used
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 8
            },
            num_stages=4,
            num_warps=4),
        # # other
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 256,
        #         'BLOCK_SIZE_K': 64,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=3,
        #     num_warps=8),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 64,
        #         'BLOCK_SIZE_N': 256,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 128,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 64,
        #         'BLOCK_SIZE_N': 128,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 32,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 64,
        #         'BLOCK_SIZE_N': 32,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=5,
        #     num_warps=2),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 32,
        #         'BLOCK_SIZE_N': 64,
        #         'BLOCK_SIZE_K': 32,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=5,
        #     num_warps=2),
        # # Good config for fp8 inputs.
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 256,
        #         'BLOCK_SIZE_K': 128,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=3,
        #     num_warps=8),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 256,
        #         'BLOCK_SIZE_N': 128,
        #         'BLOCK_SIZE_K': 128,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=3,
        #     num_warps=8),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 256,
        #         'BLOCK_SIZE_N': 64,
        #         'BLOCK_SIZE_K': 128,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 64,
        #         'BLOCK_SIZE_N': 256,
        #         'BLOCK_SIZE_K': 128,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 128,
        #         'BLOCK_SIZE_K': 128,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 64,
        #         'BLOCK_SIZE_K': 64,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_SIZE_M': 128,
        #         'BLOCK_SIZE_N': 32,
        #         'BLOCK_SIZE_K': 64,
        #         'GROUP_SIZE_M': 8
        #     },
        #     num_stages=4,
        #     num_warps=4),
    ]


@triton.jit
def _get_unpacked_order(offs_n, elem_per_int: tl.constexpr):
    """get unpacked order."""
    origin_order = offs_n % elem_per_int
    unpacked_order = (origin_order & 1) * 4 + origin_order // 2
    return unpacked_order


@triton.jit
def _broadcast_pack(weight, width: tl.constexpr):
    """broadcast pack."""
    broadcast_tmp = tl.arange(0, width)
    BLOCK_SIZE_K: tl.constexpr = weight.shape[0]
    BLOCK_SIZE_QN: tl.constexpr = weight.shape[1]
    BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_QN * width
    weight = tl.broadcast(weight[:, :, None], broadcast_tmp[None, None, :])
    weight = tl.reshape(weight, (BLOCK_SIZE_K, BLOCK_SIZE_N))
    return weight


@triton.jit
def _unpack_weight(weight, order):
    """unpack weight."""
    weight = _broadcast_pack(weight, 8)
    weight = weight >> (order * 4)
    # cast to float16
    immLut = (0xf0 & 0xcc) | 0xaa
    BOTTOM_MASK = 0xf
    I4s_TO_F16s_MAGIC_NUM = 0x6400
    FP16_TOP_MAGIC_NUM = 0x6400
    weight = tl.inline_asm_elementwise(
        """lop3.b32 $1, $1, $2, $3, $4;
    sub.f16x2 $1, $1, $5;
    mov.b32 {$0, _}, $1;""",
        '=h, r, n, n, n, r', [
            weight, BOTTOM_MASK, I4s_TO_F16s_MAGIC_NUM, immLut,
            FP16_TOP_MAGIC_NUM
        ],
        dtype=tl.float16,
        is_pure=False,
        pack=1)
    return weight


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M_NEXT_P2', 'N', 'K'],
)
@wrap_jit_func
@triton.jit
def awq_linear_kernel(
        a_ptr,
        qw_ptr,
        s_ptr,
        qz_ptr,
        c_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am,
        stride_ak: tl.constexpr,  #
        stride_wk: tl.constexpr,
        stride_wn: tl.constexpr,  #
        stride_sk: tl.constexpr,
        stride_sn: tl.constexpr,  #
        stride_zk: tl.constexpr,
        stride_zn: tl.constexpr,  #
        stride_cm,
        stride_ck: tl.constexpr,
        stride_cn: tl.constexpr,
        # Meta-parameters
        M_NEXT_P2: tl.constexpr,
        Q_GROUP_SIZE: tl.constexpr,
        SPLIT_K_ITERS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    ELEM_PER_INT = 8
    if Q_GROUP_SIZE > BLOCK_SIZE_K:
        GROUP_SIZE_K: tl.constexpr = BLOCK_SIZE_K
    else:
        GROUP_SIZE_K: tl.constexpr = Q_GROUP_SIZE
    K_PER_GROUP: tl.constexpr = Q_GROUP_SIZE // GROUP_SIZE_K

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    split_kid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    BLOCK_SIZE_QN: tl.constexpr = BLOCK_SIZE_N // 8
    offs_wn = pid_n * BLOCK_SIZE_QN + tl.arange(0, BLOCK_SIZE_QN)
    offs_k = tl.arange(0, GROUP_SIZE_K)
    unpacked_order = _get_unpacked_order(offs_bn, ELEM_PER_INT)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    qw_ptrs = qw_ptr + (offs_k[:, None] * stride_wk +
                        offs_wn[None, :] * stride_wn)
    s_ptrs = s_ptr + offs_bn * stride_sn
    qz_ptrs = qz_ptr + offs_wn * stride_zn

    # split k
    NUM_K_BLOCKS = K // GROUP_SIZE_K
    K_PER_SPLIT = tl.cdiv(NUM_K_BLOCKS, SPLIT_K_ITERS)
    k_start = split_kid * K_PER_SPLIT
    k_last = min(k_start + K_PER_SPLIT, NUM_K_BLOCKS)
    a_ptrs += k_start * GROUP_SIZE_K * stride_ak
    qw_ptrs += k_start * GROUP_SIZE_K * stride_wk
    qg_id = k_start // K_PER_GROUP

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    s = tl.zeros((1, BLOCK_SIZE_N), dtype=s_ptrs.dtype.element_ty)
    zs = tl.zeros((1, BLOCK_SIZE_N), dtype=s_ptrs.dtype.element_ty)

    # prefetch
    next_qw = tl.load(qw_ptrs)
    qw_ptrs += GROUP_SIZE_K * stride_wk

    for k in range(k_start, k_last):
        a = tl.load(a_ptrs)
        qw = next_qw
        if k + 1 < k_last:
            next_qw = tl.load(qw_ptrs)
        w = _unpack_weight(qw, unpacked_order)

        if k == k_start or k % K_PER_GROUP == 0:
            s = tl.load(s_ptrs + qg_id * stride_sk)[None, :]
            qz = tl.load(qz_ptrs + qg_id * stride_zk)[None, :]
            qg_id += 1
            z = _unpack_weight(qz, unpacked_order)
            zs = -z * s
        b = w * s + zs

        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block.
        a_ptrs += GROUP_SIZE_K * stride_ak
        qw_ptrs += GROUP_SIZE_K * stride_wk

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:,
                                         None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if stride_ck > 0:
        c_ptrs += split_kid * stride_ck
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def awq_linear(x, qweight, scales, qzeros):
    """awq linear."""
    M = x.size(0)
    K = qweight.size(0)
    N = scales.size(1)
    SPLIT_K_ITERS = 4
    group_size = K // scales.size(0)

    def grid(META):
        """grid."""
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) *
                triton.cdiv(N, META['BLOCK_SIZE_N']), SPLIT_K_ITERS)

    out = scales.new_empty(M, SPLIT_K_ITERS, N)
    M_NEXT_P2 = triton.next_power_of_2(M)

    kernel_meta = get_kernel_meta(x)
    awq_linear_kernel[grid](
        # Pointers to matrices
        x,
        qweight,
        scales,
        qzeros,
        out,
        # Matrix dimensions
        M,
        N,
        K,
        stride_am=x.stride(0),
        stride_ak=x.stride(1),  #
        stride_wk=qweight.stride(0),
        stride_wn=qweight.stride(1),  #
        stride_sk=scales.stride(0),
        stride_sn=scales.stride(1),  #
        stride_zk=qzeros.stride(0),
        stride_zn=qzeros.stride(1),  #
        stride_cm=out.stride(0),
        stride_ck=out.stride(1),
        stride_cn=out.stride(2),
        # Meta-parameters
        M_NEXT_P2=M_NEXT_P2,
        Q_GROUP_SIZE=group_size,
        SPLIT_K_ITERS=SPLIT_K_ITERS,
        **kernel_meta)

    return out.sum(1)
