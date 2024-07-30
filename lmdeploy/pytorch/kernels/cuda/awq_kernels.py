import triton
import triton.language as tl
from torch import Tensor
import torch
from .triton_utils import get_kernel_meta, wrap_jit_func


@triton.jit
def _get_unpacked_order(offs_n, elem_per_int):
    """get unpacked order."""
    origin_order = offs_n % elem_per_int
    unpacked_order = (origin_order % 2) * 4 + origin_order // 2
    return unpacked_order

@triton.jit
def _unpack_weight(weight, order):
    """unpack weight."""
    weight = weight >> (order * 4)
    weight = weight & 0xf
    return weight


@wrap_jit_func
@triton.jit
def _dequantize_weights_kernel(
    QWeight, Scales, Qzeros, Out,
    in_size: tl.constexpr,
    out_size: tl.constexpr,
    group_size: tl.constexpr,
    elem_per_int: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_zk: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_ok: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ):
    """dequantize weight kernel."""
    k_block_id = tl.program_id(0)
    n_block_id = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K) + k_block_id * BLOCK_K
    offs_n = tl.arange(0, BLOCK_N) + n_block_id * BLOCK_N
    offs_wk = offs_k
    offs_wn = offs_n // elem_per_int
    offs_sk = offs_k // group_size
    offs_sn = offs_n
    offs_zk = offs_sk
    offs_zn = offs_wn
    mask_k = offs_k < in_size
    mask_n = offs_n < out_size
    mask = mask_k[:, None] and mask_n[None, :]
    unpacked_order = _get_unpacked_order(offs_n, elem_per_int)

    qw_ptr = QWeight + offs_wk[:, None] * stride_wk + offs_wn[None, :] * stride_wn
    s_ptr = Scales + offs_sk[:, None] * stride_sk + offs_sn[None, :] * stride_sn
    qz_ptr = Qzeros + offs_zk[:, None] * stride_zk + offs_zn[None, :] * stride_zn
    
    out_dtype = Out.dtype.element_ty
    qw = tl.load(qw_ptr, mask=mask)
    s = tl.load(s_ptr, mask=mask).to(out_dtype)
    qz = tl.load(qz_ptr, mask=mask)

    # unpack w, z
    w = _unpack_weight(qw, unpacked_order)
    z = _unpack_weight(qz, unpacked_order)

    o = (w - z).to(s.dtype) * s
    o_ptr = Out + offs_k[:, None] * stride_ok + offs_n[None, :] * stride_on
    tl.store(o_ptr, o, mask=mask)


def awq_dequantize_weights(qweight: Tensor, scales: Tensor, qzeros: Tensor):
    """dequantize weights of awq."""
    in_size = qweight.size(0)
    out_size = scales.size(1)
    group_size = in_size // qzeros.size(0)
    elem_per_int = out_size // qzeros.size(1)

    output = scales.new_empty(in_size, out_size, dtype=scales.dtype)

    BLOCK_K = 128
    BLOCK_N = 64
    kernel_meta = get_kernel_meta(qweight)
    grid = (triton.cdiv(in_size, BLOCK_K), triton.cdiv(out_size, BLOCK_N))
    _dequantize_weights_kernel[grid](
        qweight, scales, qzeros, output,
        in_size, out_size, group_size, elem_per_int,
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        qzeros.stride(0),
        qzeros.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        **kernel_meta,
    )

    return output
