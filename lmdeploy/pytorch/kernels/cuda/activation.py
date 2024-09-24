# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from packaging import version

TRITON_VERSION = version.parse(triton.__version__)

if TRITON_VERSION >= version.parse('3.0.0'):

    fast_expf = tl.math.exp
else:
    tanh = tl.math.tanh
    fast_expf = tl.math.fast_expf


@triton.jit
def _silu_and_mul_kernel(
    gateup_ptr,
    out_ptr,
    N: tl.constexpr,
    stride_gum: tl.constexpr,
    stride_gun: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """silu and mul kernel."""
    m_id = tl.program_id(0)

    up_ptr = gateup_ptr + N * stride_gun

    offs_n = tl.arange(0, BLOCK_SIZE_N)
    gate_ptrs = gateup_ptr + m_id * stride_gum + offs_n * stride_gun
    up_ptrs = up_ptr + m_id * stride_gum + offs_n * stride_gun
    out_ptrs = out_ptr + m_id * stride_om + offs_n * stride_on

    for _ in range(0, N, BLOCK_SIZE_N):
        gate = tl.load(gate_ptrs).to(tl.float32)
        up = tl.load(up_ptrs).to(tl.float32)

        gate = gate / (1 + fast_expf(-gate))
        out = gate * up

        tl.store(out_ptrs, out)

        gate_ptrs += BLOCK_SIZE_N * stride_gun
        up_ptrs += BLOCK_SIZE_N * stride_gun
        out_ptrs += BLOCK_SIZE_N * stride_on


@triton.jit
def _silu_and_mul_no_align_kernel(
    gateup_ptr,
    out_ptr,
    N: tl.constexpr,
    stride_gum: tl.constexpr,
    stride_gun: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """silu and mul kernel."""
    m_id = tl.program_id(0)

    up_ptr = gateup_ptr + N * stride_gun

    offs_n = tl.arange(0, BLOCK_SIZE_N)
    gate_ptrs = gateup_ptr + m_id * stride_gum + offs_n * stride_gun
    up_ptrs = up_ptr + m_id * stride_gum + offs_n * stride_gun
    out_ptrs = out_ptr + m_id * stride_om + offs_n * stride_on

    for n in range(0, N, BLOCK_SIZE_N):
        mask = n + offs_n < N
        gate = tl.load(gate_ptrs, mask=mask).to(tl.float32)
        up = tl.load(up_ptrs, mask=mask).to(tl.float32)

        gate = gate / (1 + fast_expf(-gate))
        out = gate * up

        tl.store(out_ptrs, out, mask=mask)

        gate_ptrs += BLOCK_SIZE_N * stride_gun
        up_ptrs += BLOCK_SIZE_N * stride_gun
        out_ptrs += BLOCK_SIZE_N * stride_on


def silu_and_mul(gate_up: torch.Tensor, out: torch.Tensor = None):
    """silu and mul."""
    assert gate_up.dim() == 2

    M = gate_up.size(0)
    N = gate_up.size(-1) // 2
    if out is None:
        out_shape = (M, N)
        out = gate_up.new_empty(out_shape)

    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, 1024)
    num_warps = 4
    num_stages = 2
    grid = (M, )
    if N % BLOCK_SIZE_N == 0:
        _silu_and_mul_kernel[grid](gate_up,
                                   out,
                                   N,
                                   stride_gum=gate_up.stride(0),
                                   stride_gun=gate_up.stride(1),
                                   stride_om=out.stride(0),
                                   stride_on=out.stride(1),
                                   BLOCK_SIZE_N=BLOCK_SIZE_N,
                                   num_warps=num_warps,
                                   num_stages=num_stages)
    else:
        _silu_and_mul_no_align_kernel[grid](gate_up,
                                            out,
                                            N,
                                            stride_gum=gate_up.stride(0),
                                            stride_gun=gate_up.stride(1),
                                            stride_om=out.stride(0),
                                            stride_on=out.stride(1),
                                            BLOCK_SIZE_N=BLOCK_SIZE_N,
                                            num_warps=num_warps,
                                            num_stages=num_stages)

    return out
