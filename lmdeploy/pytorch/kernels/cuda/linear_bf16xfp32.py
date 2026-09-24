# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .utils import is_cuda

# One fp32 weight is represented as ``w_hi + w_lo / _SPLIT_SCALE`` with both
# halves in bf16. ``_SPLIT_SCALE`` is a power of two, so the scaling is exact
# in bf16 (only the exponent moves) and the pair carries ~16 mantissa bits.
_SPLIT_SCALE = tl.constexpr(256.0)
_SPLIT_SCALE_INV = tl.constexpr(1.0 / 256.0)

# ``tl.dot`` requires every dimension to be at least 16.
_MIN_BLOCK_N = 16
_MAX_BLOCK_N = 128

# Split-K factors and the M threshold separating them. Values come from a sweep
# over the DSV4 HC shapes on H20: 16 is the flat optimum for a busy M grid,
# while a small M still benefits from splitting deeper.
_SPLIT_K_SMALL_M = 32
_SPLIT_K_LARGE_M = 16
_SMALL_M_LIMIT = 1024

# Below this row count the fp32 path still wins, so the kernel declines the job.
# Measured on H20 against `F.linear(x.float(), weight)` for the HC shapes
# (N=24/4, K=28672): the kernel is 1.25-1.45x slower at m<=256, 1.15-1.27x
# faster at m=512 and 3.2-4.4x faster at m=4096. The deficit at small m is
# launcher overhead (a second kernel launch for the split-K reduction), not
# GPU work, so this gate is deliberately conservative.
_MIN_M = 512


def get_cuda_autotune_config() -> list[triton.Config]:
    """Autotune configs.

    ``N`` is small enough that a single ``BLOCK_N`` covers it, so only
    ``BLOCK_M``/``BLOCK_K`` are tuned. ``BLOCK_M`` spans decode (16) to prefill
    (256); ``num_stages`` matters because the K loop is long and the per-tile
    compute is far too small to hide memory latency on its own.
    """
    return [
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
    ]


@triton.autotune(configs=get_cuda_autotune_config(), key=['N', 'K', 'SPLIT_K', 'M_HINT'])
@triton.jit(do_not_specialize=['M', 'M_HINT'])
def _linear_bf16xfp32_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    M_HINT,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
) -> None:
    """``C = A @ B.T`` with bf16 ``A`` against an fp32 ``B``.

    ``B`` is split into two bf16 halves per K tile, so both dots run on bf16
    tensor cores while the accumulator stays fp32. ``N`` is small enough that a
    single ``BLOCK_N`` covers it, hence a flat 1D M grid plus a K-split axis.

    ``M_HINT`` is never read; it only feeds the autotune key so that decode and
    prefill do not share one config. ``M`` itself changes every step and must
    stay out of the key.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc_hi = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_lo = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Splits beyond the K range simply run zero iterations and store zeros,
    # which the partial reduction absorbs.
    num_iter = tl.cdiv(tl.maximum(K - pid_k * BLOCK_K, 0), SPLIT_K * BLOCK_K)
    for _ in range(0, num_iter):
        k_mask = offs_k < K
        # ``other=0.0`` is required: K is the reduction axis, and the split
        # would amplify any garbage in ``w - w_hi`` into NaN.
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        w_hi = w.to(tl.bfloat16)
        w_lo = ((w - w_hi.to(tl.float32)) * _SPLIT_SCALE).to(tl.bfloat16)
        acc_hi = tl.dot(a, w_hi, acc_hi)
        acc_lo = tl.dot(a, w_lo, acc_lo)
        offs_k += SPLIT_K * BLOCK_K
        a_ptrs += SPLIT_K * BLOCK_K * stride_ak
        b_ptrs += SPLIT_K * BLOCK_K * stride_bk

    c = acc_hi + acc_lo * _SPLIT_SCALE_INV
    c_ptrs = C + pid_k * stride_ck + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=m_mask[:, None] & n_mask[None, :])


def _get_split_k(m: int) -> int:
    """Pick the K-split factor.

    Splitting K is measured to help at every size once the partial reduction is
    amortised: ``N`` is so small that a single K pass has too little compute to
    hide memory latency. Deeper splits only pay off while ``m`` is too small to
    keep the M grid busy, and past ~16 they start to lose to the extra partial
    traffic.
    """
    return _SPLIT_K_SMALL_M if m < _SMALL_M_LIMIT else _SPLIT_K_LARGE_M


def _get_m_hint(m: int) -> int:
    """Coarse M bucket feeding the autotune key.

    ``M`` changes every decode step, so it must stay out of the key, but decode
    and prefill want different ``BLOCK_M``. Bucketing is enough to separate
    them without letting the tuning cache grow per step.
    """
    if m <= 128:
        return 0
    if m <= 1024:
        return 1
    return 2


def linear_bf16xfp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Linear with bf16 activations against an fp32 weight.

    ``x`` is ``[..., K]`` bf16, ``weight`` is ``[N, K]`` fp32, the result is
    ``[..., N]`` fp32. ``weight`` is split into two bf16 halves so both dots run
    on bf16 tensor cores while the accumulator stays fp32.

    Falls back to an fp32 ``F.linear``, which reproduces the upcast-then-linear
    behaviour exactly, when the kernel cannot pay for itself: non-bf16
    activations, a non-fp32 weight, fewer than ``_MIN_M`` rows, an ``N`` too
    wide for one tile, or no CUDA.

    Args:
        x (torch.Tensor): Activation of shape ``[..., K]``.
        weight (torch.Tensor): Weight of shape ``[N, K]``.

    Returns:
        torch.Tensor: The fp32 projection of shape ``[..., N]``.
    """
    n, k = weight.size()
    # Cheap guards first: `is_cuda()` (a Triton driver probe) and the power-of-two
    # rounding are only worth paying once the kernel is actually going to run.
    if x.dtype != torch.bfloat16 or weight.dtype != torch.float32 or x.shape[-1] != k or x.numel() == 0:
        return F.linear(x.to(torch.float32), weight.to(torch.float32))

    m = x.numel() // k
    if m < _MIN_M or not x.is_cuda or not is_cuda() or triton.next_power_of_2(n) > _MAX_BLOCK_N:
        return F.linear(x.to(torch.float32), weight)

    assert weight.is_contiguous(), 'weight must be contiguous'
    x2d = x.reshape(m, k)
    x_shape = x.shape

    block_n = max(_MIN_BLOCK_N, triton.next_power_of_2(n))
    split_k = _get_split_k(m)
    if split_k > 1:
        out = torch.empty(split_k, m, n, device=x.device, dtype=torch.float32)
        stride_ck = m * n
    else:
        out = torch.empty(m, n, device=x.device, dtype=torch.float32)
        stride_ck = 0

    def grid(meta):
        return (triton.cdiv(m, meta['BLOCK_M']), split_k)

    _linear_bf16xfp32_kernel[grid](
        x2d,
        weight,
        out,
        m,
        n,
        k,
        _get_m_hint(m),
        x2d.stride(0),
        x2d.stride(1),
        weight.stride(1),
        weight.stride(0),
        stride_ck,
        out.stride(-2),
        out.stride(-1),
        BLOCK_N=block_n,
        SPLIT_K=split_k,
    )
    if split_k > 1:
        out = out.sum(dim=0)

    return out.view(*x_shape[:-1], n)
