from __future__ import annotations

import torch

_SM90_FP8_FUSED_SILU_BLOCK = 128
_SM90_BF16_FUSED_SILU_BLOCK = 64


def fused_silu_block(weight_type: str) -> int:
    return {
        'fp8_e4m3': _SM90_FP8_FUSED_SILU_BLOCK,
        'bf16': _SM90_BF16_FUSED_SILU_BLOCK,
    }[weight_type]


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise ValueError(f'shape_mismatch_{tuple(actual.shape)}_{tuple(expected.shape)}')
    a = actual.detach().float()
    e = expected.detach().float()
    abs_diff = (a - e).abs()
    denom = e.abs().clamp_min(1e-6)
    rel = abs_diff / denom
    return {
        'max_abs': float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        'mean_abs': float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        'max_rel': float(rel.max().item()) if rel.numel() else 0.0,
        'mean_rel': float(rel.mean().item()) if rel.numel() else 0.0,
    }


def dense_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # x: [M, K], weight: [K, N] (MN-major storage as in testbed_v3)
    return x @ weight


def block_pack_w1w3(w1: torch.Tensor, w3: torch.Tensor, block: int) -> torch.Tensor:
    """Pack [g0:block|u0:block|…] along the output dim (matches builder
    `_block_pack_w1w3`)."""
    n = w1.shape[-1]
    if n % block != 0:
        raise ValueError(f'inter_{n}_not_divisible_by_block_{block}')
    shape = w1.shape[:-1]
    w1b = w1.reshape(*shape, n // block, block)
    w3b = w3.reshape(*shape, n // block, block)
    return torch.stack([w1b, w3b], dim=-2).reshape(*shape, n * 2).contiguous()


def apply_block_fused_silu(c: torch.Tensor, block: int) -> torch.Tensor:
    """Apply silu(gate)*up on block-packed GEMM output ``[M, 2*inter]`` → ``[M,
    inter]``."""
    m, n = c.shape
    if n % (2 * block) != 0:
        raise ValueError(f'packed_n_{n}_not_aligned_to_{2 * block}')
    inter = n // 2
    out = torch.empty((m, inter), device=c.device, dtype=torch.float32)
    cf = c.float()
    for b in range(inter // block):
        g0 = b * 2 * block
        u0 = g0 + block
        o0 = b * block
        gate = cf[:, g0:g0 + block]
        up = cf[:, u0:u0 + block]
        out[:, o0:o0 + block] = gate * torch.sigmoid(gate) * up
    return out.to(dtype=c.dtype)


def quantize_symm_row_fp8(
        src: torch.Tensor,
        group_size: int = _SM90_FP8_FUSED_SILU_BLOCK,
        qmax: float = 448.0):
    """Match QuantizeSymm: per-row group-absmax → e4m3 + float scales [ceil(N/gs), M].

    Returns (fp8_tensor as uint8 view for storage, scales float32, dequant bf16).
    """
    m, n = src.shape
    xf = src.float()
    s_dim = (n + group_size - 1) // group_size
    scales = torch.empty((s_dim, m), device=src.device, dtype=torch.float32)
    out_f = torch.empty_like(xf)
    for g in range(s_dim):
        c0 = g * group_size
        c1 = min(c0 + group_size, n)
        block = xf[:, c0:c1]
        amax = block.abs().amax(dim=1).clamp_min(1e-8)
        scales[g] = amax / qmax
        inv = qmax / amax
        out_f[:, c0:c1] = block * inv[:, None]
    # Round-trip through CUDA fp8 to match kernel cast.
    fp8 = out_f.to(torch.float8_e4m3fn)
    dequant = (fp8.float() * scales.t().repeat_interleave(group_size, dim=1)[:, :n])
    return fp8, scales, dequant.to(dtype=torch.bfloat16)


def moe_reference(
    x: torch.Tensor,
    expert_weights: list[torch.Tensor],
    f2n: torch.Tensor | None,
    offsets: torch.Tensor,
    scales: torch.Tensor,
    en2f: torch.Tensor,
    experts_per_token: int,
    combine_experts: bool,
) -> torch.Tensor:
    """Mirror testbed_v3 GetReference MoE path using gather/GEMM/scatter.

    Layouts match testbed_v3 Route()/GetReference():
    - f2n: [tokens * experts_per_token] packed expert-major token indices.
      If None, ``x`` is already expert-packed (MoE down / w2).
    - offsets: [expert_num + 1] exclusive prefix counts
    - scales: [experts_per_token * batch] with index e * batch + tok
    - en2f: [experts_per_token * batch] maps (e, tok) -> packed slot
    - expert_weights[e]: [K, N]
    """
    n = expert_weights[0].shape[1]
    device = x.device
    dtype = x.dtype
    if f2n is None:
        xe = x
        token_count = int(x.shape[0])
        m = token_count // experts_per_token
    else:
        m, k = x.shape
        token_count = int(f2n.numel())
        xe = x[f2n.long()]
    de = torch.empty((token_count, n), device=device, dtype=dtype)
    h_offsets = offsets.detach().cpu().tolist()
    for e, w in enumerate(expert_weights):
        base = int(h_offsets[e])
        end = int(h_offsets[e + 1])
        if end > base:
            de[base:end] = dense_gemm(xe[base:end], w)
    if not combine_experts:
        return de
    # Match invokeMoeCombine / MoeReduceKernel: accumulate in float, then cast.
    out_f = torch.zeros((m, n), device=device, dtype=torch.float32)
    for e in range(experts_per_token):
        for tok in range(m):
            packed = int(en2f[e * m + tok].item())
            out_f[tok] += de[packed].float() * float(scales[e * m + tok].item())
    return out_f.to(dtype)
