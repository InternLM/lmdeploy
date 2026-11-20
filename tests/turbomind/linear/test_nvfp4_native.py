from __future__ import annotations

import math

import pytest
import torch

from lmdeploy import turbomind

if not turbomind.is_available():
    pytest.skip('TurboMind is not built', allow_module_level=True)

from lmdeploy.turbomind.weight_format import NVFP4Format

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _quantize_nvfp4(w_dense: torch.Tensor):
    """Quantize a [N, K] dense weight to NVFP4 (raw checkpoint layout).

    Follows the compressed-tensors nvfp4-pack-quantized convention: the
    per-tensor ``weight_global_scale`` is the divisor ``2688 / amax_w`` and
    the e4m3 block scales are ``448 * amax_g / amax_w`` (<= 448), so the
    reconstruction is ``codes * scales / global_scale == codes * amax_g/6``.
    """
    n, k = w_dense.shape
    groups = w_dense.view(n, k // 16, 16)
    amax_w = w_dense.abs().amax().clamp_min(1e-8)
    scale = (groups.abs().amax(dim=-1, keepdim=True) * (448.0 / amax_w)).clamp_max(448.0)
    scale = scale.to(torch.float8_e4m3fn).float()  # quantize the scale itself
    # codes = round(w * global_scale / scale): the fp4 grid covers [-6, 6]
    # regardless of the weight amplitude. Dividing by the raw scale alone
    # (the historical bug) collapses everything but amax_w ~ 2688 weights
    # onto the zero code and makes the reconstruction vacuous.
    scaled = groups * ((2688.0 / amax_w) / scale)

    grid = torch.tensor(_E2M1_VALUES, device=w_dense.device)
    mag = torch.argmin((scaled.unsqueeze(-1) - grid).abs(), dim=-1)  # [N, K/16, 16]
    sign = (scaled < 0).to(torch.uint8)
    codes = (mag.to(torch.uint8) & 0x7) | (sign << 3)  # [N, K/16, 16]

    codes = codes.view(n, k)
    # Two codes per byte: even k in the low nibble, odd k in the high nibble.
    bits = codes[..., 0::2] | (codes[..., 1::2] << 4)  # [N, K/2] uint8

    global_scale = (2688.0 / amax_w).float()
    return bits, scale.view(n, k // 16), global_scale


def _dequant_nvfp4(bits: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """First-principles fp32 reconstruction: codes * scales / global_scale."""
    n, k2 = bits.shape
    lo = (bits & 0xF).long()
    hi = (bits >> 4).long()
    codes = torch.stack((lo, hi), dim=-1).view(n, k2 * 2)  # [N, K]
    mag = codes & 0x7
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    grid = torch.tensor(_E2M1_VALUES)
    values = grid[mag] * sign  # [N, K]
    scales = scale.repeat_interleave(16, dim=-1)  # [N, K]
    return (values * scales / global_scale).t().contiguous()  # [K, N]


def test_split_qkv_passes_global_scale_through():
    """Per-tensor global_scale bypasses output-dim transforms (split/fuse)
    and lands in every output Linear: the fold path used to pop the tensor,
    the native bundle keeps it, so the fused GDN pipeline must carry it."""
    from lmdeploy.turbomind.builders.deltanet import split_qkv
    from lmdeploy.turbomind.builders.linear import Linear

    fmt = NVFP4Format()
    tensors = fmt.finalize({
        'weight': torch.zeros((64, 8), dtype=torch.uint8),
        'scales': torch.full((64 // 16, 8), 1.0, dtype=torch.float8_e4m3fn),
        'global_scale': torch.tensor([2.0], dtype=torch.float32),
    })
    linear = Linear(tensors=tensors, weight_format=fmt)

    q, k, v = split_qkv(linear, num_k_heads=2, num_v_heads=4)
    assert q.tensors['weight'].shape == (64, 2)
    for part in (q, k, v):
        assert 'global_scale' in part.tensors
        assert torch.equal(part.tensors['global_scale'], tensors['global_scale'])


def test_w1w3_merge_rejects_mismatched_global_scales():
    """The merged w1w3 layout has one global-scale slot and the merge keeps
    only the first input's per-tensor tensors: separately-quantized gate/up
    with differing global scales must raise loudly instead of silently applying
    the gate's scale to the whole fused output."""
    from lmdeploy.turbomind.builders.ffn import _block_pack_w1w3, _check_w1w3_global_scales
    from lmdeploy.turbomind.builders.linear import Linear

    def mk(gs):
        fmt = NVFP4Format()
        tensors = fmt.finalize({
            'weight': torch.zeros((64, 8), dtype=torch.uint8),
            'scales': torch.full((64 // 16, 8), 1.0, dtype=torch.float8_e4m3fn),
            'global_scale': torch.tensor([gs], dtype=torch.float32),
        })
        return Linear(tensors=tensors, weight_format=fmt)

    with pytest.raises(ValueError, match='global scale'):
        _check_w1w3_global_scales(mk(2.0), mk(100.0))

    # Equal scales (fused checkpoints like QUASAR share one gs) still merge.
    gate, up = mk(2.0), mk(2.0)
    _check_w1w3_global_scales(gate, up)
    fused = _block_pack_w1w3(gate, up, groups=1)
    assert 'global_scale' in fused.tensors
    assert torch.equal(fused.tensors['global_scale'], gate.tensors['global_scale'])

    # Mixed presence (exactly one side carries gs) rejects loudly: the merged
    # layout has one gs slot, so the bare half would silently lose or gain the
    # scalar — the same all-or-none convention LinkLinearExperts enforces.
    one_sided = mk(3.0)
    bare = Linear(tensors={k: v for k, v in one_sided.tensors.items() if k != 'global_scale'},
                  weight_format=one_sided.weight_format)
    with pytest.raises(ValueError, match='exactly one of gate/up'):
        _check_w1w3_global_scales(one_sided, bare)
    with pytest.raises(ValueError, match='exactly one of gate/up'):
        _check_w1w3_global_scales(bare, one_sided)

    # Both absent: nothing to conflict, plain merge.
    _check_w1w3_global_scales(bare, Linear(tensors={}, weight_format=one_sided.weight_format))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_nvfp4_native_forward_matches_reference():
    """End-to-end: raw NVFP4 checkpoint -> native GEMM vs first-principles
    reference.

    This is the load-bearing proof that the kernel consumes the per-tensor
    global scale on the SM80/SM86 lane: the raw scales stay e4m3, the global
    scale rides through LinearWeight and the epilogue multiplies the output
    by it. A kernel that ignored the scale would be off by `global_scale`
    (a large divisor) and fail hard.
    """
    from lmdeploy.turbomind.linear import get_linear

    torch.manual_seed(0)
    fmt = NVFP4Format()
    for K, N, M in ((512, 128, 64), (2048, 256, 8)):
        w_dense = torch.randn(N, K) * (0.1 / math.sqrt(K))
        bits, scale, global_scale = _quantize_nvfp4(w_dense)

        finalized = fmt.finalize({
            'weight': bits,
            'scales': scale.to(torch.float8_e4m3fn),
            'global_scale': global_scale.reshape(1),
        })
        assert torch.equal(finalized['scales'].view(torch.uint8), scale.to(torch.float8_e4m3fn).view(torch.uint8))
        epilogue_scale = finalized['global_scale']
        assert torch.allclose(epilogue_scale, torch.reciprocal(global_scale.reshape(1).float()), rtol=0, atol=0)

        linear = get_linear()
        with linear._activate():
            plan = linear.get_weight_plan(weight_format=fmt, dtype=torch.float16)
            # prepare_weight finalizes internally: the raw checkpoint
            # global scale (a divisor) goes in as-is.
            weight = linear.prepare_weight(
                bits.cuda(),
                plan=plan,
                scales=scale.to(torch.float8_e4m3fn).cuda(),
                global_scale=global_scale.reshape(1).cuda(),
            )
            x = torch.randn(M, K, device='cuda', dtype=torch.float16)
            exec_plan, out, _ = linear.tune(x, weight)

        assert out.dtype == torch.float16

        # Reference 1: exact fp32 reconstruction of the raw bundle.
        w_ref = _dequant_nvfp4(bits, scale, global_scale)  # [K, N]
        ref = x.float() @ w_ref.cuda().float()
        err = (out.float() - ref).abs().max().item()
        assert err < 1e-2, f'K={K} N={N} M={M}: max abs err {err}'

        # Reference 2: the retired fold emulation (fp16 scales/gs, no
        # epilogue multiply) must agree with the native path up to the fp16
        # subnormal noise the fold suffered -- the superseded fd4d1cb0 math.
        folded_scales = (scale / global_scale).to(torch.float16).float()
        codes = torch.stack(((bits & 0xF).long(), (bits >> 4).long()), dim=-1).view(N, K)
        grid = torch.tensor(_E2M1_VALUES)
        values_fold = grid[codes & 0x7] * torch.where((codes & 0x8) != 0, -1.0, 1.0)
        w_fold = (values_fold * folded_scales.repeat_interleave(16, dim=-1)).t().contiguous()
        ref_fold = x.float() @ w_fold.cuda().float()
        err_fold = (out.float() - ref_fold).abs().max().item()
        assert err_fold < 1e-2, f'K={K} N={N} M={M}: fold-emulation abs err {err_fold}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_nvfp4_grouped_forward_matches_reference():
    """Grouped (MoE) proof: per-expert NVFP4 weights with DISTINCT per-tensor
    global scales routed through an uneven offsets layout must reconstruct each
    expert exactly.

    Constant/shared-scale kernel bugs are detectable only because the amplitudes (and hence gs) differ pairwise by 10x.
    """
    from lmdeploy.turbomind.linear import get_linear

    torch.manual_seed(0)
    E, K, N, T = 3, 512, 128, 24
    offsets = torch.tensor([0, 8, 18, 24], dtype=torch.int32, device='cuda')
    amplitudes = [0.02, 0.2, 2.0]

    fmt = NVFP4Format()
    bits_l, scales_l, gs_l, refs_l = [], [], [], []
    for amp in amplitudes:
        w = torch.randn(N, K) * (amp / math.sqrt(K))
        bits, scale, gs = _quantize_nvfp4(w)
        assert torch.isfinite(gs) and gs > 0
        bits_l.append(bits.cuda())
        scales_l.append(scale.to(torch.float8_e4m3fn).cuda())
        gs_l.append(gs.reshape(1).cuda())
        refs_l.append(_dequant_nvfp4(bits, scale, gs))  # [K, N] fp32
    for i in range(E):
        for j in range(i + 1, E):
            assert not torch.allclose(gs_l[i].cpu(), gs_l[j].cpu()), \
                f'global scales must be pairwise distinct (experts {i},{j})'

    linear = get_linear()
    with linear._activate():
        plan = linear.get_weight_plan(weight_format=fmt, dtype=torch.float16, grouped=True)
        weight = linear.prepare_weight(bits_l, plan=plan, scales=scales_l, global_scale=gs_l)

        x = torch.randn(T, K, device='cuda', dtype=torch.float16)
        exec_plan, out, _ = linear.tune(x, weight, offsets=offsets)
        assert out.dtype == torch.float16

    # Reference: each expert's segment against its own dequantized weight.
    for e in range(E):
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        if lo == hi:
            continue
        ref = x[lo:hi].float() @ refs_l[e].cuda().float()
        err = (out[lo:hi].float() - ref).abs().max().item()
        assert err < 1e-2, f'expert {e}: max abs err {err}'

    # Sensitivity control: with a shared (expert-0) scale the error must be
    # large -- otherwise the test cannot detect constant-scale bugs.
    wrong = [
        _dequant_nvfp4(
            bits_l[e].cpu(),
            scales_l[e].cpu().float(),
            gs_l[0].cpu().reshape(1),
        )
        for e in range(E)
    ]
    wrong_err = max(
        (x[int(offsets[e]):int(offsets[e + 1])].float() @ wrong[e].cuda().float()
         - out[int(offsets[e]):int(offsets[e + 1])].float()).abs().max().item()
        for e in range(E)
        if int(offsets[e]) < int(offsets[e + 1])
    )
    assert wrong_err > 1.0, f'wrong-scale control error too small: {wrong_err}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_fuse_weight_rejects_global_scale():
    """Insurance: the fused-SiLU lane has a single per-tensor scale slot;
    a global-scale bundle must be rejected loudly instead of silently
    serving two differently-scaled projections with one scalar."""
    from lmdeploy.turbomind.linear import get_linear

    linear = get_linear()
    with linear._activate():
        plan = linear.get_weight_plan(weight_format=NVFP4Format(), dtype=torch.float16)
        w = torch.zeros(16, 64, dtype=torch.uint8)
        with pytest.raises(ValueError, match='global scale'):
            linear.fuse_weight((w, w), plan=plan, global_scale=torch.tensor([2.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cpu_source_packing_parity():
    """B6 prerequisite gate: the nvfp4 dense lane packed from HOST-resident
    raw-e4m3 codes plus the reciprocal global-scale bridge must match the
    device-source pack byte for byte (the prior round's documented gap)."""
    from lmdeploy.turbomind.linear import get_linear

    torch.manual_seed(0)
    fmt = NVFP4Format()
    K, N, M = 512, 128, 32
    w_dense = torch.randn(N, K) * (0.1 / math.sqrt(K))
    bits, scale, global_scale = _quantize_nvfp4(w_dense)

    linear = get_linear()
    with linear._activate():
        plan = linear.get_weight_plan(weight_format=fmt, dtype=torch.float16)
        weight_cpu = linear.prepare_weight(
            bits,
            plan=plan,
            scales=scale.to(torch.float8_e4m3fn),
            global_scale=global_scale.reshape(1),
        )
        weight_gpu = linear.prepare_weight(
            bits.cuda(),
            plan=plan,
            scales=scale.to(torch.float8_e4m3fn).cuda(),
            global_scale=global_scale.reshape(1).cuda(),
        )
        x = torch.randn(M, K, device='cuda', dtype=torch.float16)
        _, out_cpu, _ = linear.tune(x, weight_cpu)
        _, out_gpu, _ = linear.tune(x, weight_gpu)

    assert torch.equal(out_cpu, out_gpu), 'nvfp4 CPU-source packing diverged'
    # and the CPU-source pack is still numerically correct
    ref = x.float() @ _dequant_nvfp4(bits, scale, global_scale).cuda().float()
    assert (out_cpu.float() - ref).abs().max().item() < 1e-2
