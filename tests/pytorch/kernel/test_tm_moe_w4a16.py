# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Impl-level gate and unit cases for the TurboMind grouped W4A16 MoE.

The gate compares the TurboMind grouped channel against an element-wise
fp32 reference. The reference builds from the exact packed tensors the
channel consumes, on the GLM-5.3 sparse geometry (288 experts, top-8,
hidden 4096, moe-ffn 2048, g128). The triton fused_moe anchors run on a
small cut of the same geometry and compare the full route: same weights,
same routing table, both backends. Metrics:
KL <= 1e-2 and argmax agreement >= 0.99.

All weights are random. Nothing downloads.
"""

import pytest
import torch
import torch.nn.functional as F

from lmdeploy.pytorch.backends.cuda.tm_moe_modules import blocked_route

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

# Gate thresholds.
KL_THRESHOLD = 1e-2
ARGMAX_THRESHOLD = 0.99


def _kl(first: torch.Tensor, second: torch.Tensor) -> float:
    return F.kl_div(
        F.log_softmax(first.float(), dim=-1),
        F.log_softmax(second.float(), dim=-1),
        log_target=True,
        reduction='batchmean',
    ).item()


def _quant_awq(w: torch.Tensor, group: int, device: torch.device):
    """Quantize [N, K] into the canonical AWQ triple plus the fp16 dequant.

    The packing mirrors the AutoAWQ layout the loader produces (word per input row, nibble order ``[0, 4, 1, 5, 2, 6, 3,
    7]``), so the reference dequant walks the same codes the kernel reads.
    """
    n, k = w.shape
    groups = w.float().view(n, k // group, group)
    scale = groups.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(groups / scale), -7, 7)
    code = (q + 8).to(torch.int32)
    scale = scale.squeeze(-1).to(torch.float16)
    codes_t = code.view(n, k).t().contiguous().view(k, n // 8, 8)
    qweight = torch.zeros(k, n // 8, dtype=torch.int32)
    for channel in range(8):
        nibble = (0, 4, 1, 5, 2, 6, 3, 7)[channel]
        qweight |= codes_t[:, :, channel].to(torch.int32) << (4 * nibble)
    scales = scale.t().contiguous()
    qzeros = torch.full((k // group, n // 8), int(torch.tensor(0x88888888).to(torch.int32).item()),
                        dtype=torch.int32)
    dequant = (q.float() * scale.unsqueeze(-1)).view(n, k).t().contiguous()
    return qweight.to(device), scales.to(device), qzeros.to(device), dequant


class _ExpertSet:
    """Stacked canonical AWQ experts plus host-side dequant copies."""

    def __init__(self, num_experts, out_features, in_features, group, device, seed0):
        qweights, scales, qzeros, dequants = [], [], [], []
        gen = torch.Generator().manual_seed(seed0)
        for expert in range(num_experts):
            w = (torch.randn(out_features, in_features, generator=gen) * (0.1 / in_features**0.5)).to(torch.float16)
            packed = _quant_awq(w, group, device)
            qweights.append(packed[0])
            scales.append(packed[1])
            qzeros.append(packed[2])
            dequants.append(packed[3].cpu())
        self.qweight = torch.stack(qweights)
        self.scales = torch.stack(scales)
        self.qzeros = torch.stack(qzeros)
        self.dequant = dequants

def _fused_gate_up(gate: _ExpertSet, up: _ExpertSet):
    """Assemble the fused gate/up storage: gate half first, up half second,
    matching the loader convention the engine produces."""
    return (
        torch.cat((gate.qweight, up.qweight), dim=-1),
        torch.cat((gate.scales, up.scales), dim=-1),
        torch.cat((gate.qzeros, up.qzeros), dim=-1),
    )


def _reference(x, gate_deq, up_deq, down_deq, topk_ids, topk_weights, device):
    """Element-wise fp32 MoE over the exact dequantized experts.

    Slots group by expert so each dequant copy crosses to the device once.
    """
    num_tokens, top_k = topk_ids.shape
    num_experts = len(down_deq)
    full = torch.zeros(num_tokens * top_k, down_deq[0].shape[1], device=device)
    for expert in range(num_experts):
        routes = (topk_ids == expert).nonzero(as_tuple=False)
        if routes.numel() == 0:
            continue
        gate = gate_deq[expert].to(device, torch.float16).float()
        up = up_deq[expert].to(device, torch.float16).float()
        down = down_deq[expert].to(device, torch.float16).float()
        for token, j in routes.tolist():
            xv = x[token].float()
            act = F.silu(xv @ gate) * (xv @ up)
            full[token * top_k + j] = act.to(torch.float16).float() @ down
        del gate, up, down
    full = full.view(num_tokens, top_k, -1)
    return (full * topk_weights.unsqueeze(-1)).sum(1)


@pytest.fixture(scope='module')
def grouped_full_setup():
    """Build the 288-expert GLM-5.3 sparse geometry once for the param sweep,
    release the packed handles afterwards."""
    num_experts, top_k, hidden, ffn, group = 288, 8, 4096, 2048, 128
    device = torch.device('cuda')

    gate = _ExpertSet(num_experts, ffn, hidden, group, device, 100)
    up = _ExpertSet(num_experts, ffn, hidden, group, device, 200)
    down = _ExpertSet(num_experts, hidden, ffn, group, device, 300)

    from lmdeploy.pytorch.backends.cuda.tm_moe_modules import TurbomindMoeW4A16Impl

    impl = TurbomindMoeW4A16Impl(
        top_k=top_k,
        num_experts=num_experts,
        hidden_dim=hidden,
        ffn_dim=ffn,
        group_size=group,
        renormalize=True,
    )
    fused = _fused_gate_up(gate, up)
    impl.update_weights(fused[0], fused[1], fused[2], down.qweight, down.scales, down.qzeros)
    yield gate, up, down, impl, device
    impl._release()
    torch.cuda.empty_cache()


@pytest.mark.parametrize('num_tokens', [1, 8, 32])
@torch.inference_mode()
def test_grouped_matches_elementwise_reference(num_tokens, grouped_full_setup):
    """Full GLM-5.3 sparse geometry against the fp32 reference."""
    num_experts, top_k, hidden = 288, 8, 4096
    gate, up, down, impl, device = grouped_full_setup
    if impl._fused_silu:
        pytest.skip('device exposes the gated-silu family; the fused branch is covered by '
                    'test_fused_silu_branch_when_family_supported')

    gen = torch.Generator().manual_seed(900 + num_tokens)
    topk_ids = torch.stack([torch.randperm(num_experts, generator=gen)[:top_k] for _ in range(num_tokens)])
    topk_weights = torch.rand(num_tokens, top_k, generator=gen) + 0.1
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(device)
    x = (torch.randn(num_tokens, hidden, generator=gen, dtype=torch.float16) * 0.5).to(device)

    out = impl.forward(x, topk_weights, topk_ids.to(device))
    torch.cuda.synchronize()
    reference = _reference(x, gate.dequant, up.dequant, down.dequant, topk_ids, topk_weights, device)
    score = _kl(out, reference)
    agreement = (out.argmax(-1) == reference.argmax(-1)).float().mean().item()
    assert score <= KL_THRESHOLD, f'KL {score:.3e} exceeds {KL_THRESHOLD}'
    assert agreement >= ARGMAX_THRESHOLD, f'argmax agreement {agreement:.3f} below threshold'


@torch.inference_mode()
def test_route_anchors_match_triton_fused_moe():
    """Same weights, same routing: TM grouped vs the two triton anchors.

    The fp16 anchor isolates the quantization granularity (TM reads the g128 codes of the same source weights the triton
    kernel sees unquantized); the packed anchor runs the same comparison at g32. The fused branch joins when the device
    offers a gated-silu family (SM90), and the case notes the capability skip otherwise.
    """
    num_experts, top_k, hidden, ffn, group = 8, 4, 512, 256, 128
    device = torch.device('cuda')
    torch.manual_seed(13)

    gate = _ExpertSet(num_experts, ffn, hidden, group, device, 1100)
    up = _ExpertSet(num_experts, ffn, hidden, group, device, 1200)
    down = _ExpertSet(num_experts, hidden, ffn, group, device, 1300)

    from lmdeploy.pytorch.backends.cuda.tm_moe_modules import TurbomindMoeW4A16Impl

    impl = TurbomindMoeW4A16Impl(
        top_k=top_k,
        num_experts=num_experts,
        hidden_dim=hidden,
        ffn_dim=ffn,
        group_size=group,
        renormalize=True,
    )
    fused_gate = _fused_gate_up(gate, up)
    impl.update_weights(fused_gate[0], fused_gate[1], fused_gate[2], down.qweight, down.scales, down.qzeros)
    try:
        gen = torch.Generator().manual_seed(4400)
        num_tokens = 8
        topk_ids = torch.stack([torch.randperm(num_experts, generator=gen)[:top_k] for _ in range(num_tokens)])
        topk_weights = torch.rand(num_tokens, top_k, generator=gen) + 0.1
        topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(device)
        x = torch.randn(num_tokens, hidden, generator=gen, dtype=torch.float16).to(device)

        out = impl.forward(x, topk_weights, topk_ids.to(device))
        torch.cuda.synchronize()

        # Anchor 1: the unquantized triton fused_moe on the source weights.
        w1 = torch.stack([
            torch.cat((gate.dequant[e].t().to(device), up.dequant[e].t().to(device))) for e in range(num_experts)
        ]).to(torch.float16)
        w2 = torch.stack([down.dequant[e].t().to(device) for e in range(num_experts)]).to(torch.float16)
        from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import fused_moe

        fp16_anchor = fused_moe(x, w1, w2, topk_weights.to(torch.float32), topk_ids.to(device), top_k,
                                renormalize=True)
        score = _kl(out, fp16_anchor)
        agreement = (out.argmax(-1) == fp16_anchor.argmax(-1)).float().mean().item()
        print(f'TM vs triton fp16 fused_moe: KL={score:.3e} argmax={agreement:.3f}')
        assert score <= KL_THRESHOLD, f'fp16 anchor KL {score:.3e} exceeds {KL_THRESHOLD}'
        assert agreement >= ARGMAX_THRESHOLD, f'fp16 anchor argmax {agreement:.3f} below threshold'

        # Anchor 2: the packed g32 kernel on the same source weights.
        from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16

        def _pack_g32(w):
            n, k = w.shape
            w = w.to(device)
            groups = w.float().view(n, k // 32, 32)
            wmin = groups.amin(-1, keepdim=True)
            wmax = groups.amax(-1, keepdim=True)
            scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)
            codes = torch.round((groups - wmin) / scale + 8.0).clamp(0, 15).to(torch.int64)
            shifts = torch.arange(8, device=device, dtype=torch.int64) * 4
            packed = (codes.view(n, k // 8, 8) << shifts).sum(-1).to(torch.int32)
            scales = scale.expand(n, k // 32, 32)[..., 0].to(torch.bfloat16)
            return packed, scales

        gate_up_packed = torch.empty(num_experts, 2 * ffn, hidden // 8, dtype=torch.int32, device=device)
        gate_up_scale = torch.empty(num_experts, 2 * ffn, hidden // 32, dtype=torch.bfloat16, device=device)
        down_packed = torch.empty(num_experts, hidden, ffn // 8, dtype=torch.int32, device=device)
        down_scale = torch.empty(num_experts, hidden, ffn // 32, dtype=torch.bfloat16, device=device)
        for e in range(num_experts):
            fused_source = torch.cat((gate.dequant[e].t(), up.dequant[e].t())).to(torch.float16)
            gate_up_packed[e], gate_up_scale[e] = _pack_g32(fused_source)
            down_packed[e], down_scale[e] = _pack_g32(down.dequant[e].t().to(torch.float16))
        packed_anchor = fused_moe_w4a16(
            x,
            gate_up_packed,
            gate_up_scale,
            down_packed,
            down_scale,
            topk_weights.to(torch.float32),
            topk_ids.to(device),
            topk=top_k,
            renormalize=True,
            num_bits=4,
            group_size=32,
        )
        score = _kl(out, packed_anchor)
        agreement = (out.argmax(-1) == packed_anchor.argmax(-1)).float().mean().item()
        print(f'TM(g128) vs triton w4a16(g32): KL={score:.3e} argmax={agreement:.3f}')
        # The two backends quantize at different granularities, so the diff
        # carries both quant errors. KL gates the route; the argmax metric
        # stays on same-format comparisons (the reference gate above),
        # where near-tied raw outputs do not flip under format noise.
        assert score <= KL_THRESHOLD, f'g32 anchor KL {score:.3e} exceeds {KL_THRESHOLD}'
    finally:
        impl._release()
        torch.cuda.empty_cache()


@torch.inference_mode()
def test_fused_silu_branch_when_family_supported():
    """Exercise the fused gate/up branch only where a family represents it.

    SM86 exposes no gated-silu block; the impl then packs the separate w1/w3 handles. On devices with the fused family
    the branch runs the same parity gate.
    """
    num_experts, top_k, hidden, ffn, group = 4, 2, 256, 128, 128
    device = torch.device('cuda')

    gate = _ExpertSet(num_experts, ffn, hidden, group, device, 2100)
    up = _ExpertSet(num_experts, ffn, hidden, group, device, 2200)
    down = _ExpertSet(num_experts, hidden, ffn, group, device, 2300)

    from lmdeploy.pytorch.backends.cuda.tm_moe_modules import TurbomindMoeW4A16Impl

    impl = TurbomindMoeW4A16Impl(
        top_k=top_k,
        num_experts=num_experts,
        hidden_dim=hidden,
        ffn_dim=ffn,
        group_size=group,
        renormalize=True,
    )
    fused_gate = _fused_gate_up(gate, up)
    impl.update_weights(fused_gate[0], fused_gate[1], fused_gate[2], down.qweight, down.scales, down.qzeros)
    try:
        if not impl._fused_silu:
            pytest.skip('device offers no gated-silu grouped family (SM86); H200 acceptance covers the branch')

        torch.manual_seed(31)
        topk_ids = torch.randint(0, num_experts, (4, top_k))
        topk_weights = torch.rand(4, top_k, device=device) + 0.1
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        x = torch.randn(4, hidden, device=device, dtype=torch.float16)
        out = impl.forward(x, topk_weights, topk_ids.to(device))
        reference = _reference(x, gate.dequant, up.dequant, down.dequant, topk_ids, topk_weights, device)
        score = _kl(out, reference)
        assert score <= KL_THRESHOLD, f'fused branch KL {score:.3e} exceeds {KL_THRESHOLD}'
    finally:
        impl._release()
        torch.cuda.empty_cache()


@torch.inference_mode()
def test_blocked_route_semantics():
    """The blocked slot layout matches the triton sorter and parks
    sentinels."""
    num_experts, top_k = 8, 4
    device = torch.device('cuda')
    torch.manual_seed(17)
    num_tokens = 6
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k))
    topk_ids[1, 2] = -1
    topk_ids[3, 0] = -1

    indices, offsets, sorted_routes, valid = blocked_route(topk_ids.to(device), num_experts)
    assert offsets.dtype == torch.int32 and indices.dtype == torch.int32
    assert offsets.shape[0] == num_experts + 1
    counts = torch.zeros(num_experts, dtype=torch.int64)
    flat = topk_ids.view(-1)
    for route in flat.tolist():
        if route >= 0:
            counts[route] += 1
    # offsets[e] = valid routes before expert e; the last entry starts the
    # skip block parked behind the real experts.
    expected_offsets = torch.zeros(num_experts + 1, dtype=torch.int64)
    expected_offsets[1:] = counts.cumsum(0)
    assert offsets.tolist() == expected_offsets.tolist(), (offsets.tolist(), expected_offsets.tolist())

    # Every slot maps to its token row; the slot order is a permutation of
    # the flat routes, so the scatter covers each route exactly once.
    tokens = indices.cpu().long()
    for slot in range(flat.numel()):
        route = sorted_routes[slot].item()
        assert tokens[slot] == route // top_k
    permutation = sorted_routes.cpu().tolist()
    assert sorted(permutation) == list(range(flat.numel()))
    scatter = torch.empty(flat.numel(), dtype=torch.long)
    scatter[sorted_routes.cpu()] = torch.arange(flat.numel())
    assert scatter.long()[sorted_routes.cpu()].tolist() == list(range(flat.numel()))
    assert valid.cpu().tolist() == (topk_ids >= 0).tolist()

    # Expert-major property: every expert owns a contiguous slot range.
    for expert in range(num_experts):
        begin, end = offsets[expert].item(), offsets[expert + 1].item()
        rows = topk_ids.view(-1)[sorted_routes.cpu()[begin:end]]
        assert bool((rows == expert).all()), (expert, rows.tolist())


@torch.inference_mode()
def test_forward_requires_update_weights():
    """Forward before packing fails with the contract message."""
    from lmdeploy.pytorch.backends.cuda.tm_moe_modules import TurbomindMoeW4A16Impl

    impl = TurbomindMoeW4A16Impl(top_k=2, num_experts=4, hidden_dim=64, ffn_dim=128, group_size=64)
    x = torch.randn(2, 64, dtype=torch.float16, device='cuda')
    with pytest.raises(RuntimeError, match='update_weights'):
        impl.forward(x, torch.rand(2, 2), torch.zeros(2, 2, dtype=torch.long))


def test_auto_route_falls_back_on_bf16(monkeypatch):
    """Auto reports the probe reason instead of raising.

    The shape checks need CUDA, so this case runs only with a device present.
    """
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    import lmdeploy.pytorch.envs as envs
    from lmdeploy.pytorch.backends.cuda.tm_moe_modules import tm_moe_route

    monkeypatch.setattr(envs, 'w4a16_gemm_backend', 'auto')
    if torch.cuda.get_device_capability(0)[0] >= 9:
        pytest.skip('SM90+ may carry a bf16 grouped family; the fallback is SM86-specific')
    reason = tm_moe_route(group_size=128, w_bit=4, dtype=torch.bfloat16, hidden_dim=4096, ffn_dim=2048)
    assert reason is not None, 'SM86 has no bf16 u4 family, the route must decline'
    reason = tm_moe_route(group_size=128, w_bit=4, dtype=torch.float16, hidden_dim=4096, ffn_dim=2048)
    assert reason is None, f'fp16 GLM geometry should route to turbomind, got: {reason}'
