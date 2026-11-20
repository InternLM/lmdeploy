# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Unit parity gates for the TurboMind grouped MXFP4 MoE route.

The impl is compared against an element-wise fp32 reference built from the
exact raw blocks the channel consumes: checkpoint-native ``[E, out, in // 2]``
uint8 blocks (two E2M1 codes per byte, low nibble first) and ``[E, out,
in // 32]`` E8M0 scale bytes (value ``2 ** (s - 127)``), group 32 — the same
fixture shape the TurboMind linear suite quantizes with QuantizeGroupwise.

The B1 regression case routes sentinel ids (``-1``, EP foreign experts) with
bias enabled: the skip block past ``offsets[num_experts]`` must map to a
parked expert id instead of shortening the slot-expert vector and crashing
the bias gather.
"""


import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN = 256
FFN = 256
GROUP = 32

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _quant_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize [out, in] fp16 to the checkpoint-native mxfp4 triple.

    Returns ``(blocks [out, in // 2] uint8, scales [out, in // 32] uint8,
    dequant [out, in] fp16)``.
    """
    out, in_features = w.shape
    groups = w.float().view(out, in_features // GROUP, GROUP)
    amax = groups.abs().amax(-1)
    e8m0_exp = torch.where(amax > 0, torch.ceil(torch.log2(amax / 6.0 + 1e-12)), torch.zeros_like(amax))
    e8m0_exp = e8m0_exp.clamp(min=-120)
    scale = torch.pow(2.0, e8m0_exp)
    values = torch.tensor(_E2M1_VALUES)
    # Nearest E2M1 code per element.
    dist = (groups.unsqueeze(-1) - values).abs()
    codes = dist.argmin(-1)
    blocks = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(out, in_features // 2).to(torch.uint8)
    scales = (e8m0_exp + 127).to(torch.uint8)
    dequant = (values[codes] * scale.unsqueeze(-1)).reshape(out, in_features).to(torch.float16)
    return blocks, scales, dequant


class _Mxfp4ExpertSet:
    """Stacked raw mxfp4 blocks plus host dequant copies."""

    def __init__(self, out_features, in_features, device, seed0):
        blocks, scales, dequants = [], [], []
        gen = torch.Generator().manual_seed(seed0)
        for _ in range(NUM_EXPERTS):
            w = (torch.randn(out_features, in_features, generator=gen) * (0.1 / in_features**0.5)).to(torch.float16)
            b, s, d = _quant_mxfp4(w)
            blocks.append(b)
            scales.append(s)
            dequants.append(d)
        self.blocks = torch.stack(blocks)
        self.scales = torch.stack(scales)
        self.dequant = dequants


def _silu_mul(inter: torch.Tensor) -> torch.Tensor:
    """The python-side activation override that keeps the impl on the split
    (bias-capable) lane on every device."""
    return F.silu(inter[..., :FFN]) * inter[..., FFN:]


def _build_impl(gate, up, down, gate_up_bias, down_bias, device):
    from lmdeploy.pytorch.backends.cuda.tm_moe_modules import TurbomindMoeMXFP4Impl

    impl = TurbomindMoeMXFP4Impl(
        top_k=TOP_K,
        num_experts=NUM_EXPERTS,
        hidden_dim=HIDDEN,
        ffn_dim=FFN,
        group_size=GROUP,
        renormalize=True,
        output_dtype=torch.bfloat16,
        act_func=_silu_mul,
    )
    # The impl consumes the deinterleaved fused layout: gate rows first,
    # up rows second (the loader splits the interleaved checkpoint tensor).
    gate_up_blocks = torch.cat((gate.blocks, up.blocks), dim=1).to(device)
    gate_up_scales = torch.cat((gate.scales, up.scales), dim=1).to(device)
    down_blocks = down.blocks.to(device)
    down_scales = down.scales.to(device)
    impl.update_weights(gate_up_blocks, gate_up_scales, gate_up_bias, down_blocks, down_scales, down_bias)
    return impl


def _reference(x, gate_deq, up_deq, down_deq, gate_up_bias, down_bias, topk_ids, topk_weights, device):
    """Element-wise fp32 MoE over the exact dequantized experts (with bias)."""
    num_tokens, top_k = topk_ids.shape
    full = torch.zeros(num_tokens * top_k, HIDDEN, device=device)
    for expert in range(NUM_EXPERTS):
        routes = (topk_ids == expert).nonzero(as_tuple=False)
        if routes.numel() == 0:
            continue
        gate = gate_deq[expert].to(device).float()
        up = up_deq[expert].to(device).float()
        down = down_deq[expert].to(device).float()
        gb = gate_up_bias[expert].float() if gate_up_bias is not None else None
        gate_bias = None if gb is None else gb[:FFN]
        up_bias = None if gb is None else gb[FFN:]
        db = down_bias[expert].float() if down_bias is not None else None
        for token, j in routes.tolist():
            xv = x[token].float()
            gate_out = xv @ gate if gate_bias is None else xv @ gate + gate_bias
            up_out = xv @ up if up_bias is None else xv @ up + up_bias
            act = F.silu(gate_out) * up_out
            down_out = act.to(torch.float16).float() @ down
            full[token * top_k + j] = down_out if db is None else down_out + db
    full = full.view(num_tokens, top_k, -1)
    return (full * topk_weights.unsqueeze(-1)).sum(1)


def _routing(num_tokens: int, sentinels: int, device, seed: int = 77):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_tokens, NUM_EXPERTS, generator=gen)
    weights, ids = logits.topk(TOP_K, dim=-1)
    weights = torch.softmax(weights, dim=-1).to(device)
    if sentinels:
        ids = ids.clone()
        # EP foreign experts: park ``sentinels`` routes on invalid ids.
        for row in range(min(sentinels, num_tokens)):
            ids[row, 0] = -1
            weights[row, 0] = 0.0
    return weights.to(device), ids.to(device)


def _run_parity(num_tokens, sentinels, with_bias, seed):
    device = torch.device('cuda')
    gate = _Mxfp4ExpertSet(FFN, HIDDEN, device, seed)
    up = _Mxfp4ExpertSet(FFN, HIDDEN, device, seed + 1)
    down = _Mxfp4ExpertSet(HIDDEN, FFN, device, seed + 2)

    gen = torch.Generator().manual_seed(seed + 10)
    gate_up_bias = (torch.randn(NUM_EXPERTS, 2 * FFN, generator=gen) * 0.02).to(device) if with_bias else None
    down_bias = (torch.randn(NUM_EXPERTS, HIDDEN, generator=gen) * 0.02).to(device) if with_bias else None

    impl = _build_impl(gate, up, down, gate_up_bias, down_bias, device)
    try:
        topk_weights, topk_ids = _routing(num_tokens, sentinels, device, seed + 20)
        if sentinels:
            # The impl builds with renormalize=True: sentinel weights are
            # zero, so the surviving weights renormalize by 1/(valid sum).
            # Pre-renormalizing here makes the impl's internal renormalize
            # a no-op and lets the raw-weight reference consume the same
            # mixing weights — without it the row-scale gap dominates the
            # error budget for tokens carrying sentinels.
            topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).contiguous()
        x = (torch.randn(num_tokens, HIDDEN, generator=gen, dtype=torch.float32) * 0.5).to(device,
                                                                                             torch.bfloat16)

        out = impl.forward(x, topk_weights, topk_ids)
        torch.cuda.synchronize()
        reference = _reference(
            x.float(), [d.to(torch.float32) for d in gate.dequant], [d.to(torch.float32) for d in up.dequant],
            [d.to(torch.float32) for d in down.dequant],
            None if gate_up_bias is None else gate_up_bias, None if down_bias is None else down_bias,
            topk_ids.cpu(), topk_weights.float(), device)
        diff = (out.float() - reference).abs()
        denom = reference.abs().clamp(min=1.0)
        assert (diff / denom).median() < 1e-2, f'median rel err {(diff / denom).median():.3e}'
        assert diff.mean() < 0.1, f'mean abs err {diff.mean():.3e}'
    finally:
        impl._release()
        torch.cuda.empty_cache()


@torch.inference_mode()
def test_parity_with_bias():
    """Full route with gate/up and down bias matches the dequant reference."""
    _run_parity(num_tokens=8, sentinels=0, with_bias=True, seed=100)


@torch.inference_mode()
def test_parity_without_bias():
    """The bias-free route matches the dequant reference."""
    _run_parity(num_tokens=8, sentinels=0, with_bias=False, seed=200)


@torch.inference_mode()
def test_sentinel_routing_with_bias_does_not_crash():
    """B1: sentinel (-1) routes lengthen the skip block past
    ``offsets[num_experts]``; with bias the slot-expert vector must still
    cover every slot and forward must match the reference (foreign routes
    contribute zero)."""
    _run_parity(num_tokens=8, sentinels=4, with_bias=True, seed=300)


@torch.inference_mode()
def test_pack_rejects_bias_without_act_override():
    """N1: the fused-SiLU-vs-bias conflict fails at pack time with a
    ValueError, not on every forward call."""
    device = torch.device('cuda')
    gate = _Mxfp4ExpertSet(FFN, HIDDEN, device, 400)
    up = _Mxfp4ExpertSet(FFN, HIDDEN, device, 401)
    down = _Mxfp4ExpertSet(HIDDEN, FFN, device, 402)

    from lmdeploy.pytorch.backends.cuda.tm_moe_modules import TurbomindMoeMXFP4Impl

    impl = TurbomindMoeMXFP4Impl(
        top_k=TOP_K,
        num_experts=NUM_EXPERTS,
        hidden_dim=HIDDEN,
        ffn_dim=FFN,
        group_size=GROUP,
        output_dtype=torch.bfloat16,
        act_func=None,
    )
    gate_up_blocks = torch.cat((gate.blocks, up.blocks), dim=1).to(device)
    gate_up_scales = torch.cat((gate.scales, up.scales), dim=1).to(device)
    with pytest.raises(ValueError, match='bias'):
        impl.update_weights(gate_up_blocks, gate_up_scales, torch.zeros(NUM_EXPERTS, 2 * FFN, device=device),
                            down.blocks.to(device), down.scales.to(device), None)


def _ep_dist_context(ep: int, ep_rank: int, dp: int = 1):
    """Single-process fake EP context (mirrors the W4A16 suite's helper)."""
    from lmdeploy.pytorch.config import DistConfig
    from lmdeploy.pytorch.distributed import DistContext, DistGroup

    return DistContext(rank=ep_rank,
                       dp_rank=0,
                       ep_rank=ep_rank,
                       dist_config=DistConfig(ep=ep, dp=dp),
                       moe_tp_group=DistGroup(rank=0))


def test_global_top_k_invariant_rejects_build():
    """The global router invariant (top_k <= num_experts) fires on any EP width
    before device/impl work, so a plain CPU build suffices."""
    from lmdeploy.pytorch.distributed import get_dist_manager
    from lmdeploy.pytorch.nn.moe.tm_mxfp4 import FusedMoETmMXFP4

    with get_dist_manager().context(_ep_dist_context(2, 0)):
        with pytest.raises(RuntimeError, match='top_k=8 <= num_experts=4'):
            FusedMoETmMXFP4(
                hidden_dim=64,
                ffn_dim=32,
                num_experts=4,
                top_k=8,
                dtype=torch.float16,
                device=torch.device('cpu'),
                group_size=32,
                layer_idx=0,
            )


def _build_mx_layer(device, ep=1, ep_rank=0):
    """Construct the module under the rank's EP view (mirrors the W4A16
    suite)."""
    from lmdeploy.pytorch.distributed import get_dist_manager
    from lmdeploy.pytorch.nn.moe.tm_mxfp4 import FusedMoETmMXFP4

    rank_ctx = _ep_dist_context(ep, ep_rank)
    with get_dist_manager().context(rank_ctx):
        layer = FusedMoETmMXFP4(
            hidden_dim=HIDDEN,
            ffn_dim=FFN,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            bias=True,
            renormalize=True,
            dtype=torch.bfloat16,
            device=device,
            all_reduce=False,
            group_size=GROUP,
            layer_idx=0,
            act_func=_silu_mul,
        )
    return layer, rank_ctx


def _feed_cpu_experts(layer, rank_ctx, seed):
    """Fill the CPU-resident parameters through the real weight_loader path.

    Mirrors the gpt-oss loader: per-expert, already-deinterleaved
    projections (gate rows first, up rows second). Scale bytes stay valid
    e8m0 (exponent <= 127, i.e. byte <= 254).
    """
    from lmdeploy.pytorch.distributed import get_dist_manager

    gen = torch.Generator().manual_seed(seed)
    # Random block bytes keep the round trip input-agnostic; scales stay in
    # [2^-127, 2^0] so the dequantized experts cannot overflow bf16 into
    # NaNs (NaN != NaN would break the bitwise equality assertion).
    gate_up_blocks = torch.randint(0, 255, (NUM_EXPERTS, 2 * FFN, HIDDEN // 2), dtype=torch.uint8)
    gate_up_scales = torch.randint(0, 128, (NUM_EXPERTS, 2 * FFN, HIDDEN // 32), dtype=torch.uint8)
    gate_up_bias = torch.randn(NUM_EXPERTS, 2 * FFN, generator=gen) * 0.02
    down_blocks = torch.randint(0, 255, (NUM_EXPERTS, HIDDEN, FFN // 2), dtype=torch.uint8)
    down_scales = torch.randint(0, 128, (NUM_EXPERTS, HIDDEN, FFN // 32), dtype=torch.uint8)
    down_bias = torch.randn(NUM_EXPERTS, HIDDEN, generator=gen) * 0.02

    sources = {
        (layer.gate_up.blocks, gate_up_blocks, 'gate_up', 'blocks'),
        (layer.gate_up.scales, gate_up_scales, 'gate_up', 'scales'),
        (layer.gate_up.bias, gate_up_bias, 'gate_up', 'bias'),
        (layer.down.blocks, down_blocks, 'down', 'blocks'),
        (layer.down.scales, down_scales, 'down', 'scales'),
        (layer.down.bias, down_bias, 'down', 'bias'),
    }
    with get_dist_manager().context(rank_ctx):
        for expert in range(NUM_EXPERTS):
            for param, stacked, weight_type, _part in sources:
                if weight_type == 'gate_up':
                    # The loader deinterleaves the fused output dim into
                    # gate rows (first) and up rows (second) before feeding.
                    for shard_id, rows in (('gate', slice(0, FFN)), ('up', slice(FFN, 2 * FFN))):
                        param.weight_loader(param, stacked[expert][rows].contiguous(), expert_id=expert,
                                            shard_id=shard_id)
                else:
                    param.weight_loader(param, stacked[expert].contiguous(), expert_id=expert, shard_id='down')
        layer.update_weights()


@torch.inference_mode()
def test_cpu_resident_sleep_wakeup_roundtrip():
    """CPU-direct sleep/wakeup: the CPU-resident parameters stay put across
    model moves, sleep frees only the packed handles, and wakeup re-packs from
    the host bytes reproducing the forward bitwise."""
    from lmdeploy.pytorch.distributed import get_dist_manager

    device = torch.device('cuda')
    layer, rank_ctx = _build_mx_layer(device)

    # Deployment order: the engine moves the whole model to CUDA after the
    # load (and again on wakeup). The cpu_resident parameters must skip it.
    layer.to(device)
    assert layer.gate_up.blocks.device.type == 'cpu'
    assert layer.down.blocks.device.type == 'cpu'

    _feed_cpu_experts(layer, rank_ctx, seed=7)
    assert layer.gate_up.blocks.numel() != 0

    gen = torch.Generator().manual_seed(77)
    x = (torch.randn(4, HIDDEN, generator=gen, dtype=torch.float32) * 0.5).to(device, torch.bfloat16)
    topk_ids = torch.stack([torch.randperm(NUM_EXPERTS, generator=gen)[:TOP_K] for _ in range(4)])
    topk_weights = torch.rand(4, TOP_K, generator=gen) + 0.1
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(device)
    topk_ids = topk_ids.to(device)

    with get_dist_manager().context(rank_ctx):
        out_first = layer.forward(x, topk_weights, topk_ids).cpu()
        torch.cuda.synchronize()

        # Forward-side bias copies are device-resident even though their
        # source parameters are CPU-resident.
        assert layer.impl._gate_up_bias.device.type == 'cuda'
        assert layer.impl._down_bias.device.type == 'cuda'

        # sleep level 1: the packed handles are freed; the CPU bytes stay.
        layer.sleep_release()
        impl = layer.impl
        assert impl._w13 is None and impl._w_gate is None and impl._w_up is None and impl._w_down is None
        assert impl._linear is None
        assert layer.gate_up.blocks.device.type == 'cpu' and layer.gate_up.blocks.numel() != 0

        # wakeup: the model move skips the CPU-resident parameters again,
        # then sleep_wakeup re-packs from the unchanged host bytes.
        layer.to(device)
        assert layer.gate_up.blocks.device.type == 'cpu'
        layer.sleep_wakeup()
        out_second = layer.forward(x, topk_weights, topk_ids).cpu()
        torch.cuda.synchronize()

    assert torch.equal(out_first, out_second)

    layer.impl.sleep_release()
    torch.cuda.empty_cache()
