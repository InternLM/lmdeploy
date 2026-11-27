# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""MoE-TP regressions for the TurboMind grouped MXFP4 route.

The down projection is row-parallel over moe-TP, so its bias must reach
the group's single outer reduce exactly once. These tests drive the real
weight loader on mocked per-rank contexts: the moe-TP lead rank owns the
down bias, non-lead ranks keep zero storage and must not expect the bias
key in their completeness ledger.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

NUM_EXPERTS = 8
TOP_K = 1
HIDDEN = 256
FFN = 256
GROUP = 32


def _silu_mul(inter: torch.Tensor):
    """The python-side activation override that keeps the impl on the split
    (bias-capable) lane."""
    return F.silu(inter[..., :FFN // 2]) * inter[..., FFN // 2:]


def _moe_tp_dist_context(moe_tp: int, moe_rank: int, ep: int = 1, ep_rank: int = 0):
    """Single-process fake rank context (mirrors the kernel suite's helper)."""
    from lmdeploy.pytorch.config import DistConfig
    from lmdeploy.pytorch.distributed import DistContext, DistGroup

    return DistContext(rank=ep_rank if ep > 1 else moe_rank,
                       dp_rank=0,
                       ep_rank=ep_rank,
                       dist_config=DistConfig(tp=moe_tp, ep=ep),
                       moe_tp_group=DistGroup(rank=moe_rank))


def _build_layer(device, moe_tp: int, moe_rank: int, ep: int = 1, ep_rank: int = 0):
    from lmdeploy.pytorch.distributed import get_dist_manager
    from lmdeploy.pytorch.nn.moe.tm_mxfp4 import FusedMoETmMXFP4

    rank_ctx = _moe_tp_dist_context(moe_tp, moe_rank, ep, ep_rank)
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
            layer_idx=moe_rank + ep_rank,
            act_func=_silu_mul,
        )
    return layer, rank_ctx


def _feed_experts(layer, rank_ctx, down_bias: torch.Tensor, gate_up_bias: torch.Tensor | None = None):
    """Feed zero experts plus the given down bias through the real
    weight_loader path, mirroring the gpt-oss loader: full checkpoint-width
    projections, per-expert, gate rows first and up rows second.

    Zero block bytes quantize to exact zeros, so the local matmul outputs vanish and only the bias reaches the rank
    output.
    """
    from lmdeploy.pytorch.distributed import get_dist_manager

    gate_up_blocks = torch.zeros(NUM_EXPERTS, 2 * FFN, HIDDEN // 2, dtype=torch.uint8)
    gate_up_scales = torch.zeros(NUM_EXPERTS, 2 * FFN, HIDDEN // 32, dtype=torch.uint8)
    if gate_up_bias is None:
        gate_up_bias = torch.zeros(NUM_EXPERTS, 2 * FFN)
    down_blocks = torch.zeros(NUM_EXPERTS, HIDDEN, FFN // 2, dtype=torch.uint8)
    down_scales = torch.zeros(NUM_EXPERTS, HIDDEN, FFN // 32, dtype=torch.uint8)

    sources = (
        (layer.gate_up.blocks, gate_up_blocks, 'gate_up'),
        (layer.gate_up.scales, gate_up_scales, 'gate_up'),
        (layer.gate_up.bias, gate_up_bias, 'gate_up'),
        (layer.down.blocks, down_blocks, 'down'),
        (layer.down.scales, down_scales, 'down'),
        (layer.down.bias, down_bias, 'down'),
    )
    with get_dist_manager().context(rank_ctx):
        for expert in range(NUM_EXPERTS):
            for param, stacked, weight_type in sources:
                if weight_type == 'gate_up':
                    for shard_id, rows in (('gate', slice(0, FFN)), ('up', slice(FFN, 2 * FFN))):
                        param.weight_loader(param, stacked[expert][rows].contiguous(), expert_id=expert,
                                            shard_id=shard_id)
                else:
                    param.weight_loader(param, stacked[expert].contiguous(), expert_id=expert, shard_id='down')
        layer.update_weights()


def _release(layer):
    layer.sleep_release()
    torch.cuda.empty_cache()


@torch.inference_mode()
def test_moe_tp2_down_bias_counts_once():
    """MoE-TP=2, zero weights, unit down bias: summing the per-rank outputs
    must reproduce the routing weight times the bias exactly once.

    A copy of the bias on every rank doubles the sum.
    """
    device = torch.device('cuda')
    from lmdeploy.pytorch.distributed import get_dist_manager

    layers = []
    try:
        for rank in (0, 1):
            layer, rank_ctx = _build_layer(device, moe_tp=2, moe_rank=rank)
            _feed_experts(layer, rank_ctx, down_bias=torch.ones(NUM_EXPERTS, HIDDEN))
            layers.append((layer, rank_ctx))

        gen = torch.Generator().manual_seed(11)
        x = torch.randn(4, HIDDEN, generator=gen).to(device, torch.bfloat16)
        topk_weights = torch.ones(4, TOP_K, device=device)
        topk_idx = torch.zeros(4, TOP_K, dtype=torch.long, device=device)

        total = None
        for layer, rank_ctx in layers:
            with get_dist_manager().context(rank_ctx):
                # all_reduce=False keeps forward off the wire: the returned
                # tensor is this rank's partial of the moe-TP sum.
                out = layer.forward(x, topk_weights, topk_idx).float()
            total = out if total is None else total + out

        torch.testing.assert_close(total, torch.ones_like(total), rtol=0, atol=0)
        for rank, (layer, _rank_ctx) in enumerate(layers):
            expected_bias_sum = NUM_EXPERTS * HIDDEN if rank == 0 else 0
            assert layer.down.bias.data.sum().item() == expected_bias_sum
    finally:
        for layer, _rank_ctx in layers:
            _release(layer)


@torch.inference_mode()
def test_ep_only_down_bias_single_owner():
    """EP-only (moe_tp=1): every rank is the moe-TP lead, each expert has a
    single owner, the bias loads and validates exactly as before."""
    device = torch.device('cuda')
    from lmdeploy.pytorch.distributed import get_dist_manager

    gen = torch.Generator().manual_seed(3)
    down_bias = torch.randn(NUM_EXPERTS, HIDDEN, generator=gen)
    gate_up_bias = torch.randn(NUM_EXPERTS, 2 * FFN, generator=gen)
    num_local = NUM_EXPERTS // 2
    for ep_rank in (0, 1):
        layer, rank_ctx = _build_layer(device, moe_tp=1, moe_rank=0, ep=2, ep_rank=ep_rank)
        try:
            _feed_experts(layer, rank_ctx, down_bias=down_bias, gate_up_bias=gate_up_bias)
            with get_dist_manager().context(rank_ctx):
                layer.gate_up.validate_complete()
                layer.down.validate_complete()
            first, last = ep_rank * num_local, (ep_rank + 1) * num_local
            assert torch.equal(layer.down.bias.data, down_bias[first:last].to(torch.bfloat16))
            assert torch.equal(layer.gate_up.bias.data, gate_up_bias[first:last].to(torch.bfloat16))
        finally:
            _release(layer)


@torch.inference_mode()
def test_nonlead_accepts_stream_without_down_bias():
    """A non-lead moe-TP rank's completeness ledger must accept a stream
    without down-bias parts; later bias arrivals are skipped silently and the
    storage stays zero.

    Its own gate/up bias slice still loads.
    """
    from lmdeploy.pytorch.distributed import get_dist_manager

    device = torch.device('cuda')
    layer, rank_ctx = _build_layer(device, moe_tp=2, moe_rank=1)
    try:
        gen = torch.Generator().manual_seed(5)
        gate_up_bias = torch.randn(NUM_EXPERTS, 2 * FFN, generator=gen)
        gate_up_blocks = torch.zeros(NUM_EXPERTS, 2 * FFN, HIDDEN // 2, dtype=torch.uint8)
        gate_up_scales = torch.zeros(NUM_EXPERTS, 2 * FFN, HIDDEN // 32, dtype=torch.uint8)
        down_blocks = torch.zeros(NUM_EXPERTS, HIDDEN, FFN // 2, dtype=torch.uint8)
        down_scales = torch.zeros(NUM_EXPERTS, HIDDEN, FFN // 32, dtype=torch.uint8)
        with get_dist_manager().context(rank_ctx):
            for expert in range(NUM_EXPERTS):
                for shard_id, rows in (('gate', slice(0, FFN)), ('up', slice(FFN, 2 * FFN))):
                    layer.gate_up.blocks.weight_loader(layer.gate_up.blocks, gate_up_blocks[expert][rows],
                                                       expert_id=expert, shard_id=shard_id)
                    layer.gate_up.scales.weight_loader(layer.gate_up.scales, gate_up_scales[expert][rows],
                                                       expert_id=expert, shard_id=shard_id)
                    layer.gate_up.bias.weight_loader(layer.gate_up.bias, gate_up_bias[expert][rows],
                                                     expert_id=expert, shard_id=shard_id)
                layer.down.blocks.weight_loader(layer.down.blocks, down_blocks[expert], expert_id=expert,
                                                shard_id='down')
                layer.down.scales.weight_loader(layer.down.scales, down_scales[expert], expert_id=expert,
                                                shard_id='down')
            # The bias-less stream is complete for a non-lead rank. Rank 1
            # owns the second moe-TP chunk of each projection: its bias
            # storage is gate rows [FFN//2:FFN] followed by up rows
            # [FFN + FFN//2:2*FFN].
            layer.gate_up.validate_complete()
            layer.down.validate_complete()
            local_ffn = FFN // 2
            expected_gate_up_bias = torch.cat(
                (gate_up_bias[:, local_ffn:FFN], gate_up_bias[:, FFN + local_ffn:2 * FFN]), dim=1)
            assert torch.equal(layer.gate_up.bias.data, expected_gate_up_bias.to(torch.bfloat16))
            assert layer.down.bias.data.abs().sum().item() == 0

            # A late down-bias arrival is skipped without touching the
            # storage or the ledger.
            unit = torch.ones(NUM_EXPERTS, HIDDEN)
            for expert in range(NUM_EXPERTS):
                layer.down.bias.weight_loader(layer.down.bias, unit[expert], expert_id=expert, shard_id='down')
            assert layer.down.bias.data.abs().sum().item() == 0
            layer.down.validate_complete()
    finally:
        _release(layer)


@torch.inference_mode()
def test_hot_update_resets_ledger_per_ownership():
    """A complete module re-fed from the start resets the ledger on the owner
    rank (storage takes the new stream) and leaves the non-lead storage
    untouched-zero."""
    device = torch.device('cuda')
    layers = []
    try:
        for rank in (0, 1):
            layer, rank_ctx = _build_layer(device, moe_tp=2, moe_rank=rank)
            _feed_experts(layer, rank_ctx, down_bias=torch.zeros(NUM_EXPERTS, HIDDEN))
            _feed_experts(layer, rank_ctx, down_bias=torch.full((NUM_EXPERTS, HIDDEN), 2.0))
            layers.append(layer)
            if rank == 0:
                assert layer.down.bias.data.sum().item() == 2.0 * NUM_EXPERTS * HIDDEN
            else:
                assert layer.down.bias.data.abs().sum().item() == 0
    finally:
        for layer in layers:
            _release(layer)
