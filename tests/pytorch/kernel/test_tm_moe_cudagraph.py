# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""CUDA graph capture gates for the TurboMind grouped W4A16 MoE route.

The route captures as ordinary kernels (no eager boundary): the forward path
is capture-clean except for two stream hazards this suite pins.

- DLPack exchange: the native side consumes every tensor export on the current
  torch stream, and ``Linear`` passes that stream to ``__dlpack__``. On torch
  builds inside the pytorch/pytorch#150217..#163242 window a no-stream export
  maps the consumer to the legacy default stream, which would invalidate any
  capture on a side stream.
- Stream-bound workspace: the native GEMM workspace is created per stream and
  its construction synchronizes (gemm.cu), so every capture stream must have
  seen one warmup call before capture begins. A cold capture aborts the
  process (TM fatal), asserted in a subprocess.

Acceptance (01-gptq-moe.md 验收 5 and the #5149 lesson):

- capture/replay numeric equality is bitwise on identical inputs;
- a graph captured on sentinel-only routing (all ``-1``), the DP1/EP dummy
  shape, still routes real tokens after replay: routed expert output matches
  fresh eager execution bitwise. The route consumes no graph-baked mask —
  ``blocked_route`` re-derives offsets from live routing on the device.
"""

import os
import subprocess
import sys

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.tm_moe_modules import TurbomindMoeW4A16Impl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tm_moe_w4a16 import _ExpertSet, _fused_gate_up  # noqa: E402

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN = 128
FFN = 256
GROUP = 64


def _packed_impl(seed: int = 7) -> TurbomindMoeW4A16Impl:
    """Build one packed impl on a small expert set."""
    device = torch.device('cuda')
    gate = _ExpertSet(NUM_EXPERTS, FFN, HIDDEN, GROUP, device, seed)
    up = _ExpertSet(NUM_EXPERTS, FFN, HIDDEN, GROUP, device, seed + 3)
    down = _ExpertSet(NUM_EXPERTS, HIDDEN, FFN, GROUP, device, seed + 1)
    impl = TurbomindMoeW4A16Impl(
        top_k=TOP_K,
        num_experts=NUM_EXPERTS,
        hidden_dim=HIDDEN,
        ffn_dim=FFN,
        group_size=GROUP,
        output_dtype=torch.float16,
    )
    gate_up_qw, gate_up_sc, gate_up_qz = _fused_gate_up(gate, up)
    impl.update_weights(gate_up_qw.cuda(), gate_up_sc.cuda(), gate_up_qz.cuda(), down.qweight.cuda(),
                        down.scales.cuda(), down.qzeros.cuda())
    return impl


def _routing(num_tokens: int, sentinel: bool):
    """Routing table with real expert spread; ``sentinel`` parks every route in
    the skip block (the DP1/EP dummy shape)."""
    gen = torch.Generator().manual_seed(11)
    logits = torch.randn(num_tokens, NUM_EXPERTS, generator=gen)
    weights, ids = logits.topk(TOP_K, dim=-1)
    weights = torch.softmax(weights, dim=-1)
    if sentinel:
        ids = ids.clone()
        ids[:, -1] = -1
        weights = torch.zeros_like(weights)
    return weights.cuda(), ids.cuda()


def _warm_on_stream(impl, stream: torch.cuda.Stream, inputs, iterations: int = 3) -> None:
    """Run eager warmup calls on ``stream`` so its stream-bound workspace and
    context exist before capture."""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(iterations):
            impl.forward(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    stream.synchronize()


def _capture(impl, inputs, stream: torch.cuda.Stream) -> tuple[torch.cuda.CUDAGraph, tuple]:
    """Capture one full forward on ``stream`` over stable input buffers."""
    graph = torch.cuda.CUDAGraph()
    static_inputs = tuple(value.clone() for value in inputs)
    with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle(), stream=stream):
        output = impl.forward(*static_inputs)
    return graph, static_inputs, output


def test_capture_replay_is_bitwise_identical():
    """验收 5: replaying the captured inputs reproduces the capture-time output
    bitwise."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    impl = _packed_impl()
    inputs = (torch.randn(6, HIDDEN, dtype=torch.float16, device='cuda'), *_routing(6, sentinel=False))
    eager = impl.forward(*inputs)

    stream = torch.cuda.Stream()
    _warm_on_stream(impl, stream, inputs)
    graph, static_inputs, captured = _capture(impl, inputs, stream)
    torch.cuda.synchronize()

    for static, value in zip(static_inputs, inputs):
        static.copy_(value)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)
    # A second replay must be stable too.
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)


def test_replay_tracks_mutated_inputs():
    """Replay with different inputs matches fresh eager execution — the
    captured routing path reads live routing, not baked values."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    impl = _packed_impl(seed=17)
    stream = torch.cuda.Stream()

    first = (torch.randn(6, HIDDEN, dtype=torch.float16, device='cuda'), *_routing(6, sentinel=False))
    _warm_on_stream(impl, stream, first)
    graph, static_inputs, captured = _capture(impl, first, stream)
    torch.cuda.synchronize()

    second = (torch.randn(6, HIDDEN, dtype=torch.float16, device='cuda'), *_routing(6, sentinel=True))
    eager_second = impl.forward(*(value.clone() for value in second))
    for static, value in zip(static_inputs, second):
        static.copy_(value)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager_second)


def test_dummy_capture_replays_real_routing():
    """#5149 scenario: a graph captured on the dummy shape (every route parked
    in the skip block) still feeds routed experts on replay."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    impl = _packed_impl(seed=23)
    stream = torch.cuda.Stream()

    real = (torch.randn(6, HIDDEN, dtype=torch.float16, device='cuda'), *_routing(6, sentinel=False))
    dummy = (real[0], *_routing(6, sentinel=True))
    _warm_on_stream(impl, stream, real)
    graph, static_inputs, captured = _capture(impl, dummy, stream)
    torch.cuda.synchronize()

    for static, value in zip(static_inputs, real):
        static.copy_(value)
    graph.replay()
    torch.cuda.synchronize()
    eager = impl.forward(*(value.clone() for value in real))
    assert captured.abs().sum() > 0, 'routed experts received no tokens on replay'
    assert torch.equal(captured, eager)


def test_cold_capture_without_stream_warmup_aborts():
    """Pin the H2 hazard: the first TM call on an unwarmed capture stream
    constructs its workspace inside the capture, and the construction
    synchronizes — the TM fatal aborts the process."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    program = f'''
import sys
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
import torch
from test_tm_moe_w4a16 import _ExpertSet, _fused_gate_up
from lmdeploy.pytorch.backends.cuda.tm_moe_modules import TurbomindMoeW4A16Impl

impl = TurbomindMoeW4A16Impl(top_k={TOP_K}, num_experts={NUM_EXPERTS}, hidden_dim={HIDDEN},
                             ffn_dim={FFN}, group_size={GROUP}, output_dtype=torch.float16)
gate = _ExpertSet({NUM_EXPERTS}, {FFN}, {HIDDEN}, {GROUP}, torch.device('cuda'), 7)
up = _ExpertSet({NUM_EXPERTS}, {FFN}, {HIDDEN}, {GROUP}, torch.device('cuda'), 10)
down = _ExpertSet({NUM_EXPERTS}, {HIDDEN}, {FFN}, {GROUP}, torch.device('cuda'), 8)
qw, sc, qz = _fused_gate_up(gate, up)
impl.update_weights(qw.cuda(), sc.cuda(), qz.cuda(), down.qweight.cuda(), down.scales.cuda(),
                    down.qzeros.cuda())
x = torch.randn(4, {HIDDEN}, dtype=torch.float16, device='cuda')
weights, ids = torch.rand(4, {TOP_K}, device='cuda'), torch.randint(0, {NUM_EXPERTS}, (4, {TOP_K}),
                                                                    device='cuda')
impl.forward(x, weights, ids)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
# Fresh side stream, no warmup call on it: capture must fail.
side = torch.cuda.Stream()
with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle(), stream=side):
    impl.forward(x, weights, ids)
print('COLD CAPTURE UNEXPECTEDLY SUCCEEDED')
'''
    result = subprocess.run(
        [sys.executable, '-c', program],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, 'LMDEPLOY_USE_MODELSCOPE': 'true'},
    )
    assert result.returncode != 0, 'cold capture must not succeed without stream warmup'
    assert 'COLD CAPTURE UNEXPECTEDLY SUCCEEDED' not in result.stdout
    assert 'stream is capturing' in result.stderr


def test_piecewise_trace_with_zero_warmup_iterations():
    """Engine-shaped PCG flow: one eager warmup on the capture stream, then
    ``trace_piecewise_cuda_graph`` with ``warmup_iterations=0`` — the invariant
    ``PiecewiseGraphManager.prepare`` guarantees."""
    from lmdeploy.pytorch.backends.cuda.graph_runner.piecewise import trace_piecewise_cuda_graph

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    impl = _packed_impl(seed=31)
    inputs = (torch.randn(8, HIDDEN, dtype=torch.float16, device='cuda'), *_routing(8, sentinel=False))

    stream = torch.cuda.Stream()
    _warm_on_stream(impl, stream, inputs, iterations=1)

    def forward(x, topk_weights, topk_ids):
        return impl.forward(x, topk_weights, topk_ids)

    plan = trace_piecewise_cuda_graph(
        forward,
        inputs,
        warmup_iterations=0,
        stream=stream,
    )
    torch.cuda.synchronize()
    replayed = plan.replay(*inputs)
    torch.cuda.synchronize()
    eager = impl.forward(*(value.clone() for value in inputs))
    assert torch.equal(replayed, eager)

    other = (torch.randn(8, HIDDEN, dtype=torch.float16, device='cuda'), *_routing(8, sentinel=True))
    eager_other = impl.forward(*(value.clone() for value in other))
    replayed_other = plan.replay(*other)
    torch.cuda.synchronize()
    assert torch.equal(replayed_other, eager_other)
