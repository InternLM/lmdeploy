# Copyright (c) OpenMMLab. All rights reserved.
"""Check stream dependencies and the model hook without allocating a full LLM.

The branch computations are small BF16 MLPs. Distributed DeepEP and performance are validated separately by the eight-
rank benchmark and full-model HTTP runs.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from lmdeploy.pytorch.models import glm5_next

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')


class Gate(nn.Module):
    def forward(self, x, routed_experts=None):
        ids = torch.arange(8, device=x.device).expand(len(x), 8).contiguous()
        if routed_experts is not None:
            routed_experts.copy_(ids)
        return torch.ones(len(x), 8, device=x.device) / 8, ids


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('up', torch.randn(1024, 128, device='cuda', dtype=torch.bfloat16) / 16)
        self.register_buffer('down', torch.randn(128, 1024, device='cuda', dtype=torch.bfloat16) / 32)

    def forward(self, x, *routing):
        return F.linear(F.silu(F.linear(x, self.up)), self.down)


def make_moe(monkeypatch, decoding):
    moe = glm5_next.Glm5NextMoE.__new__(glm5_next.Glm5NextMoE)
    nn.Module.__init__(moe)
    moe.experts = MLP()
    moe.shared_experts = MLP()
    moe.gate = Gate()
    moe.layer_idx = 0
    moe._all_reduce = moe._fp32_tp_reduce = False
    moe._enable_shared_expert_overlap = True
    moe._shared_expert_ready_event = None
    context = SimpleNamespace(global_is_decoding=lambda: decoding)
    monkeypatch.setattr(glm5_next, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(current_context=lambda: context))
    return moe


@pytest.mark.parametrize('rows', [1, 6, 12, 32, 192])
def test_overlap_matches_serial_and_graph_replay(monkeypatch, rows):
    moe = make_moe(monkeypatch, True)
    x = torch.randn(1, rows, 128, device='cuda', dtype=torch.bfloat16)
    routed = torch.empty(rows, 1, 8, device='cuda', dtype=torch.int64)
    with torch.inference_mode():
        expected = (moe.experts(x) + moe.shared_experts(x))
        actual = moe(x, all_routed_experts=routed)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(routed[:, 0], torch.arange(8, device='cuda').expand(rows, 8))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                moe(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = moe(x)
        for _ in range(3):
            x.normal_()
            expected = moe.experts(x) + moe.shared_experts(x)
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_globally_prefilling_step_keeps_serial_path(monkeypatch):
    moe = make_moe(monkeypatch, False)
    monkeypatch.setattr(glm5_next, '_get_glm53_shared_expert_stream',
                        lambda device: pytest.fail('prefill must not fork an auxiliary stream'))
    x = torch.randn(1, 6, 128, device='cuda', dtype=torch.bfloat16)
    with torch.inference_mode():
        torch.testing.assert_close(moe(x), moe.experts(x) + moe.shared_experts(x), rtol=0, atol=0)


def test_disabled_overlap_keeps_serial_path(monkeypatch):
    moe = make_moe(monkeypatch, True)
    moe._enable_shared_expert_overlap = False
    monkeypatch.setattr(glm5_next, '_get_glm53_shared_expert_stream',
                        lambda device: pytest.fail('disabled overlap must not use an auxiliary stream'))
    x = torch.randn(1, 6, 128, device='cuda', dtype=torch.bfloat16)
    with torch.inference_mode():
        torch.testing.assert_close(moe(x), moe.experts(x) + moe.shared_experts(x), rtol=0, atol=0)
