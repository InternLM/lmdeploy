# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch
from torch import nn

from lmdeploy.pytorch.models import hy3 as hy3_module
from lmdeploy.pytorch.models.hy3 import Hy3MoE


class _FakeRouter(nn.Module):

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_bias: torch.Tensor,
    ):
        self.calls += 1
        num_tokens = hidden_states.shape[0]
        logits = hidden_states.new_zeros((num_tokens, 1))
        weights = hidden_states.new_ones((num_tokens, 1))
        expert_ids = torch.zeros(
            (num_tokens, 1),
            dtype=torch.int64,
            device=hidden_states.device,
        )
        return logits, weights, expert_ids


class _FakeRoutedExperts(nn.Module):

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        self.calls += 1
        del topk_weights, topk_ids
        return hidden_states * 2


class _FakeSharedExpert(nn.Module):

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        self.calls += 1
        return hidden_states * 3


def _make_test_moe(
    *,
    device: torch.device,
    enable_overlap: bool,
    enable_fp32_combine: bool,
):
    moe = Hy3MoE.__new__(Hy3MoE)
    nn.Module.__init__(moe)

    moe.hidden_size = 16
    moe.enable_moe_fp32_combine = enable_fp32_combine
    moe.router = _FakeRouter()
    moe.experts = _FakeRoutedExperts()
    moe.shared_mlp = _FakeSharedExpert()
    moe.expert_bias = nn.Parameter(
        torch.zeros(
            1,
            dtype=torch.float32,
            device=device,
        ),
        requires_grad=False,
    )
    moe._all_reduce = False
    moe._enable_shared_expert_overlap = enable_overlap
    moe._shared_expert_ready_event = None
    return moe


def _expected_output(
    hidden_states: torch.Tensor,
    enable_fp32_combine: bool,
):
    routed_output = hidden_states * 2
    shared_output = hidden_states * 3
    if enable_fp32_combine:
        return (
            routed_output.float()
            + shared_output.float()
        ).to(hidden_states.dtype)
    return routed_output + shared_output


@pytest.mark.parametrize('num_tokens', [1, 32])
@pytest.mark.parametrize('enable_fp32_combine', [False, True])
def test_hy3_shared_expert_overlap_cpu_uses_serial_fallback(
    num_tokens,
    enable_fp32_combine,
):
    device = torch.device('cpu')
    moe = _make_test_moe(
        device=device,
        enable_overlap=True,
        enable_fp32_combine=enable_fp32_combine,
    )
    hidden_states = torch.randn(
        (num_tokens, 16),
        dtype=torch.bfloat16,
        device=device,
    )
    input_copy = hidden_states.clone()

    output = moe(hidden_states)
    expected = _expected_output(
        hidden_states,
        enable_fp32_combine,
    )

    assert torch.equal(output, expected)
    assert torch.equal(hidden_states, input_copy)
    assert moe._shared_expert_ready_event is None
    assert moe.router.calls == 1
    assert moe.experts.calls == 1
    assert moe.shared_mlp.calls == 1


@pytest.mark.parametrize('enable_overlap', [False, True])
def test_hy3_shared_expert_overlap_reads_environment_gate(
    monkeypatch,
    enable_overlap,
):
    monkeypatch.setattr(
        hy3_module._envs,
        'hy3_shared_expert_overlap',
        enable_overlap,
    )
    monkeypatch.setattr(
        hy3_module,
        'Hy3Router',
        lambda *args, **kwargs: _FakeRouter(),
    )
    monkeypatch.setattr(
        hy3_module,
        'build_fused_moe',
        lambda *args, **kwargs: _FakeRoutedExperts(),
    )
    monkeypatch.setattr(
        hy3_module,
        'Hy3MLP',
        lambda *args, **kwargs: _FakeSharedExpert(),
    )
    monkeypatch.setattr(
        hy3_module,
        'get_tp_world_rank',
        lambda: (1, 0),
    )

    config = type(
        '_Config',
        (),
        {
            'hidden_size': 16,
            'enable_moe_fp32_combine': False,
            'quantization_config': None,
            'num_experts': 1,
            'num_experts_per_tok': 1,
            'moe_intermediate_size': 8,
            'num_shared_experts': 1,
        },
    )()
    moe = Hy3MoE(
        config,
        layer_idx=0,
        dtype=torch.float32,
        device=torch.device('cpu'),
    )

    assert moe._enable_shared_expert_overlap is enable_overlap


def test_hy3_shared_expert_overlap_keeps_one_all_reduce(
    monkeypatch,
):
    device = torch.device('cpu')
    moe = _make_test_moe(
        device=device,
        enable_overlap=False,
        enable_fp32_combine=True,
    )
    moe._all_reduce = True
    hidden_states = torch.randn(
        (4, 16),
        dtype=torch.bfloat16,
        device=device,
    )
    expected = _expected_output(
        hidden_states,
        enable_fp32_combine=True,
    )
    all_reduce_inputs = []

    def _fake_all_reduce(output):
        all_reduce_inputs.append(output.clone())

    monkeypatch.setattr(
        hy3_module.dist,
        'all_reduce',
        _fake_all_reduce,
    )

    output = moe(hidden_states)

    assert len(all_reduce_inputs) == 1
    assert torch.equal(all_reduce_inputs[0], expected)
    assert torch.equal(output, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA is required for shared-expert overlap',
)
@pytest.mark.parametrize('num_tokens', [1, 32])
@pytest.mark.parametrize('enable_fp32_combine', [False, True])
def test_hy3_shared_expert_overlap_cuda_is_bitwise_exact(
    num_tokens,
    enable_fp32_combine,
):
    device = torch.device('cuda')
    serial_moe = _make_test_moe(
        device=device,
        enable_overlap=False,
        enable_fp32_combine=enable_fp32_combine,
    )
    overlap_moe = _make_test_moe(
        device=device,
        enable_overlap=True,
        enable_fp32_combine=enable_fp32_combine,
    )
    hidden_states = torch.randn(
        (num_tokens, 16),
        dtype=torch.bfloat16,
        device=device,
    )
    input_copy = hidden_states.clone()

    expected = serial_moe(hidden_states)
    output = overlap_moe(hidden_states)
    torch.cuda.synchronize()

    assert torch.equal(output, expected)
    assert torch.equal(hidden_states, input_copy)
    assert overlap_moe._shared_expert_ready_event is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA is required for shared-expert overlap',
)
@pytest.mark.parametrize('num_tokens', [1, 32])
def test_hy3_shared_expert_overlap_cuda_graph_replays(
    num_tokens,
):
    device = torch.device('cuda')
    serial_moe = _make_test_moe(
        device=device,
        enable_overlap=False,
        enable_fp32_combine=True,
    )
    overlap_moe = _make_test_moe(
        device=device,
        enable_overlap=True,
        enable_fp32_combine=True,
    )
    static_input = torch.randn(
        (num_tokens, 16),
        dtype=torch.bfloat16,
        device=device,
    )

    # Initialize the process-local stream and per-layer event outside capture.
    overlap_moe(static_input)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = overlap_moe(static_input)

    for replay_id in range(1, 4):
        replay_input = torch.full_like(
            static_input,
            replay_id / 4,
        )
        static_input.copy_(replay_input)
        graph.replay()
        torch.cuda.synchronize()

        expected = serial_moe(replay_input)
        torch.cuda.synchronize()
        assert torch.equal(graph_output, expected)
