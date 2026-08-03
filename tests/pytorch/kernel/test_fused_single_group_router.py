# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        'CUDA is required for the fused router module',
        allow_module_level=True,
    )

from lmdeploy.pytorch.backends.cuda.moe_router import (
    TritonRouterNoauxTCImpl,
)
from lmdeploy.pytorch.backends.default.moe_router import (
    DefaultRouterNoauxTCImpl,
)


def _make_router(
    *,
    num_experts=192,
    top_k=8,
    n_group=1,
    topk_group=1,
    scoring_func='sigmoid',
    renormalize=True,
    router_n_groups=-1,
    routed_scaling_factor=1.25,
):
    return TritonRouterNoauxTCImpl(
        scoring_func=scoring_func,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        n_routed_experts=num_experts,
        routed_scaling_factor=routed_scaling_factor,
        renormalize=renormalize,
        router_n_groups=router_n_groups,
    )


def _make_reference(routed_scaling_factor=1.25):
    return DefaultRouterNoauxTCImpl(
        scoring_func='sigmoid',
        top_k=8,
        n_group=1,
        topk_group=1,
        n_routed_experts=192,
        routed_scaling_factor=routed_scaling_factor,
        renormalize=True,
        router_n_groups=-1,
    )


def _canonicalize(weights, expert_ids):
    order = expert_ids.argsort(dim=-1)
    return (
        weights.gather(1, order),
        expert_ids.gather(1, order),
    )


def _assert_matches_reference(router, logits, bias):
    reference = _make_reference(router.routed_scaling_factor)
    expected_weights, expected_ids = reference.forward(logits, bias)
    actual_weights, actual_ids = router.forward(logits, bias)
    expected_weights, expected_ids = _canonicalize(
        expected_weights,
        expected_ids,
    )
    actual_weights, actual_ids = _canonicalize(
        actual_weights,
        actual_ids,
    )

    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(
        actual_weights,
        expected_weights,
        atol=2e-7,
        rtol=2e-6,
    )


def test_single_group_fused_router_is_default_off(monkeypatch):
    monkeypatch.delenv(
        'LMDEPLOY_ROUTER_SINGLE_GROUP_FUSED',
        raising=False,
    )
    router = _make_router()
    assert not router.enable_single_group_fused


@pytest.mark.parametrize(
    'overrides',
    [
        {
            'num_experts': 128,
        },
        {
            'top_k': 4,
        },
        {
            'n_group': 2,
            'topk_group': 1,
        },
        {
            'scoring_func': 'softmax',
        },
        {
            'renormalize': False,
        },
        {
            'router_n_groups': 2,
        },
    ],
)
def test_single_group_fused_router_rejects_ineligible_shapes(
    monkeypatch,
    overrides,
):
    monkeypatch.setenv(
        'LMDEPLOY_ROUTER_SINGLE_GROUP_FUSED',
        '1',
    )
    router = _make_router(**overrides)
    assert not router.enable_single_group_fused


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA is required for the fused router',
)
@pytest.mark.parametrize('num_tokens', [1, 2, 8, 32, 128, 512])
@pytest.mark.parametrize('logit_scale', [0.25, 1.0, 4.0])
def test_single_group_fused_router_matches_reference(
    monkeypatch,
    num_tokens,
    logit_scale,
):
    monkeypatch.setenv(
        'LMDEPLOY_ROUTER_SINGLE_GROUP_FUSED',
        '1',
    )
    router = _make_router()
    generator = torch.Generator(device='cuda')
    generator.manual_seed(20260729 + num_tokens)
    logits = (
        torch.randn(
            (num_tokens, 192),
            dtype=torch.float32,
            device='cuda',
            generator=generator,
        )
        * logit_scale
    )
    bias = torch.linspace(
        -0.125,
        0.125,
        192,
        dtype=torch.float32,
        device='cuda',
    )
    _assert_matches_reference(router, logits, bias)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA is required for the fused router',
)
def test_single_group_fused_router_matches_near_ties(monkeypatch):
    monkeypatch.setenv(
        'LMDEPLOY_ROUTER_SINGLE_GROUP_FUSED',
        '1',
    )
    router = _make_router()
    logits = torch.zeros(
        (32, 192),
        dtype=torch.float32,
        device='cuda',
    )
    bias = (
        torch.arange(
            192,
            dtype=torch.float32,
            device='cuda',
        )
        * 2e-6
    )
    _assert_matches_reference(router, logits, bias)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA is required for the fused router',
)
def test_single_group_fused_router_exact_tie_invariants(monkeypatch):
    monkeypatch.setenv(
        'LMDEPLOY_ROUTER_SINGLE_GROUP_FUSED',
        '1',
    )
    routed_scaling_factor = 1.25
    router = _make_router(
        routed_scaling_factor=routed_scaling_factor,
    )
    logits = torch.zeros(
        (32, 192),
        dtype=torch.float32,
        device='cuda',
    )
    bias = torch.zeros(
        192,
        dtype=torch.float32,
        device='cuda',
    )

    weights, expert_ids = router.forward(logits, bias)

    assert torch.all((expert_ids >= 0) & (expert_ids < 192))
    assert torch.all(
        expert_ids.sort(dim=-1).values.diff(dim=-1) > 0
    )
    torch.testing.assert_close(
        weights,
        torch.full_like(
            weights,
            routed_scaling_factor / 8,
        ),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA is required for the fused router',
)
@pytest.mark.parametrize('num_tokens', [1, 32])
def test_single_group_fused_router_cuda_graph_replays(
    monkeypatch,
    num_tokens,
):
    monkeypatch.setenv(
        'LMDEPLOY_ROUTER_SINGLE_GROUP_FUSED',
        '1',
    )
    router = _make_router()
    static_logits = torch.randn(
        (num_tokens, 192),
        dtype=torch.float32,
        device='cuda',
    )
    bias = torch.linspace(
        -0.125,
        0.125,
        192,
        dtype=torch.float32,
        device='cuda',
    )

    # Compile the Triton kernel before capture.
    router.forward(static_logits, bias)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_weights, graph_ids = router.forward(
            static_logits,
            bias,
        )

    for replay_id in range(1, 4):
        replay_logits = torch.randn_like(static_logits)
        replay_logits.add_(replay_id / 4)
        static_logits.copy_(replay_logits)
        graph.replay()
        torch.cuda.synchronize()

        reference = _make_reference(
            router.routed_scaling_factor,
        )
        expected_weights, expected_ids = reference.forward(
            replay_logits,
            bias,
        )
        expected_weights, expected_ids = _canonicalize(
            expected_weights,
            expected_ids,
        )
        actual_weights, actual_ids = _canonicalize(
            graph_weights,
            graph_ids,
        )

        assert torch.equal(actual_ids, expected_ids)
        torch.testing.assert_close(
            actual_weights,
            expected_weights,
            atol=2e-7,
            rtol=2e-6,
        )
