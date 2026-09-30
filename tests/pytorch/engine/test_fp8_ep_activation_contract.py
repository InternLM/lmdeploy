# Copyright (c) OpenMMLab. All rights reserved.
"""Custom EP callbacks fail at construction, before any device resources."""
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.moe import blocked_fp8
from lmdeploy.pytorch.backends.moe import FusedMoEBlockedF8BuildSpec
from lmdeploy.pytorch.models.glm5_next import _GLM53_COMPACT_FP8_MOE_ACT
from lmdeploy.pytorch.models.gpt_oss import GateupAct


def spec(callback, ep_size=2):
    return FusedMoEBlockedF8BuildSpec(top_k=2, num_experts=4, hidden_dim=128, renormalize=False,
        block_size=128, ep_size=ep_size, ep_group=None, output_dtype=torch.bfloat16,
        fp8_dtype=torch.float8_e4m3fn, num_max_dispatch_tokens_per_rank=128,
        layer_idx=0, act_func=callback, scale_fmt=None)


@pytest.mark.parametrize('callback', [GateupAct.build(7., 1.702), lambda x: x,
                                      lambda x, *, masked_m: x])
def test_incompatible_callback_rejected_before_ep_allocation(monkeypatch, callback):
    constructor = Mock()
    monkeypatch.setattr(blocked_fp8, 'FusedDeepEpMoEBlockedF8Impl', constructor)
    with pytest.raises(TypeError, match='masked_m'):
        blocked_fp8._build_fused_moe_blocked_f8(spec(callback))
    constructor.assert_not_called()


@pytest.mark.parametrize('callback', [None, _GLM53_COMPACT_FP8_MOE_ACT,
                                      lambda x, masked_m=None: x])
def test_supported_ep_callback(monkeypatch, callback):
    constructor = Mock()
    monkeypatch.setattr(blocked_fp8, 'FusedDeepEpMoEBlockedF8Impl', constructor)
    assert blocked_fp8._build_fused_moe_blocked_f8(spec(callback)) is constructor.return_value
    constructor.assert_called_once()


def test_tensor_only_callback_still_supported_without_ep(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(blocked_fp8, 'TritonFusedMoEBlockedF8Impl', constructor)
    blocked_fp8._build_fused_moe_blocked_f8(spec(GateupAct.build(7., 1.702), ep_size=1))
    constructor.assert_called_once()


@pytest.mark.parametrize('renormalize', [False, True])
@pytest.mark.parametrize('output_scale', [1.0, 2.5])
def test_deepep_routed_scale_applied_once_at_combine(renormalize, output_scale):
    impl = blocked_fp8.FusedDeepEpMoEBlockedF8Impl.__new__(blocked_fp8.FusedDeepEpMoEBlockedF8Impl)
    impl.renormalize = renormalize
    impl.output_scale = output_scale
    weights = torch.tensor([[0.125, 0.375]], dtype=torch.float32)
    routed_weights = impl.do_renormalize(weights)
    layer = blocked_fp8.FusedMoELowLatency.__new__(blocked_fp8.FusedMoELowLatency)
    layer.output_scale = output_scale
    layer.token_dispatcher = Mock()
    layer.token_dispatcher.combine_async.side_effect = (
        lambda hidden_states, topk_idx, topk_weights, handle, async_finish: topk_weights.sum(-1)
    )
    output = layer.combine_async(torch.ones(1, 2), torch.tensor([[0, 1]]), routed_weights, None, False)
    expected_weights = weights / weights.sum(-1, keepdim=True) if renormalize else weights
    torch.testing.assert_close(output, expected_weights.sum(-1) * output_scale, rtol=0, atol=0)
