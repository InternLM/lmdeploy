# Copyright (c) OpenMMLab. All rights reserved.
"""CPU dispatch regressions; these do not qualify GPU numerics or GSM8K."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl
from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaRuleImpl
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent


@pytest.mark.parametrize('is_decoding', [False, True])
@pytest.mark.parametrize('num_spec_tokens', [0, 3, 5])
@pytest.mark.parametrize('normalize', [False, True])
def test_flag_off_delegates_all_arguments_unchanged(monkeypatch, is_decoding,
                                                  num_spec_tokens, normalize):
    impl = CudaGatedDeltaRuleImpl()
    state = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    saved_state = state.clone()
    meta = SimpleNamespace(is_decoding=is_decoding, num_spec_tokens=num_spec_tokens)
    args = tuple(object() for _ in range(7)) + (state, meta, 4, normalize)
    result = object()
    calls = []

    def legacy_forward(self, *received):
        assert self is impl
        assert len(received) == len(args)
        assert all(actual is expected for actual, expected in zip(received, args))
        calls.append(received)
        return result

    monkeypatch.setattr(GatedDeltaRuleImpl, 'forward', legacy_forward)
    assert impl.forward(*args) is result
    assert len(calls) == 1
    assert not hasattr(impl, '_replayssm_workspaces')
    assert torch.equal(state, saved_state)


@pytest.mark.parametrize('is_decoding', [False, True])
@pytest.mark.parametrize('kv_ratio', [1, 3, 4])
@pytest.mark.parametrize('rank4_ba', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_flag_off_default_prepare_matches_legacy_formula(is_decoding, kv_ratio,
                                                        rank4_ba, dtype):
    generator = torch.Generator().manual_seed(20260918)
    batch, seqlen, heads, dim = 2, 5, 2, 8
    qkv = torch.randn(batch, seqlen, heads * (2 + kv_ratio) * dim,
                      generator=generator, dtype=dtype)
    q, k, _ = qkv.split([heads * dim, heads * dim, heads * kv_ratio * dim], dim=-1)
    q, k = (x.unflatten(-1, (heads, dim)) for x in (q, k))
    shape = (batch, seqlen, heads, kv_ratio) if rank4_ba else (batch, seqlen, heads * kv_ratio)
    a = torch.randn(shape, generator=generator, dtype=dtype)
    b = torch.randn(shape, generator=generator, dtype=dtype)
    bias = torch.randn(heads * kv_ratio, generator=generator, dtype=dtype)
    a_log_exp = -torch.rand(heads * kv_ratio, generator=generator)
    mask = torch.tensor([True, False, True, False, False])
    beta = b.sigmoid().flatten(-2, -1) if rank4_ba else b.sigmoid()
    a_float = a.float().flatten(-2, -1) if rank4_ba else a.float()
    g = a_log_exp * F.softplus(a_float + bias)
    if not is_decoding:
        g = g.masked_fill(mask[None, :, None], -1.0e6)
    expected = (q.repeat_interleave(kv_ratio, dim=-2),
                k.repeat_interleave(kv_ratio, dim=-2), g, beta)
    impl = CudaGatedDeltaRuleImpl()
    actual = impl.prepare_inputs(q, k, b, a, bias, a_log_exp, kv_ratio,
                                 is_decoding=is_decoding, init_token_mask=mask)
    for value, reference in zip(actual[:4], expected):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
    assert actual[4] is False
    assert not hasattr(impl, '_replayssm_workspaces')


@pytest.mark.parametrize('is_decoding', [False, True])
def test_flag_off_commit_never_accesses_replay_state(is_decoding):
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.model_config = SimpleNamespace(enable_replayssm=False)
    # Deliberately omit state_cache_engine, spec_agent and rejection counts:
    # the flag-off path must return before accessing any ReplaySSM state.
    agent._commit_mtp_replayssm(SimpleNamespace(is_decoding=is_decoding), object())
