# Copyright (c) OpenMMLab. All rights reserved.
"""CPU allocation/ownership checks; numerical coverage lives in kernel
tests."""
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('channelwise', [False, True], ids=['gdn', 'kda'])
def test_workspace_capacity_tracks_batch_not_prefix_slots(monkeypatch, channelwise):
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl
    from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaReplayCursor, GatedDeltaReplayState
    from lmdeploy.pytorch.kernels.cuda import gated_delta_replayssm_spec as spec
    from lmdeploy.pytorch.kernels.cuda import kda_replayssm

    # Meta tensors expose a large prefix budget without allocating its storage.
    slots, heads, dim, ring, steps = 65536, 2, 16, 8, 3
    raw = torch.empty(slots, heads, ring, dim, device='meta', dtype=torch.bfloat16)
    gate_shape = raw.shape if channelwise else raw.shape[:-1]
    state = GatedDeltaReplayState(
        conv_state=raw, checkpoint=torch.empty(slots, heads, dim, dim, device='meta'),
        d_ring=raw, k_ring=raw, g_ring=torch.empty(gate_shape, device='meta'),
        cursor=GatedDeltaReplayCursor(torch.empty(slots, 3, device='meta'), ring, ring, steps),
        raw_v_ring=raw, raw_k_ring=raw, beta_ring=torch.empty(slots, heads, ring, device='meta'))
    seen = []

    def verify(q, k, v, g, beta, *args, workspace, **kwargs):
        seen.append(workspace)
        assert all(x.shape[0] == 1 << (q.shape[0] - 1).bit_length() for x in workspace)
        return v, state.checkpoint

    monkeypatch.setattr(spec, 'fused_recurrent_gdn_replayssm_spec', verify)
    if channelwise:
        # Only dispatch/allocation is under test; do not import optional FLA or run CUDA kernels.
        impl = CudaKdaImpl.__new__(CudaKdaImpl)
        impl.shared_conv_update = lambda x, *args, **kwargs: x
        monkeypatch.setattr(kda_replayssm, 'prepare_kda_replay_gates', lambda g, beta, *args: (g, beta))
    else:
        impl = CudaGatedDeltaRuleImpl()
        monkeypatch.setattr(impl, 'prepare_inputs', lambda q, k, b, a, *args, **kwargs: (q, k, a, b, False))
        with pytest.raises(ValueError, match='Q/K L2 normalization'):
            impl.forward(None, None, None, None, None, None, None, state, None, 1, False)

    for batch in (3, 5, 2, 3):
        ids = torch.arange(slots - batch, slots)
        meta = SimpleNamespace(is_decoding=True, num_spec_tokens=steps - 1, origin_state_ids=ids, state_ids=ids,
                               q_seqlens=torch.full((batch,), steps), cache_seqlens=torch.zeros(batch),
                               is_init_token=None)
        q = torch.empty(1, batch * steps, heads, dim)
        beta = torch.empty(1, batch * steps, heads)
        if channelwise:
            impl._forward_replay(torch.empty(1, batch * steps, 3 * heads * dim), q, beta,
                                 meta, state,
                                 num_heads=heads, head_dim=dim, conv_weight=torch.empty(3 * heads * dim, 4),
                                 conv_bias=None, a_log=None, dt_bias=None, lower_bound=-5)
        else:
            impl.forward(q, q, q, beta, beta, None, None, state, meta, 1, True)
    # Growing and shrinking the batch never replaces storage retained by an earlier graph.
    assert len(impl._replayssm_workspaces) == 3
    assert seen[0] is seen[-1]
    assert {id(x) for x in impl._replayssm_workspaces.values()} == {id(x) for x in seen}


@pytest.mark.parametrize('is_decoding', [False, True], ids=['prefill', 'decode'])
@pytest.mark.parametrize('num_spec_tokens', [0, 3], ids=['standard', 'speculative'])
def test_gdn_replay_dispatch_uses_step_metadata(monkeypatch, is_decoding, num_spec_tokens):
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl
    from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaReplayState
    from lmdeploy.pytorch.kernels.cuda import gated_delta_replayssm as standard
    from lmdeploy.pytorch.kernels.cuda import gated_delta_replayssm_spec as spec

    class Cursor:
        R = 8
        packed_cursor = torch.zeros(1, 3, dtype=torch.int32)

        @property
        def Tmax(self):
            raise AssertionError('Backend dispatch must use metadata, not cursor capacity')

    # Kernel calls are intercepted: this checks dispatch, not an invalid state ABI.
    ring = torch.empty(1, 2, 8, 16)
    state = GatedDeltaReplayState(
        conv_state=torch.empty(0), checkpoint=torch.empty(1, 2, 16, 16),
        d_ring=ring, k_ring=ring, g_ring=torch.empty(1, 2, 8), cursor=Cursor(),
        raw_v_ring=ring, raw_k_ring=ring, beta_ring=torch.empty(1, 2, 8))
    ids = torch.zeros(1, dtype=torch.int64)
    # A speculative prefill can contain just one token without becoming Standard.
    steps = num_spec_tokens + 1 if is_decoding else 1
    meta = SimpleNamespace(num_spec_tokens=num_spec_tokens, is_decoding=is_decoding,
                           is_init_token=None, origin_state_ids=ids, state_ids=ids,
                           q_seqlens=torch.tensor([steps]), cu_seqlens=torch.tensor([0, steps]))
    q = torch.empty(1, steps, 2, 16)
    gates = torch.empty(1, steps, 2)
    impl = CudaGatedDeltaRuleImpl()
    monkeypatch.setattr(impl, 'prepare_inputs', lambda *args, **kwargs: (q, q, gates, gates, False))
    calls = []

    def record(name):
        def call(*args, **kwargs):
            calls.append(name)
            return q, state.checkpoint
        return call

    monkeypatch.setattr(standard, 'fold_gdn_replayssm_standard_history', record('standard_fold'))
    monkeypatch.setattr(spec, 'fold_gdn_replayssm_mtp_history', record('spec_fold'))
    monkeypatch.setattr(standard, 'fused_recurrent_gdn_replayssm_decode', record('standard_decode'))
    monkeypatch.setattr(spec, 'fused_recurrent_gdn_replayssm_spec', record('spec_decode'))
    monkeypatch.setattr(impl, 'chunk_gated_delta_rule', record('prefill'))
    impl.forward(q, q, q, gates, gates, None, None, state, meta, 1, True)
    kind = 'spec' if num_spec_tokens > 0 else 'standard'
    assert calls == ([f'{kind}_decode'] if is_decoding else [f'{kind}_fold', 'prefill'])
