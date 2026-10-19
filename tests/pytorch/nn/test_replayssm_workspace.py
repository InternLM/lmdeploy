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
    from lmdeploy.pytorch.kernels.cuda import gated_delta_replayssm_spec as spec
    from lmdeploy.pytorch.kernels.cuda import kda_replayssm
    from lmdeploy.pytorch.nn.gated_delta import GatedDeltaReplayCursor, GatedDeltaReplayState

    # Meta tensors expose a large prefix budget without allocating its storage.
    slots, heads, dim, ring, steps = 65536, 2, 16, 8, 3
    raw = torch.empty(slots, heads, ring, dim, device='meta', dtype=torch.bfloat16)
    gate_shape = raw.shape if channelwise else raw.shape[:-1]
    state = GatedDeltaReplayState(
        mode='mtp', conv_state=raw, checkpoint=torch.empty(slots, heads, dim, dim, device='meta'),
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

    for batch in (3, 5, 2, 3):
        ids = torch.arange(slots - batch, slots)
        meta = SimpleNamespace(is_decoding=True, origin_state_ids=ids, state_ids=ids,
                               q_seqlens=torch.full((batch,), steps), cache_seqlens=torch.zeros(batch),
                               is_init_token=None)
        q = torch.empty(1, batch * steps, heads, dim)
        beta = torch.empty(1, batch * steps, heads)
        if channelwise:
            impl._forward_replay(torch.empty(1, batch * steps, 3 * heads * dim), q, beta,
                                 state.conv_state, state.checkpoint, meta, state,
                                 num_heads=heads, head_dim=dim, conv_weight=torch.empty(3 * heads * dim, 4),
                                 conv_bias=None, a_log=None, dt_bias=None, lower_bound=-5)
        else:
            impl.forward(q, q, q, beta, beta, None, None, state, meta, 1, True)
    # Growing and shrinking the batch never replaces storage retained by an earlier graph.
    assert len(impl._replayssm_workspaces) == 3
    assert seen[0] is seen[-1]
    assert {id(x) for x in impl._replayssm_workspaces.values()} == {id(x) for x in seen}
