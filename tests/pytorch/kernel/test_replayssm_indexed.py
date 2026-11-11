# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaReplayCursor, GatedDeltaReplayState
from lmdeploy.pytorch.compact_state import SSMPrefillMetadata, SSMStateMetadata


def _state(spec, kind='kda'):
    n, h, d, length = 5, 1, 128, 16
    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(*shape, device='cuda', dtype=dtype)
    return GatedDeltaReplayState(
        conv_state=zeros(n, 3*d, 4+spec, dtype=torch.bfloat16), checkpoint=zeros(n, h, d, d),
        d_ring=zeros(n, h, length, d, dtype=torch.float32 if kind == 'kda' or not spec else torch.bfloat16),
        k_ring=zeros(n, h, length, d, dtype=torch.float32 if kind == 'kda' or not spec else torch.bfloat16),
        g_ring=zeros(n, h, length, *([d] if kind == 'kda' else [])),
        cursor=GatedDeltaReplayCursor(zeros(n, 3, dtype=torch.int32), length, length, spec+1),
        raw_v_ring=zeros(n, h, length, d, dtype=torch.bfloat16) if spec else None,
        raw_k_ring=zeros(n, h, length, d, dtype=torch.bfloat16) if spec else None,
        beta_ring=zeros(n, h, length) if spec else None,
        shared_checkpoint=zeros(9, h, d, d), shared_conv_state=zeros(9, 3*d, 3, dtype=torch.bfloat16))


def _metadata(monkeypatch, lengths, histories, slots, sources, cps, offsets, spec, decode=False):
    from lmdeploy.pytorch.backends import gated_delta_rule as module
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaReplayGatedDeltaMeta
    monkeypatch.setattr(module, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(build_ctx=SimpleNamespace(num_spec_tokens=spec)))
    prefill = SSMPrefillMetadata.from_lengths(lengths, histories, checkpoint_ids=None if decode else cps,
                                             checkpoint_offsets=None if decode else offsets).to_device('cuda')
    ids = SSMStateMetadata(torch.empty((len(slots), 0), device='cuda', dtype=torch.long),
                           torch.tensor(sources, device='cuda'),
                           torch.tensor(cps, device='cuda')[:, None] if decode else
                           torch.full((len(slots), 1), -1, device='cuda', dtype=torch.long))
    q = torch.tensor(lengths, device='cuda', dtype=torch.int32)
    attn = SimpleNamespace(is_decoding=decode, cu_seqlens_q=prefill.cu_seqlens, q_seqlens=q,
                           kv_seqlens=q+torch.tensor(histories, device='cuda'))
    context = SimpleNamespace(ssm_state_meta=ids, ssm_prefill_meta=None if decode else prefill)
    return CudaReplayGatedDeltaMeta(sum(lengths), 4, torch.tensor(slots, device='cuda'), attn, context)


@pytest.mark.parametrize('spec', [0, 5])
@pytest.mark.parametrize('offset', [1, 2, 64, 72])
def test_conv_restore_and_internal_export(monkeypatch, spec, offset):
    from lmdeploy.pytorch.backends.cuda.causal_conv1d import CausalConv1dTilelangImpl
    from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_fn
    torch.manual_seed(701)
    state = _state(spec)
    state.conv_state.fill_(float('nan'))
    state.shared_conv_state.normal_()
    shared_before = state.shared_conv_state.clone()
    # Restore, fresh recycled slot, and a retained-history row with unrelated ID order.
    slots, histories, sources = [3, 1, 4], [256, 0, 97], [7, -1, -1]
    initial = torch.randn(3, 384, 3, device='cuda', dtype=torch.bfloat16)
    initial[0] = state.shared_conv_state[7]
    initial[1].zero_()
    positions = ((histories[2] + torch.arange(-3, 0, device='cuda')) % (4+spec)
                 if spec else torch.arange(1, 4, device='cuda'))
    state.conv_state[4, :, positions] = initial[2]
    meta = _metadata(monkeypatch, [72]*3, histories, slots, sources, [1, 5, 8], [64]*3, spec)
    # Conv alone also supports short offsets; recurrence planner only admits chunk boundaries.
    meta.ssm_prefill_meta = replace(meta.ssm_prefill_meta,
                                    checkpoint_offsets=torch.full((3,), offset, device='cuda', dtype=torch.long))
    x = torch.randn(1, 216, 384, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(384, 4, device='cuda', dtype=torch.bfloat16)/8
    expected = causal_conv1d_fn(x.transpose(1, 2), weight, seq_idx=meta.seq_idx,
                                initial_states=initial, activation='silu').transpose(1, 2)
    actual = CausalConv1dTilelangImpl(indexed_state=True).forward(x, weight, None, state, meta, 'silu')
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for row, (slot, history, cp) in enumerate(zip(slots, histories, [1, 5, 8])):
        raw = x[0, row*72:(row+1)*72].T
        checkpoint = torch.cat((initial[row], raw[:, :offset]), dim=1)[:, -3:]
        torch.testing.assert_close(state.shared_conv_state[cp], checkpoint, rtol=0, atol=0)
        shared_before[cp] = checkpoint
        positions = ((history+72+torch.arange(-4, 0, device='cuda')) % (4+spec)
                     if spec else torch.arange(4, device='cuda'))
        torch.testing.assert_close(state.conv_state[slot, :, positions], raw[:, -4:], rtol=0, atol=0)
    torch.testing.assert_close(state.shared_conv_state, shared_before, rtol=0, atol=0)


@pytest.mark.parametrize('spec', [0, 5])
@pytest.mark.parametrize('kind', ['kda', 'gdn'])
def test_backend_prefill_shared_checkpoint_restore(monkeypatch, spec, kind):
    from lmdeploy.pytorch.backends.cuda.causal_conv1d import CausalConv1dTilelangImpl
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl
    torch.manual_seed(730)
    state = _state(spec, kind)
    legacy_state = replace(_state(spec, kind), shared_checkpoint=None, shared_conv_state=None)
    impl = CudaKdaImpl(indexed_state=True) if kind == 'kda' else CudaGatedDeltaRuleImpl(indexed_state=True)
    legacy = CudaKdaImpl() if kind == 'kda' else CudaGatedDeltaRuleImpl()
    conv = CausalConv1dTilelangImpl(indexed_state=True)
    legacy_conv = CausalConv1dTilelangImpl()
    raw = torch.randn(1, 256, 384, device='cuda', dtype=torch.bfloat16)/4
    gate = torch.randn(1, 256, 128 if kind == 'kda' else 1, device='cuda', dtype=torch.bfloat16)
    beta = torch.randn(1, 256, 1, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(384, 4, device='cuda', dtype=torch.bfloat16)/8
    if kind == 'kda':
        weight = weight.float() + 0.0001  # Real GLM keeps FP32 conv parameters.
    a_log = torch.zeros(1, device='cuda')
    bias = torch.zeros(128 if kind == 'kda' else 1, device='cuda')

    def run(backend, conv_backend, st, start, end, slot, source=-1, cp=-1, offset=0, decode=False):
        from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm import reset_gdn_replayssm_cursor
        meta = _metadata(monkeypatch, [end-start], [start], [slot], [source], [cp], [offset], spec, decode=decode)
        if not decode and (start == 0 or source > 0):
            reset_gdn_replayssm_cursor(st.cursor.packed_cursor, meta.origin_state_ids)
        if kind == 'kda':
            output = backend.forward(raw[:, start:end], gate[:, start:end], beta[:, start:end], weight, None,
                                      a_log, bias, None, st, meta, 1, 128, -5.0)
        else:
            mixed = conv_backend.forward(raw[:, start:end], weight, None,
                                          st if st.shared_conv_state is not None else st.conv_state, meta, 'silu')
            q, k, v = [x.unflatten(-1, (1, 128)) for x in mixed.split(128, dim=-1)]
            output = backend.forward(q, k, v, beta[:, start:end], gate[:, start:end], bias,
                                      -a_log.exp(), st, meta, 1, True)
        if not decode:
            reset_gdn_replayssm_cursor(st.cursor.packed_cursor, meta.origin_state_ids)
        return output

    actual = run(impl, conv, state, 0, 72, 1, cp=1, offset=64)
    expected = run(legacy, legacy_conv, legacy_state, 0, 72, 1)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    torch.testing.assert_close(state.checkpoint[1], legacy_state.checkpoint[1], rtol=0.001, atol=0.001)
    run(legacy, legacy_conv, legacy_state, 0, 64, 2)
    torch.testing.assert_close(state.shared_checkpoint[1], legacy_state.checkpoint[2], rtol=0.001, atol=0.001)
    restored = run(impl, conv, state, 64, 72, 3, source=1)
    ref = run(legacy, legacy_conv, legacy_state, 64, 72, 2)
    torch.testing.assert_close(restored, ref, rtol=0.02, atol=0.002)
    torch.testing.assert_close(state.checkpoint[3], legacy_state.checkpoint[2], rtol=0.001, atol=0.001)

    # Fixed-slot decoding, partial acceptance and L-wrap preserve the original Replay math.
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm import advance_gdn_replayssm_standard_cursor
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm_spec import commit_gdn_replayssm_spec_cursor
    history = 72
    for step in range(20):
        width = spec + 1
        accepted = step % (width+1) if spec else 1
        got = run(impl, conv, state, history, history+width, 3, decode=True)
        ref = run(legacy, legacy_conv, legacy_state, history, history+width, 2, decode=True)
        torch.testing.assert_close(got, ref, rtol=0.02, atol=0.002)
        for st, slot in ((state, 3), (legacy_state, 2)):
            ids = torch.tensor([slot], device='cuda')
            if spec:
                commit_gdn_replayssm_spec_cursor(st.cursor.packed_cursor, ids,
                                                torch.tensor([width], device='cuda'),
                                                torch.tensor([width-accepted], device='cuda'), 16, width)
            else:
                advance_gdn_replayssm_standard_cursor(st.cursor.packed_cursor, ids, 16)
        history += accepted
    # Retained-session prefill must fold only accepted history before consuming the suffix.
    got = run(impl, conv, state, history, history+8, 3)
    ref = run(legacy, legacy_conv, legacy_state, history, history+8, 2)
    torch.testing.assert_close(got, ref, rtol=0.02, atol=0.002)
    torch.testing.assert_close(state.checkpoint[3], legacy_state.checkpoint[2], rtol=0.001, atol=0.001)

    if not spec:
        # AR may publish between natural L flushes. Both matrix and conv must be exact.
        flagged = replace(state, force_flush=torch.ones(1, device='cuda', dtype=torch.bool))
        legacy_flagged = replace(legacy_state, force_flush=flagged.force_flush)
        checkpoint_before = state.shared_checkpoint[1].clone()
        got = run(impl, conv, flagged, history+8, history+9, 3, cp=4, decode=True)
        ref = run(legacy, legacy_conv, legacy_flagged, history+8, history+9, 2, decode=True)
        torch.testing.assert_close(got, ref, rtol=0.02, atol=0.002)
        torch.testing.assert_close(state.shared_checkpoint[4], legacy_state.checkpoint[2], rtol=0.001, atol=0.001)
        torch.testing.assert_close(state.shared_conv_state[4], legacy_state.conv_state[2, :, 1:], rtol=0, atol=0)
        torch.testing.assert_close(state.shared_checkpoint[1], checkpoint_before, rtol=0, atol=0)


def test_auxiliary_cursor_graph_staging_and_padding():
    from lmdeploy.pytorch.compact_state import SSMStateCursor
    from lmdeploy.pytorch.kernels.cuda.compact_state import stage_compact_state_ids
    cursor = SSMStateCursor(8, device='cuda')
    slots = torch.tensor([4, 1], device='cuda')
    state = SSMStateMetadata(torch.empty(2, 0, device='cuda', dtype=torch.long),
                             torch.full((2,), -1, device='cuda'), torch.full((2, 1), -1, device='cuda'))
    bound = cursor.bind(state, slots, torch.tensor([0, 64], device='cuda'), torch.tensor([3, 7], device='cuda'))
    assert bound.input_state_ids.tolist() == [-1, 7]
    state = cursor.advance(bound, slots, torch.tensor([1, 1], device='cuda'), width=6)
    static = SSMStateMetadata(torch.empty(4, 0, device='cuda', dtype=torch.long),
                              torch.zeros(4, device='cuda', dtype=torch.long),
                              torch.zeros(4, 6, device='cuda', dtype=torch.long))
    stage_compact_state_ids(state, static)
    assert static.input_state_ids.tolist() == [-1]*4
    assert static.output_state_ids.tolist() == [[-1]*6]*4
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage_compact_state_ids(state, static)
    state.output_state_ids[1, 5] = 8
    graph.replay()
    assert static.output_state_ids[1, 5] == 8
    assert static.output_state_ids[2:].eq(-1).all()
