# Copyright (c) OpenMMLab. All rights reserved.
"""Real CUDA backend dispatch, not scheduler/serving or model-forward proof."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.compact_state import SSMPrefillMetadata
from tests.pytorch.compact_state_reference import plan_inputs


def test_host_prefill_plan_rejects_unsupported_boundary_before_dispatch():
    plan = SSMPrefillMetadata.from_lengths([0, 1, 72, 137], [0, 0, 2048, 128],
                                           checkpoint_ids=[-1, 4, 5, 6], checkpoint_offsets=[0, 1, 64, 128])
    assert plan.cu_seqlens.tolist() == [0, 0, 1, 73, 210]
    assert plan.chunk_indices.tolist() == [[1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2]]
    assert plan.chunk_offsets.tolist() == [0, 0, 1, 3, 6]
    assert plan.seq_idx.tolist() == [1] + [2] * 72 + [3] * 137
    assert plan.is_init_token.nonzero().flatten().tolist() == [0]
    with pytest.raises(ValueError, match='chunk boundary'):
        SSMPrefillMetadata.from_lengths([72], [2048], checkpoint_ids=[10], checkpoint_offsets=[56])
    assert SSMPrefillMetadata.from_lengths([], []).chunk_indices.shape == (0, 2)


def _case(monkeypatch, family, lengths, histories, width, checkpoint=False):
    from lmdeploy.pytorch.backends import gated_delta_rule as base
    from lmdeploy.pytorch.backends.cuda import gated_delta_rule as cuda
    from lmdeploy.pytorch.model_inputs import BuildModelContext, StepContextManager
    manager = StepContextManager(BuildModelContext(num_spec_tokens=width - 1))
    monkeypatch.setattr(base, 'get_step_ctx_manager', lambda: manager)
    monkeypatch.setattr(cuda, 'get_step_ctx_manager', lambda: manager)
    batch, total, dim = len(lengths), sum(lengths), 128
    prefill = lengths != [width] * batch
    # All tests include unit0 and two shared read-only inputs.
    banks = torch.arange(10, 10 + batch * (width + 1), device='cuda').reshape(batch, width + 1)
    ids = plan_inputs(banks, torch.tensor([0, 1, 1], device='cuda'), 1 if prefill else width)
    plan = None
    if prefill:
        plan = SSMPrefillMetadata.from_lengths(lengths, histories,
            checkpoint_ids=[50, 51, 52] if checkpoint else None,
            checkpoint_offsets=[min(64, n) for n in lengths] if checkpoint else None).to_device('cuda')
    qlen = torch.tensor(lengths, device='cuda')
    attn = SimpleNamespace(is_decoding=not prefill, q_seqlens=qlen,
                           kv_seqlens=qlen + torch.tensor(histories, device='cuda'),
                           cu_seqlens_q=torch.nn.functional.pad(qlen.cumsum(0), (1, 0)))
    legacy_ids = torch.arange(1, batch + 1, device='cuda')
    legacy = cuda.GatedDeltaMeta(total, 4, legacy_ids, attn)
    with manager.context(SimpleNamespace(ssm_state_meta=ids, ssm_prefill_meta=plan)):
        meta = cuda.CudaGatedDeltaMetaImpl(indexed_state=True).forward(total, 4, legacy_ids + 1000, attn)
    assert not hasattr(meta, 'conv_idx')  # legacy gather/offset setup was not executed
    assert meta.ssm_state_meta is ids
    channels = (4 if family == 'gdn' else 6) * dim
    heads = 2
    dtype = torch.bfloat16 if family == 'gdn' else torch.float32
    conv = torch.randn(64, channels, 3, device='cuda', dtype=torch.bfloat16) / 8
    rec = torch.randn(64, heads, dim, dim, device='cuda', dtype=dtype) / 16
    conv[0].zero_()
    rec[0].zero_()
    old_conv = torch.zeros(batch + 1, channels, 4 + width - 1, device='cuda', dtype=conv.dtype)
    old_rec = torch.zeros(batch + 1, width, heads, dim, dim, device='cuda', dtype=dtype)
    for row, (source, history) in enumerate(zip([0, 1, 1], histories), 1):
        offsets = ((history + torch.arange(-3, 0, device='cuda')) % old_conv.size(-1)
                   if width > 1 else torch.arange(1, 4, device='cuda'))
        old_conv[row, :, offsets] = conv[source]
        old_rec[row, history % width] = rec[source]
    if width == 1:
        old_rec = old_rec[:, 0]
    weight = torch.randn(channels, 1, 4, device='cuda', dtype=conv.dtype) / 8
    bias = torch.randn(channels, device='cuda', dtype=conv.dtype) / 16
    x = torch.randn(1, total, channels, device='cuda', dtype=conv.dtype) / 8
    if family == 'gdn':
        from lmdeploy.pytorch.backends.cuda.causal_conv1d import CausalConv1dTilelangImpl
        conv_impl = CausalConv1dTilelangImpl(indexed_state=True)
        legacy_conv = CausalConv1dTilelangImpl()
        rule = cuda.CudaGatedDeltaRuleImpl(indexed_state=True)
        legacy_rule = cuda.CudaGatedDeltaRuleImpl()
        a, b = [torch.randn(1, total, heads, device='cuda', dtype=x.dtype) for _ in range(2)]
        dt_bias = torch.zeros(heads, device='cuda')
        a_log_exp = -torch.ones(heads, device='cuda')

        def run(meta, conv, rec, selection=slice(None)):
            conv_op = conv_impl if hasattr(meta, 'ssm_state_meta') else legacy_conv
            recurrent_op = rule if hasattr(meta, 'ssm_state_meta') else legacy_rule
            mixed = conv_op.forward(x[:, selection].contiguous(), weight, bias, conv, meta, 'silu')
            q, k, v = mixed.split([dim, dim, heads * dim], dim=-1)
            return recurrent_op.forward(q.unflatten(-1, (1, dim)), k.unflatten(-1, (1, dim)),
                                v.unflatten(-1, (heads, dim)), b[:, selection].contiguous(),
                                a[:, selection].contiguous(), dt_bias, a_log_exp,
                                rec, meta, 2, True)
    else:
        from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl
        rule = CudaKdaImpl(indexed_state=True)
        from tests.pytorch.legacy_kda_reference import LegacyKdaReference
        legacy_rule = LegacyKdaReference()
        gate = torch.randn(1, total, heads * dim, device='cuda', dtype=x.dtype)
        beta = torch.randn(1, total, heads, device='cuda', dtype=x.dtype)
        a_log = torch.zeros(heads, device='cuda')
        dt_bias = torch.zeros(heads * dim, device='cuda')

        def run(meta, conv, rec, selection=slice(None)):
            recurrent_op = rule if hasattr(meta, 'ssm_state_meta') else legacy_rule
            return recurrent_op.forward(x[:, selection].contiguous(), gate[:, selection].contiguous(),
                                beta[:, selection].contiguous(), weight, bias, a_log, dt_bias,
                                conv, rec, meta, heads, dim, -5.)
    return run, meta, legacy, conv, rec, old_conv, old_rec


@pytest.mark.parametrize('family', ['gdn', 'kda'])
@pytest.mark.parametrize('width', [1, 4, 6])
def test_indexed_backend_decode_matches_legacy_and_preserves_inputs(monkeypatch, family, width):
    torch.manual_seed(451)
    histories = [0, 2048, 71]
    run, meta, legacy, conv, rec, old_conv, old_rec = _case(monkeypatch, family, [width] * 3, histories, width)
    before_conv, before_rec = conv.clone(), rec.clone()
    expected = run(legacy, old_conv, old_rec)
    actual = run(meta, conv, rec)
    # KDA AR legacy uses FLA conv while indexed/MTP use TileLang; retain the
    # predeclared BF16 output tolerance, not a tolerance chosen after results.
    torch.testing.assert_close(actual, expected, rtol=0 if family == 'gdn' else .02,
                               atol=0 if family == 'gdn' else .02)
    for row, history in enumerate(histories):
        for step, unit in enumerate(meta.ssm_state_meta.output_state_ids[row].tolist(), 1):
            expected_state = old_rec[row + 1, (history + step) % width] if width > 1 else old_rec[row + 1]
            torch.testing.assert_close(rec[unit], expected_state, rtol=0 if family == 'gdn' else 1e-4,
                                       atol=0 if family == 'gdn' else 1e-4)
    untouched = torch.ones(len(conv), device='cuda', dtype=torch.bool)
    untouched[meta.ssm_state_meta.output_state_ids.flatten()] = False
    torch.testing.assert_close(conv[untouched], before_conv[untouched], rtol=0, atol=0)
    torch.testing.assert_close(rec[untouched], before_rec[untouched], rtol=0, atol=0)


@pytest.mark.parametrize('family', ['gdn', 'kda'])
@pytest.mark.parametrize('width', [1, 4])
def test_packed_backend_prefill_checkpoint_is_observational(monkeypatch, family, width):
    torch.manual_seed(452)
    lengths, histories = [1, 72, 137], [0, 2048, 128]
    run, meta, legacy, conv, rec, old_conv, old_rec = _case(monkeypatch, family, lengths, histories, width, True)
    before_conv, before_rec = conv.clone(), rec.clone()
    prefix_conv, prefix_rec = old_conv.clone(), old_rec.clone()
    expected = run(legacy, old_conv, old_rec)
    actual = run(meta, conv, rec)
    torch.testing.assert_close(actual, expected, rtol=0 if family == 'gdn' else .02,
                               atol=0 if family == 'gdn' else .02)
    final_ids = meta.ssm_state_meta.output_state_ids[:, 0]
    for row, (history, length, unit) in enumerate(zip(histories, lengths, final_ids.tolist())):
        expected_state = old_rec[row + 1, (history + length) % width] if width > 1 else old_rec[row + 1]
        torch.testing.assert_close(rec[unit], expected_state, rtol=0 if family == 'gdn' else 1e-4,
                                   atol=0 if family == 'gdn' else 1e-4)
        offsets = ((history + length + torch.arange(-3, 0, device='cuda')) % old_conv.size(-1)
                   if width > 1 else torch.arange(1, 4, device='cuda'))
        torch.testing.assert_close(conv[unit], old_conv[row + 1, :, offsets], rtol=0, atol=0)
    checkpoint_bytes = conv[[50, 51, 52]].clone(), rec[[50, 51, 52]].clone()
    final_bytes = conv[final_ids].clone(), rec[final_ids].clone()
    no_checkpoint = replace(meta.ssm_prefill_meta, checkpoint_ids=None, checkpoint_offsets=None)
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaCompactGatedDeltaMeta
    output_off = run(CudaCompactGatedDeltaMeta(sum(lengths), False, meta.ssm_state_meta, no_checkpoint), conv, rec)
    torch.testing.assert_close(actual, output_off, rtol=0, atol=0)
    for pool, saved, final in zip((conv, rec), checkpoint_bytes, final_bytes):
        torch.testing.assert_close(pool[[50, 51, 52]], saved, rtol=0, atol=0)
        torch.testing.assert_close(pool[final_ids], final, rtol=0, atol=0)
    untouched = torch.ones(len(conv), device='cuda', dtype=torch.bool)
    untouched[torch.cat((final_ids, meta.ssm_prefill_meta.checkpoint_ids))] = False
    torch.testing.assert_close(conv[untouched], before_conv[untouched], rtol=0, atol=0)
    torch.testing.assert_close(rec[untouched], before_rec[untouched], rtol=0, atol=0)

    # Compare exported checkpoints with a legacy forward stopped at that exact
    # boundary, using the original inputs/weights and original initial states.
    from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaMeta
    start = 0
    for row, (length, history) in enumerate(zip(lengths, histories)):
        offset = min(64, length)
        qlen = torch.tensor([offset], device='cuda')
        prefix_meta = GatedDeltaMeta(offset, 4, torch.tensor([row + 1], device='cuda'), SimpleNamespace(
            is_decoding=False, q_seqlens=qlen, kv_seqlens=qlen + history,
            cu_seqlens_q=torch.tensor([0, offset], device='cuda')))
        run(prefix_meta, prefix_conv, prefix_rec, slice(start, start + offset))
        state = prefix_rec[row + 1, (history + offset) % width] if width > 1 else prefix_rec[row + 1]
        torch.testing.assert_close(rec[50 + row], state, rtol=0 if family == 'gdn' else 1e-4,
                                   atol=0 if family == 'gdn' else 1e-4)
        positions = ((history + offset + torch.arange(-3, 0, device='cuda')) % prefix_conv.size(-1)
                     if width > 1 else torch.arange(1, 4, device='cuda'))
        torch.testing.assert_close(conv[50 + row], prefix_conv[row + 1, :, positions], rtol=0, atol=0)
        start += length


def test_compact_updater_skips_fla_device_length_readback(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import gated_delta_rule as cuda

    def forbidden(*args):
        raise AssertionError('Compact metadata must not prepare FLA chunks from device lengths.')

    monkeypatch.setattr(cuda, 'prepare_chunked_gated_delta_rule', forbidden)
    updater = cuda.GatedDeltaStepMetaUpdater(indexed_state=True)
    updater.update(SimpleNamespace(is_decoding=False, ssm_state_meta=object(), ssm_prefill_meta=object()), None)
    updater.update(SimpleNamespace(is_decoding=True, ssm_state_meta=object(), ssm_prefill_meta=None), None)
    with pytest.raises(ValueError, match='prebuilt'):
        updater.update(SimpleNamespace(is_decoding=False, ssm_state_meta=object(), ssm_prefill_meta=None), None)


@pytest.mark.parametrize('family', ['gdn', 'kda'])
@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('foreign_capture_stream', [False, True])
def test_compact_backend_decode_graph_replays_live_id_buffers(monkeypatch, family, width, foreign_capture_stream):
    if foreign_capture_stream:
        if torch.cuda.device_count() < 2:
            pytest.skip('foreign capture stream regression requires two GPUs')
        other_device = (torch.cuda.current_device() + 1) % torch.cuda.device_count()
        monkeypatch.setattr(torch.cuda.graph, 'default_capture_stream', torch.cuda.Stream(device=other_device))
    torch.manual_seed(453)
    run, meta, _, conv, rec, _, _ = _case(monkeypatch, family, [width] * 3, [0, 2048, 71], width)
    run(meta, conv, rec)
    graph = torch.cuda.CUDAGraph()
    # PyTorch's cached capture stream may belong to a previous test's device.
    with torch.cuda.graph(graph, stream=torch.cuda.Stream(device=conv.device)):
        actual = run(meta, conv, rec)
    for sources in ([1, 0, 1], [0, 1, 1]):
        # Standalone graph staging buffers, updated only between completed
        # replays. This is not permission to modify owner-submitted snapshots.
        meta.ssm_state_meta.input_state_ids.copy_(torch.tensor(sources, device='cuda'))
        graph.replay()
        saved = [pool[meta.ssm_state_meta.output_state_ids].clone() for pool in (conv, rec)]
        expected = run(meta, conv, rec)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for pool, values in zip((conv, rec), saved):
            torch.testing.assert_close(pool[meta.ssm_state_meta.output_state_ids], values, rtol=0, atol=0)


@pytest.mark.parametrize('family', ['gdn', 'kda'])
def test_warmed_compact_prefill_has_no_state_gather_scatter_or_readback(monkeypatch, family):
    import os
    from pathlib import Path

    run, meta, _, conv, rec, _, _ = _case(monkeypatch, family, [1, 72, 137], [0, 2048, 128], 4, True)
    run(meta, conv, rec)
    torch.cuda.synchronize()

    def forbidden(*args, **kwargs):
        raise AssertionError('Warmed indexed backend must not read device values on the host.')

    with monkeypatch.context() as local:
        for method in ('cpu', 'item', 'tolist'):
            local.setattr(torch.Tensor, method, forbidden)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as trace:
            run(meta, conv, rec)
    forbidden_ops = {'aten::index_select', 'aten::index_copy_', 'aten::index_put_'}
    assert not forbidden_ops.intersection(event.key for event in trace.key_averages())
    # Activation contiguous() copies are expected for split projection views;
    # the pool itself must not be cloned/converted by these backend adapters.
    for event in trace.events():
        if event.name in {'aten::clone', 'aten::contiguous', 'aten::copy_', 'aten::_to_copy'}:
            for shape in event.input_shapes:
                if len(shape) >= 3:
                    assert tuple(shape[-2:]) not in (tuple(conv.shape[-2:]), tuple(rec.shape[-2:]))
    if os.environ.get('COMPACT_BACKEND_TRACE_DIR'):
        trace.export_chrome_trace(str(Path(os.environ['COMPACT_BACKEND_TRACE_DIR']) / f'{family}-prefill-trace.json'))


@pytest.mark.parametrize('family', ['gdn', 'kda'])
@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('prefill', [False, True])
@pytest.mark.parametrize('graph', [False, True])
@torch.inference_mode()
def test_exclusive_backend_dispatch_matches_disjoint_full_outputs(monkeypatch, family, width, prefill, graph):
    """Exercise real backend flag forwarding, not just primitive entry
    points."""
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaCompactGatedDeltaMeta

    torch.manual_seed(459)
    lengths = [1, 72, 137] if prefill else [width] * 3
    run, control, _, conv, rec, _, _ = _case(monkeypatch, family, lengths, [0, 2048, 128], width, prefill)
    # Private input belongs to its own minimal W-wide ring; checkpoint outputs
    # and the old shared control input stay outside all writable banks.
    banks = torch.arange(10, 10 + 3 * width, device='cuda').reshape(3, width)
    ids = plan_inputs(banks, banks[:, 0].contiguous(), 1 if prefill else width)
    alias_meta = CudaCompactGatedDeltaMeta(sum(lengths), not prefill, ids, control.ssm_prefill_meta)
    alias_conv, alias_rec = conv.clone(), rec.clone()
    for pool, original in ((alias_conv, conv), (alias_rec, rec)):
        pool[banks[:, 0]] = original[control.ssm_state_meta.input_state_ids]
    before = alias_conv.clone(), alias_rec.clone()
    expected = run(control, conv, rec)
    actual = run(alias_meta, alias_conv, alias_rec)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    captured = None
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=torch.cuda.Stream()):
            actual = run(alias_meta, alias_conv, alias_rec)
    for _ in range(2):
        alias_conv.copy_(before[0])
        alias_rec.copy_(before[1])
        if captured is not None:
            captured.replay()
        else:
            actual = run(alias_meta, alias_conv, alias_rec)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        writes = ids.output_state_ids.flatten()
        if prefill:
            writes = torch.cat((writes, alias_meta.ssm_prefill_meta.checkpoint_ids))
        untouched = torch.ones(len(conv), device='cuda', dtype=torch.bool)
        untouched[writes] = False
        for candidate, original, saved in ((alias_conv, conv, before[0]), (alias_rec, rec, before[1])):
            torch.testing.assert_close(candidate[ids.output_state_ids],
                                       original[control.ssm_state_meta.output_state_ids],
                                       rtol=0, atol=0)
            torch.testing.assert_close(candidate[untouched], saved[untouched], rtol=0, atol=0)
            if prefill:
                checkpoints = alias_meta.ssm_prefill_meta.checkpoint_ids
                torch.testing.assert_close(candidate[checkpoints], original[checkpoints], rtol=0, atol=0)


def test_backend_layout_is_bound_at_construction_not_inferred_from_metadata():
    from lmdeploy.pytorch.backends.causal_conv1d import CausalConv1dBuildSpec
    from lmdeploy.pytorch.backends.cuda.causal_conv1d import CausalConv1dTilelangImpl
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl
    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaMetaBuildSpec, GatedDeltaRuleBuildSpec

    conv = CudaOpsBackend.build_op(CausalConv1dBuildSpec(indexed_state=True))
    rule = CudaOpsBackend.build_op(GatedDeltaRuleBuildSpec(indexed_state=True))
    meta = CudaOpsBackend.build_op(GatedDeltaMetaBuildSpec(indexed_state=True))
    assert conv.forward.__func__ is CausalConv1dTilelangImpl.forward
    assert conv.indexed_state and rule.indexed_state and meta.indexed_state
    assert all('forward' not in vars(impl) for impl in (conv, rule, meta))
    assert '_build' not in vars(meta)
    assert rule.forward.__func__ is CudaGatedDeltaRuleImpl.forward
    assert rule.get_step_metadata_provider().indexed_state
    assert meta._build.__name__ == '_build'
    other = CudaOpsBackend.build_op(GatedDeltaRuleBuildSpec())
    assert other.forward.__func__ is rule.forward.__func__
    assert not other.indexed_state
    assert not other.get_step_metadata_provider().indexed_state
