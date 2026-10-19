# Copyright (c) OpenMMLab. All rights reserved.
"""Persistent compact metadata follows the real AR/spec input transitions."""
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmdeploy.pytorch.compact_state import CompactStateInputs, CompactStatePrefill, CompactStateRuntime
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta
from lmdeploy.pytorch.strategies.ar.model_inputs import (
    get_model_inputs_next_decoding,
    index_select_model_inputs,
    merge_model_inputs,
)
from lmdeploy.pytorch.strategies.ar_spec.step_inputs import ARSpecStepInputs, _reindex_model_inputs_arspec


def _inputs(batch=3, width=4, prefill=False, base=10):
    banks = torch.arange(base, base + batch * (width + 1), device='cuda').reshape(batch, width + 1)
    ids = CompactStateInputs.plan(banks, torch.zeros(batch, device='cuda', dtype=torch.long), 1 if prefill else width)
    lengths = torch.tensor([72] * batch if prefill else [width] * batch, device='cuda')
    return ModelInputs(input_ids=torch.arange(int(lengths.sum()), device='cuda')[None], seq_length=lengths,
                       history_lengths=torch.full((batch,), 2048, device='cuda'),
                       block_offsets=torch.zeros(batch, 2, device='cuda', dtype=torch.long),
                       is_decoding=not prefill, num_ignored_history=torch.zeros_like(lengths),
                       max_q_seqlen=72 if prefill else width, max_kv_seqlen=2120,
                       sum_kv_seqlen=batch * 2120, state_offsets=torch.arange(base, base + batch, device='cuda') + 1000,
                       compact_state=ids)


@pytest.mark.parametrize('width', [1, 2, 4, 6])
def test_prefill_to_decode_chooses_final_not_token_count(width):
    prefill = _inputs(width=width, prefill=True)
    old = prefill.compact_state
    decoding = get_model_inputs_next_decoding(prefill, torch.zeros(1, 3 * width, device='cuda', dtype=torch.long),
                                               max_q_seqlen=width, model_metas=None)
    torch.testing.assert_close(decoding.compact_state.input_ids, old.output_ids[:, 0], rtol=0, atol=0)
    assert decoding.compact_state.output_ids.shape == (3, width)
    assert decoding.history_lengths.tolist() == [2120] * 3
    torch.testing.assert_close(decoding.state_offsets, prefill.state_offsets, rtol=0, atol=0)
    assert prefill.compact_state is old and old.output_ids.shape == (3, 1)


@pytest.mark.parametrize('width', [1, 2, 4, 6])
@pytest.mark.parametrize('retain_runtime', [False, True])
def test_actual_spec_step_uses_consumed_count_including_target_token(monkeypatch, width, retain_runtime):
    from lmdeploy.pytorch.strategies.ar_spec import step_inputs as module
    monkeypatch.setattr(module, 'step_sampling_delta', lambda delta, next_tokens: delta)
    inputs = _inputs(width=width)
    original = inputs.compact_state
    # All rejected draft tokens still consume the initial target token.
    rejected = torch.tensor([0, width - 1, (width - 1) // 2], device='cuda')
    extras = SimpleNamespace(num_rejected_tokens=rejected,
                             output_draft_token_ids=torch.zeros(3, width - 1, device='cuda', dtype=torch.long))
    runtime = CompactStateRuntime(1100) if retain_runtime else None
    steps = ARSpecStepInputs(_num_spec_tokens=width - 1, compact_runtime=runtime)
    steps.step_decode(inputs, extras, SimpleNamespace(clone=lambda: None), None,
                      torch.zeros(3, device='cuda', dtype=torch.long), None, None)
    next_inputs = steps.model_inputs
    consumed = inputs.seq_length - rejected
    expected = original.output_ids[torch.arange(3, device='cuda'), consumed - 1]
    torch.testing.assert_close(next_inputs.compact_state.input_ids, expected, rtol=0, atol=0)
    torch.testing.assert_close(next_inputs.history_lengths, inputs.history_lengths + consumed, rtol=0, atol=0)
    assert consumed[1] == 1
    assert next_inputs.compact_state.input_ids[1] == original.output_ids[1, 0]
    torch.testing.assert_close(next_inputs.state_offsets, inputs.state_offsets, rtol=0, atol=0)
    assert inputs.compact_state is original
    if runtime is not None:
        torch.testing.assert_close(runtime.current_ids[inputs.state_offsets], expected, rtol=0, atol=0)


@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('partial_bank,other_partial', [(False, False), (True, True), (False, True), (True, False)])
def test_merge_reindex_clone_transfer_keep_unit_and_auxiliary_identities(width, partial_bank, other_partial):
    inputs = _inputs(batch=3, width=width)
    other = _inputs(batch=2, width=width, base=100)
    inputs.compact_state = replace(inputs.compact_state, partial_bank=partial_bank)
    other.compact_state = replace(other.compact_state, partial_bank=other_partial)
    merged = merge_model_inputs(inputs, other)
    assert merged.compact_state.bank_ids.shape == (5, width + 1)
    assert merged.compact_state.partial_bank == (partial_bank or other_partial)
    expected_aux = torch.cat((inputs.state_offsets, other.state_offsets))
    torch.testing.assert_close(merged.state_offsets, expected_aux, rtol=0, atol=0)
    indices = torch.tensor([4, 1, 0], device='cuda')
    if width == 1:
        selected = index_select_model_inputs(merged, indices)
    else:
        delta = ModelInputsDelta(indices, None, np.array([4, 1, 0]), 0, 0, 0)
        selected = _reindex_model_inputs_arspec(merged, delta, width - 1)
    for field in fields(CompactStateInputs):
        value = getattr(merged.compact_state, field.name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(getattr(selected.compact_state, field.name), value[indices], rtol=0, atol=0)
        else:
            assert getattr(selected.compact_state, field.name) == value
    torch.testing.assert_close(selected.state_offsets, expected_aux[indices], rtol=0, atol=0)
    assert selected.clone().compact_state is selected.compact_state
    moved = selected.to_device('cpu').to_device('cuda')
    for field in fields(CompactStateInputs):
        torch.testing.assert_close(getattr(moved.compact_state, field.name),
                                   getattr(selected.compact_state, field.name), rtol=0, atol=0)
    moved.record_stream(torch.cuda.current_stream())
    with pytest.raises(ValueError, match='compact and legacy'):
        merge_model_inputs(inputs, other.clone(compact_state=None))
    # One-shot checkpoints are intentionally absent from persistent metadata.
    assert {f.name for f in fields(CompactStateInputs)} == {
        'bank_ids', 'input_ids', 'output_ids', 'partial_bank', 'allow_input_alias'
    }


@pytest.mark.parametrize('prefill', [False, True])
def test_step_context_retains_compact_table_and_auxiliary_ids(monkeypatch, prefill):
    from lmdeploy.pytorch import model_inputs as module
    inputs = _inputs(width=4, prefill=prefill)
    # Backend dispatch is a later integration stage; check the metadata arriving
    # at that boundary rather than pretending legacy kernels support this pool.
    seen = []

    def update(context):
        seen.append(context)
        return context

    monkeypatch.setattr(module, 'get_backend', lambda: SimpleNamespace(update_step_context=update))
    plan = CompactStatePrefill.from_lengths([72] * 3, [2048] * 3).to_device('cuda') if prefill else None
    context = module.StepContext.new(inputs, model_config=None, cache_config=SimpleNamespace(window_size=0),
                                     compact_prefill=plan)
    assert seen == [context]
    assert context.compact_prefill is plan
    assert not hasattr(inputs, 'compact_prefill')
    assert context.compact_state is inputs.compact_state
    assert context.state_offsets is inputs.state_offsets


@pytest.mark.parametrize('prefill', [False, True])
def test_standard_mtp_draft_does_not_inherit_target_unit_metadata(prefill):
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent
    from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs
    inputs = _inputs(width=4, prefill=prefill)
    original = inputs.compact_state
    extra = ARSpecExtraInputs(next_token_ids=torch.ones(3, device='cuda', dtype=torch.long),
                              last_token_indices=inputs.seq_length.cumsum(0) - 1)
    agent = object.__new__(SpecModelAgent)
    draft_inputs, _ = agent._prepare_inputs_from_main(inputs, extra)
    assert draft_inputs.compact_state is None
    assert draft_inputs.state_offsets is None
    assert inputs.compact_state is original


@pytest.mark.parametrize('width', [1, 4, 6])
def test_chunk_continuation_selects_final_or_empty_input_without_host_readback(monkeypatch, width):
    previous = _inputs(batch=3, width=width, prefill=True).compact_state
    # Replace all banks: previous final IDs become read-only external inputs.
    current = _inputs(batch=3, width=width, prefill=True, base=100).compact_state
    lengths = torch.tensor([0, 72, 4096], device='cuda')
    snapshots = [t.clone() for obj in (previous, current) for t in (obj.bank_ids, obj.input_ids, obj.output_ids)]

    def forbidden(*args, **kwargs):
        pytest.fail('Chunk ID selection must not read device data back to the host.')

    with monkeypatch.context() as patch:
        for name in ('item', 'tolist', 'cpu', 'numpy'):
            patch.setattr(torch.Tensor, name, forbidden)
        continued = current.continue_prefill(previous, lengths)
    expected = torch.cat((previous.input_ids[:1], previous.output_ids[1:, 0]))
    torch.testing.assert_close(continued.input_ids, expected, rtol=0, atol=0)
    torch.testing.assert_close(continued.output_ids, current.bank_ids[:, :1], rtol=0, atol=0)
    for original, saved in zip((t for obj in (previous, current)
                                for t in (obj.bank_ids, obj.input_ids, obj.output_ids)), snapshots):
        torch.testing.assert_close(original, saved, rtol=0, atol=0)


def test_chunk_continuation_rejects_decode_or_mismatched_geometry():
    current = _inputs(batch=3, width=4, prefill=True).compact_state
    for previous in (_inputs(batch=3, width=4).compact_state, _inputs(batch=1, width=4, prefill=True).compact_state):
        with pytest.raises(ValueError, match='Chunk continuation'):
            current.continue_prefill(previous, torch.ones(3, device='cuda', dtype=torch.long))
