# Copyright (c) OpenMMLab. All rights reserved.
"""Persistent compact metadata follows the real AR/spec input transitions."""
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmdeploy.pytorch.compact_state import SSMPrefillMetadata, SSMStateCursor, SSMStateMetadata
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta
from lmdeploy.pytorch.strategies.ar.model_inputs import (
    get_model_inputs_next_decoding,
    index_select_model_inputs,
    merge_model_inputs,
)
from lmdeploy.pytorch.strategies.ar_spec.step_inputs import ARSpecStepInputs, _reindex_model_inputs_arspec
from tests.pytorch.compact_state_reference import plan_inputs


def _inputs(batch=3, width=4, prefill=False, base=10):
    banks = torch.arange(base, base + batch * (width + 1), device='cuda').reshape(batch, width + 1)
    ids = plan_inputs(banks, torch.zeros(batch, device='cuda', dtype=torch.long), 1 if prefill else width)
    lengths = torch.tensor([72] * batch if prefill else [width] * batch, device='cuda')
    return ModelInputs(input_ids=torch.arange(int(lengths.sum()), device='cuda')[None], seq_length=lengths,
                       history_lengths=torch.full((batch,), 2048, device='cuda'),
                       block_offsets=torch.zeros(batch, 2, device='cuda', dtype=torch.long),
                       is_decoding=not prefill, num_ignored_history=torch.zeros_like(lengths),
                       max_q_seqlen=72 if prefill else width, max_kv_seqlen=2120,
                       sum_kv_seqlen=batch * 2120, state_offsets=torch.arange(base, base + batch, device='cuda') + 1000,
                       ssm_state_meta=ids)


@pytest.mark.parametrize('width', [1, 2, 4, 6])
def test_prefill_to_decode_chooses_final_not_token_count(width):
    prefill = _inputs(width=width, prefill=True)
    prefill.ssm_prefill_meta = SSMPrefillMetadata.from_lengths([72] * 3, [2048] * 3).to_device('cuda')
    prefill.ssm_source_state_ids = torch.zeros(3, device='cuda', dtype=torch.long)
    old = prefill.ssm_state_meta
    decoding = get_model_inputs_next_decoding(prefill, torch.zeros(1, 3 * width, device='cuda', dtype=torch.long),
                                               max_q_seqlen=width, model_metas=None,
                                               ssm_state_cursor=SSMStateCursor(1100))
    torch.testing.assert_close(decoding.ssm_state_meta.input_state_ids, old.output_state_ids[:, 0], rtol=0, atol=0)
    assert decoding.ssm_state_meta.output_state_ids.shape == (3, width)
    assert decoding.ssm_prefill_meta is decoding.ssm_source_state_ids is None
    assert decoding.history_lengths.tolist() == [2120] * 3
    torch.testing.assert_close(decoding.state_offsets, prefill.state_offsets, rtol=0, atol=0)
    assert prefill.ssm_state_meta is old and old.output_state_ids.shape == (3, 1)


@pytest.mark.parametrize('width', [1, 2, 4, 6])
def test_actual_spec_step_uses_consumed_count_including_target_token(monkeypatch, width):
    from lmdeploy.pytorch.strategies.ar_spec import step_inputs as module
    monkeypatch.setattr(module, 'step_sampling_delta', lambda delta, next_tokens: delta)
    inputs = _inputs(width=width)
    inputs.ssm_prefill_meta = SSMPrefillMetadata.from_lengths([4] * 3, [2048] * 3).to_device('cuda')
    inputs.ssm_source_state_ids = torch.zeros(3, device='cuda', dtype=torch.long)
    original = inputs.ssm_state_meta
    # All rejected draft tokens still consume the initial target token.
    rejected = torch.tensor([0, width - 1, (width - 1) // 2], device='cuda')
    extras = SimpleNamespace(num_rejected_tokens=rejected,
                             output_draft_token_ids=torch.zeros(3, width - 1, device='cuda', dtype=torch.long))
    runtime = SSMStateCursor(1100)
    steps = ARSpecStepInputs(_num_spec_tokens=width - 1, ssm_state_cursor=runtime)
    steps.step_decode(inputs, extras, SimpleNamespace(clone=lambda: None), None,
                      torch.zeros(3, device='cuda', dtype=torch.long), None, None)
    next_inputs = steps.model_inputs
    assert next_inputs.ssm_prefill_meta is next_inputs.ssm_source_state_ids is None
    consumed = inputs.seq_length - rejected
    expected = original.output_state_ids[torch.arange(3, device='cuda'), consumed - 1]
    torch.testing.assert_close(next_inputs.ssm_state_meta.input_state_ids, expected, rtol=0, atol=0)
    torch.testing.assert_close(next_inputs.history_lengths, inputs.history_lengths + consumed, rtol=0, atol=0)
    assert consumed[1] == 1
    assert next_inputs.ssm_state_meta.input_state_ids[1] == original.output_state_ids[1, 0]
    torch.testing.assert_close(next_inputs.state_offsets, inputs.state_offsets, rtol=0, atol=0)
    assert inputs.ssm_state_meta is original
    torch.testing.assert_close(runtime.current_ids[inputs.state_offsets], expected, rtol=0, atol=0)


@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('partial_bank,other_partial', [(False, False), (True, True), (False, True), (True, False)])
def test_merge_reindex_clone_transfer_keep_unit_and_auxiliary_identities(width, partial_bank, other_partial):
    inputs = _inputs(batch=3, width=width)
    other = _inputs(batch=2, width=width, base=100)
    inputs.ssm_state_meta = replace(inputs.ssm_state_meta, partial_bank=partial_bank)
    other.ssm_state_meta = replace(other.ssm_state_meta, partial_bank=other_partial)
    merged = merge_model_inputs(inputs, other)
    assert merged.ssm_state_meta.state_ids.shape == (5, width + 1)
    assert merged.ssm_state_meta.partial_bank == (partial_bank or other_partial)
    expected_aux = torch.cat((inputs.state_offsets, other.state_offsets))
    torch.testing.assert_close(merged.state_offsets, expected_aux, rtol=0, atol=0)
    indices = torch.tensor([4, 1, 0], device='cuda')
    if width == 1:
        selected = index_select_model_inputs(merged, indices)
    else:
        delta = ModelInputsDelta(indices, None, np.array([4, 1, 0]), 0, 0, 0)
        selected = _reindex_model_inputs_arspec(merged, delta, width - 1)
    for field in fields(SSMStateMetadata):
        value = getattr(merged.ssm_state_meta, field.name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(getattr(selected.ssm_state_meta, field.name), value[indices], rtol=0, atol=0)
        else:
            assert getattr(selected.ssm_state_meta, field.name) == value
    torch.testing.assert_close(selected.state_offsets, expected_aux[indices], rtol=0, atol=0)
    assert selected.clone().ssm_state_meta is selected.ssm_state_meta
    moved = selected.to_device('cpu').to_device('cuda')
    for field in fields(SSMStateMetadata):
        torch.testing.assert_close(getattr(moved.ssm_state_meta, field.name),
                                   getattr(selected.ssm_state_meta, field.name), rtol=0, atol=0)
    moved.record_stream(torch.cuda.current_stream())
    with pytest.raises(ValueError, match='compact and legacy'):
        merge_model_inputs(inputs, other.clone(ssm_state_meta=None))
    # One-shot checkpoints are intentionally absent from persistent metadata.
    assert {f.name for f in fields(SSMStateMetadata)} == {
        'state_ids', 'input_state_ids', 'output_state_ids', 'partial_bank'
    }


@pytest.mark.parametrize('prefill', [False, True])
def test_step_context_retains_compact_table_and_auxiliary_ids(monkeypatch, prefill):
    from lmdeploy.pytorch import model_inputs as module
    inputs = _inputs(width=4, prefill=prefill)
    # Isolate the metadata boundary; backend execution is covered separately.
    seen = []

    def update(context):
        seen.append(context)
        return context

    monkeypatch.setattr(module, 'get_backend', lambda: SimpleNamespace(update_step_context=update))
    plan = SSMPrefillMetadata.from_lengths([72] * 3, [2048] * 3).to_device('cuda') if prefill else None
    inputs.ssm_prefill_meta = plan
    context = module.StepContext.new(inputs, model_config=None, cache_config=SimpleNamespace(window_size=0))
    assert seen == [context]
    assert context.ssm_prefill_meta is plan
    assert inputs.ssm_prefill_meta is plan
    assert context.ssm_state_meta is inputs.ssm_state_meta
    assert context.state_offsets is inputs.state_offsets


@pytest.mark.parametrize('prefill', [False, True])
def test_standard_mtp_draft_does_not_inherit_target_unit_metadata(prefill):
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent
    from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs
    inputs = _inputs(width=4, prefill=prefill)
    inputs.ssm_prefill_meta = SSMPrefillMetadata.from_lengths([inputs.max_q_seqlen] * 3, [2048] * 3).to_device('cuda')
    inputs.ssm_source_state_ids = torch.zeros(3, device='cuda', dtype=torch.long)
    original = inputs.ssm_state_meta
    extra = ARSpecExtraInputs(next_token_ids=torch.ones(3, device='cuda', dtype=torch.long),
                              last_token_indices=inputs.seq_length.cumsum(0) - 1)
    agent = object.__new__(SpecModelAgent)
    draft_inputs, _ = agent._prepare_inputs_from_main(inputs, extra)
    assert draft_inputs.ssm_prefill_meta is draft_inputs.ssm_source_state_ids is None
    assert draft_inputs.ssm_state_meta is None
    assert draft_inputs.state_offsets is None
    assert inputs.ssm_state_meta is original


@pytest.mark.parametrize('width', [1, 4, 6])
def test_chunk_continuation_selects_final_or_empty_input_without_host_readback(monkeypatch, width):
    previous = _inputs(batch=3, width=width, prefill=True).ssm_state_meta
    # Replace all banks: previous final IDs become read-only external inputs.
    current = _inputs(batch=3, width=width, prefill=True, base=100).ssm_state_meta
    lengths = torch.tensor([0, 72, 4096], device='cuda')
    snapshots = [t.clone() for obj in (previous, current)
                 for t in (obj.state_ids, obj.input_state_ids, obj.output_state_ids)]
    cursor = SSMStateCursor(4)
    requests = torch.arange(1, 4, device='cuda')

    def forbidden(*args, **kwargs):
        pytest.fail('Chunk ID selection must not read device data back to the host.')

    with monkeypatch.context() as patch:
        for name in ('item', 'tolist', 'cpu', 'numpy'):
            patch.setattr(torch.Tensor, name, forbidden)
        cursor.advance(previous, requests, (lengths > 0).long())
        continued = cursor.bind(current, requests, lengths)
    expected = torch.cat((previous.input_state_ids[:1], previous.output_state_ids[1:, 0]))
    torch.testing.assert_close(continued.input_state_ids, expected, rtol=0, atol=0)
    torch.testing.assert_close(continued.output_state_ids, current.state_ids[:, :1], rtol=0, atol=0)
    for original, saved in zip((t for obj in (previous, current)
                                for t in (obj.state_ids, obj.input_state_ids, obj.output_state_ids)), snapshots):
        torch.testing.assert_close(original, saved, rtol=0, atol=0)


@pytest.mark.parametrize('transition', ['step', 'prefill_to_decode'])
def test_compact_transition_requires_cursor(transition):
    inputs = _inputs(width=4, prefill=transition == 'prefill_to_decode')
    with pytest.raises(ValueError, match='requires the worker cursor'):
        if transition == 'step':
            inputs.step(inputs.input_ids)
        else:
            get_model_inputs_next_decoding(inputs, inputs.input_ids, 4, None)


def test_cursor_rebind_rejects_changed_bank_geometry():
    cursor = SSMStateCursor(4)
    current = _inputs(batch=3, width=4).ssm_state_meta
    for bank in (current.state_ids[:1], current.state_ids[:, :1].contiguous()):
        with pytest.raises(ValueError, match='bank capacity or batch rows'):
            cursor.rebind(current, bank)


def test_one_forward_model_metadata_pickle_transfer_and_stream(monkeypatch):
    import pickle
    inputs = _inputs(prefill=True)
    inputs.ssm_prefill_meta = SSMPrefillMetadata.from_lengths([72] * 3, [2048] * 3,
        checkpoint_ids=[7, 8, 9], checkpoint_offsets=[64] * 3).to_device('cuda')
    inputs.ssm_source_state_ids = torch.tensor([-1, 0, 7], device='cuda')
    restored = pickle.loads(pickle.dumps(inputs.to_device('cpu')))
    assert restored.ssm_prefill_meta.checkpoint_ids.tolist() == [7, 8, 9]
    assert restored.ssm_prefill_meta.checkpoint_offsets.tolist() == [64] * 3
    assert restored.ssm_source_state_ids.tolist() == [-1, 0, 7]
    moved = restored.to_device('meta')
    assert moved.ssm_prefill_meta.cu_seqlens.device.type == 'meta'
    assert moved.ssm_prefill_meta.checkpoint_ids.device.type == 'meta'
    assert moved.ssm_source_state_ids.device.type == 'meta'
    recorded = []
    monkeypatch.setattr(torch.Tensor, 'record_stream', lambda self, stream: recorded.append(id(self)))
    inputs.record_stream(object())
    assert id(inputs.ssm_source_state_ids) in recorded
    for f in fields(inputs.ssm_prefill_meta):
        tensor = getattr(inputs.ssm_prefill_meta, f.name)
        if tensor is not None:
            assert id(tensor) in recorded
    for invalid in (inputs.ssm_source_state_ids.int(), inputs.ssm_source_state_ids[None],
                    inputs.ssm_source_state_ids[::2]):
        with pytest.raises(ValueError, match='contiguous per-request int64'):
            inputs.clone(ssm_source_state_ids=invalid)


def test_compact_access_ticket_still_excludes_legacy_copy():
    from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs
    for field in ('state_restore_plan', 'state_save_plan'):
        with pytest.raises(ValueError, match='legacy state-copy'):
            CacheCheckpointInputs(ssm_state_access_id=1, **{field: ((1,), (2,))})


@pytest.mark.parametrize('width,checkpoint', [(1, False), (1, True), (6, False)])
def test_step_inputs_apply_delta_reindexes_then_rebinds(monkeypatch, width, checkpoint):
    from lmdeploy.pytorch.strategies.ar import step_inputs as ar
    from lmdeploy.pytorch.strategies.ar_spec import step_inputs as spec

    for module in (ar, spec):
        monkeypatch.setattr(module, 'reindex_sampling_delta', lambda sampling, delta: sampling)
    inputs = _inputs(width=width)
    inputs.ssm_state_meta = replace(inputs.ssm_state_meta, partial_bank=True)
    old = inputs.ssm_state_meta
    cursor = SSMStateCursor(1100)
    steps = (ar.ARStepInputs(ssm_state_cursor=cursor) if width == 1 else
             spec.ARSpecStepInputs(_num_spec_tokens=width - 1, ssm_state_cursor=cursor))
    steps.model_inputs = inputs
    steps.stopping_criteria = SimpleNamespace(update=lambda delta: None)
    steps.extra_inputs = SimpleNamespace(update=lambda delta: None)
    indices = torch.tensor([2, 0], device='cuda')
    bank = (old.state_ids[indices] + 50).contiguous()
    destinations = torch.tensor([150, 151], device='cuda') if checkpoint else None
    delta = ModelInputsDelta(indices, inputs.block_offsets[indices], np.array([2, 0]), width, 2120, 4240,
                             ssm_state_ids=bank, ssm_checkpoint_state_ids=destinations)
    # Decode destinations belong to this delta, not retained model state.
    steps.apply_delta(delta)
    current = steps.model_inputs.ssm_state_meta
    assert not current.partial_bank and old.partial_bank
    torch.testing.assert_close(current.input_state_ids, old.input_state_ids[indices], rtol=0, atol=0)
    torch.testing.assert_close(current.state_ids, bank, rtol=0, atol=0)
    torch.testing.assert_close(steps.model_inputs.state_offsets, inputs.state_offsets[indices], rtol=0, atol=0)
    assert steps.model_inputs.ssm_prefill_meta is steps.model_inputs.ssm_source_state_ids is None
    assert torch.all(cursor.current_ids == -1), 'rebind must not advance the current-state ledger'
    if checkpoint:
        torch.testing.assert_close(current.output_state_ids[:, 0], destinations, rtol=0, atol=0)
    else:
        assert torch.isin(current.output_state_ids, bank).all()
    assert inputs.ssm_state_meta is old


def test_delta_decode_destinations_pickle_device_and_stream(monkeypatch):
    import pickle
    delta = ModelInputsDelta(torch.tensor([0]), torch.zeros(1, 1, dtype=torch.long), np.array([0]), 1, 64, 64,
                             ssm_state_ids=torch.tensor([[10]]), ssm_checkpoint_state_ids=torch.tensor([7]))
    restored = pickle.loads(pickle.dumps(delta))
    assert restored.ssm_checkpoint_state_ids.tolist() == [7]
    assert restored.to_device('meta').ssm_checkpoint_state_ids.device.type == 'meta'
    gpu = restored.to_device('cuda')
    recorded = []
    monkeypatch.setattr(torch.Tensor, 'record_stream', lambda self, stream: recorded.append(id(self)))
    gpu.record_stream(object())
    assert id(gpu.ssm_checkpoint_state_ids) in recorded
    ids = torch.tensor([7, 8, 9])
    for invalid in (ids.int(), ids[None], ids[::2]):
        with pytest.raises(ValueError, match='contiguous per-request int64'):
            replace(delta, ssm_checkpoint_state_ids=invalid)


def test_mtp_proposer_drops_target_one_forward_metadata():
    from lmdeploy.pytorch.spec_decode.proposers.base import BaseSpecProposer

    inputs = _inputs(prefill=True)
    plan = SSMPrefillMetadata.from_lengths([inputs.max_q_seqlen] * 3, [2048] * 3).to_device('cuda')
    inputs.ssm_prefill_meta = plan
    inputs.ssm_source_state_ids = torch.zeros(3, device='cuda', dtype=torch.long)
    tokens = torch.zeros(3, device='cuda', dtype=torch.long)
    output = BaseSpecProposer.update_inputs_decoding(
        object(), inputs, SimpleNamespace(num_rejected_tokens=None), tokens[None], None, None)
    assert output.ssm_prefill_meta is output.ssm_source_state_ids is None
    assert inputs.ssm_prefill_meta is plan and inputs.ssm_source_state_ids is not None


@pytest.mark.parametrize('path', ['query', 'prefill', 'decode'])
def test_dflash_supported_inputs_do_not_carry_compact_metadata(path):
    from lmdeploy.pytorch.spec_decode.proposers.dflash import DFlash

    # Compact+DFlash is rejected at executor construction; do not invent a
    # supported producer of compact metadata in this ordinary DFlash test.
    inputs = _inputs(prefill=path != 'decode')
    inputs.ssm_state_meta = None
    assert inputs.ssm_prefill_meta is inputs.ssm_source_state_ids is None
    tokens = torch.zeros(3, device='cuda', dtype=torch.long)
    if path == 'query':
        proposer = SimpleNamespace(num_speculative_tokens=3, specdecode_config=SimpleNamespace(mask_token_id=15))
        output = DFlash._build_query_inputs(proposer, inputs, inputs.seq_length, tokens)
    else:
        proposer = SimpleNamespace(_flatten_target_hidden=lambda extra: torch.empty(0, device='cuda'),
                                   _context_lengths=lambda inputs, extra: inputs.seq_length,
                                   _query_start_positions=lambda inputs, lengths: inputs.history_lengths + lengths)
        output, *_ = DFlash._prepare_context_materialization(proposer, inputs, None)
    assert output.ssm_state_meta is output.ssm_prefill_meta is output.ssm_source_state_ids is None
    assert inputs.ssm_prefill_meta is inputs.ssm_source_state_ids is None
