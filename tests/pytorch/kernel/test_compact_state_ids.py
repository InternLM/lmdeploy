# Copyright (c) OpenMMLab. All rights reserved.
"""ID-only device planning and accepted-boundary selection."""
import pytest
import torch

from lmdeploy.pytorch.compact_state import SSMStateCursor, SSMStateMetadata
from tests.pytorch.compact_state_reference import plan_inputs


@pytest.fixture
def cursor():
    return SSMStateCursor(64)


def _advance(state, consumed):
    """Exercise the raw selection kernel, independently of serving
    persistence."""
    from lmdeploy.pytorch.kernels.cuda.compact_state import select_compact_state_ids
    inputs, outputs = select_compact_state_ids(state.state_ids, state.input_state_ids, state.output_state_ids.shape[1],
                                               output_ids=state.output_state_ids, consumed=consumed)
    return SSMStateMetadata(state.state_ids, inputs, outputs)


def _tables(batch, width):
    banks = torch.arange(10, 10 + batch * (width + 1), device='cuda').reshape(batch, width + 1).flip(1).contiguous()
    inputs = torch.zeros(batch, device='cuda', dtype=torch.int64)
    return banks, inputs


@pytest.mark.parametrize('width', [1, 2, 4, 6])
@pytest.mark.parametrize('batch', [1, 7, 40])
def test_all_consumed_counts_and_shared_inputs(width, batch):
    banks, inputs = _tables(batch, width)
    inputs[::3] = 999  # shared input is not a writable member of any bank
    metadata = plan_inputs(banks, inputs, width)
    torch.testing.assert_close(metadata.output_state_ids, banks[:, :width], rtol=0, atol=0)
    original = [t.clone() for t in (banks, metadata.input_state_ids, metadata.output_state_ids)]
    for count in range(width + 1):
        counts = torch.full((batch,), count, device='cuda', dtype=torch.int32)
        following = _advance(metadata, counts)
        expected_input = inputs if count == 0 else original[2][:, count - 1]
        torch.testing.assert_close(following.input_state_ids, expected_input, rtol=0, atol=0)
        for row, (bank, current, outputs) in enumerate(zip(banks.tolist(), expected_input.tolist(),
                                                         following.output_state_ids.tolist())):
            assert len(set(outputs)) == width
            assert current not in outputs
            assert set(outputs) <= set(bank)
            start = bank.index(current) + 1 if current in bank else 0
            assert outputs == [bank[(start + i) % len(bank)] for i in range(width)]
    for actual, expected in zip((banks, metadata.input_state_ids, metadata.output_state_ids), original):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_invalid_acceptance_and_dummy_rows_fail_closed():
    banks, inputs = _tables(5, 4)
    inputs[0] = -1
    banks[1].fill_(-1)
    metadata = plan_inputs(banks, inputs, 4)
    following = _advance(metadata, torch.tensor([0, 1, -1, 5, 4], device='cuda'))
    assert following.input_state_ids[:4].tolist() == [-1] * 4
    assert following.output_state_ids[:4].tolist() == [[-1] * 4] * 4
    assert following.input_state_ids[4] == metadata.output_state_ids[4, 3]


def test_consumed_output_cannot_select_reserved_zero_unit():
    banks, inputs = _tables(2, 4)
    metadata = SSMStateMetadata(banks, inputs, torch.zeros(2, 4, device='cuda', dtype=torch.int64))
    following = _advance(metadata, torch.tensor([0, 1], device='cuda'))
    assert following.input_state_ids.tolist() == [0, -1]
    assert following.output_state_ids[0].tolist() == banks[0, :4].tolist()
    assert following.output_state_ids[1].tolist() == [-1] * 4


def test_128_round_boundary_identity_and_eight_bank_rebindings(cursor):
    batch, width = 4, 6
    banks, inputs = _tables(batch, width)
    metadata = plan_inputs(banks, inputs, width)
    payload = torch.zeros(1024, device='cuda', dtype=torch.int64)  # test-only logical boundary oracle
    committed = torch.zeros(batch, device='cuda', dtype=torch.int64)
    frozen = {}
    for step in range(128):
        # Simulate a causal operator emitting each exact logical-time boundary;
        # do not compute its IDs through the new selection function in the oracle.
        torch.testing.assert_close(payload[metadata.input_state_ids], committed, rtol=0, atol=0)
        values = committed[:, None] + torch.arange(1, width + 1, device='cuda')
        payload[metadata.output_state_ids] = values
        counts = (torch.arange(batch, device='cuda') + step) % (width + 1)
        previous = metadata
        metadata = _advance(metadata, counts)
        committed = committed + counts
        torch.testing.assert_close(payload[metadata.input_state_ids], committed, rtol=0, atol=0)
        assert torch.all(metadata.output_state_ids != metadata.input_state_ids[:, None])
        if step % 16 == 15:
            old_id = int(metadata.input_state_ids[0])
            old_table = metadata.state_ids.clone()
            replacement = 500 + step
            frozen[old_id] = int(payload[old_id])
            new_table = metadata.state_ids.clone()
            new_table[0][new_table[0] == old_id] = replacement
            rebound = cursor.rebind(metadata, new_table)
            torch.testing.assert_close(metadata.state_ids, old_table, rtol=0, atol=0)
            assert rebound.input_state_ids[0] == old_id
            assert old_id not in rebound.output_state_ids[0].tolist()
            metadata = rebound
        for frozen_id, value in frozen.items():
            assert payload[frozen_id] == value
        assert metadata.input_state_ids.data_ptr() != previous.input_state_ids.data_ptr()
    assert len(frozen) == 8


def test_graph_replay_reads_live_counts_without_mutating_captured_bank():
    banks, inputs = _tables(3, 4)
    metadata = plan_inputs(banks, inputs, 4)
    counts = torch.zeros(3, device='cuda', dtype=torch.int64)
    _advance(metadata, counts)  # warm compilation before capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        next_metadata = _advance(metadata, counts)
    for values in ([1, 2, 4], [4, 0, 3], [0, 1, 2]):
        counts.copy_(torch.tensor(values, device='cuda'))
        graph.replay()
        expected = [int(inputs[row]) if value == 0 else int(metadata.output_state_ids[row, value - 1])
                    for row, value in enumerate(values)]
        assert next_metadata.input_state_ids.tolist() == expected
    torch.testing.assert_close(metadata.state_ids, banks, rtol=0, atol=0)


def test_metadata_shape_validation_and_empty_batch():
    banks, inputs = _tables(0, 4)
    metadata = plan_inputs(banks, inputs, 4)
    assert _advance(metadata, inputs).output_state_ids.shape == (0, 4)
    banks, inputs = _tables(2, 4)
    for width in (0, 6):
        with pytest.raises(ValueError, match='W<=V'):
            plan_inputs(banks, inputs, width)
    with pytest.raises(ValueError, match='int64'):
        plan_inputs(banks.int(), inputs, 1)
    with pytest.raises(ValueError, match='supplied together'):
        from lmdeploy.pytorch.kernels.cuda.compact_state import select_compact_state_ids
        select_compact_state_ids(banks, inputs, 1, consumed=inputs)


@pytest.mark.parametrize('batch', [0, 1, 7, 40])
@pytest.mark.parametrize('width', [1, 4, 6])
def test_graph_staging_pads_ids_and_does_not_modify_submitted_tables(batch, width):
    from lmdeploy.pytorch.kernels.cuda.compact_state import stage_compact_state_ids
    banks, inputs = _tables(batch, width)
    source = plan_inputs(banks, inputs, width)
    destination = SSMStateMetadata(torch.full((64, width + 1), 999, device='cuda'),
                                      torch.full((64,), 999, device='cuda'),
                                      torch.full((64, width), 999, device='cuda'))
    saved = [t.clone() for t in (source.state_ids, source.input_state_ids, source.output_state_ids)]
    stage_compact_state_ids(source, destination)
    for expected, actual in zip(saved, (destination.state_ids, destination.input_state_ids,
                                        destination.output_state_ids)):
        torch.testing.assert_close(actual[:batch], expected, rtol=0, atol=0)
        assert (actual[batch:] == -1).all()
    for expected, actual in zip(saved, (source.state_ids, source.input_state_ids, source.output_state_ids)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('graph', [False, True])
def test_one_shot_decode_checkpoint_is_final_then_next_bank_is_private(graph, cursor):
    banks, sources = _tables(3, 1)
    sources[:] = torch.tensor([99, 98, 97], device='cuda')
    checkpoint_ids = torch.tensor([100, -1, 101], device='cuda')
    original = plan_inputs(banks, sources, 1)
    if graph:
        cursor.rebind(original, banks, decode_ids=checkpoint_ids)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture, stream=torch.cuda.Stream()):
            bound = cursor.rebind(original, banks, decode_ids=checkpoint_ids)
        checkpoint_ids[0] = 102
        capture.replay()
    else:
        bound = cursor.rebind(original, banks, decode_ids=checkpoint_ids)
    assert bound.output_state_ids.tolist() == [[102 if graph else 100], [banks[1, 0].item()], [101]]
    assert bound.input_state_ids.tolist() == sources.tolist()
    advanced = _advance(bound, torch.tensor([1, 0, 1], device='cuda'))
    assert advanced.input_state_ids.tolist() == [102 if graph else 100, 98, 101]
    assert advanced.output_state_ids.tolist() == banks[:, :1].tolist()
    # The one-shot override never mutates retained/source metadata.
    assert original.output_state_ids.tolist() == banks[:, :1].tolist()


@pytest.mark.parametrize('alias', ['source', 'bank'])
def test_decode_checkpoint_output_alias_fails_closed(alias):
    banks, sources = _tables(1, 1)
    sources[0] = 99
    destination = sources.clone() if alias == 'source' else banks[:, 0].contiguous()
    bound = plan_inputs(banks, sources, 1, decode_ids=destination)
    assert bound.input_state_ids.tolist() == [-1] and bound.output_state_ids.tolist() == [[-1]]


@pytest.mark.parametrize('width', [1, 4, 6])
@pytest.mark.parametrize('alias', [False, True])
def test_dynamic_batch_reuses_id_kernels_and_clears_graph_padding(width, alias):
    from lmdeploy.pytorch.kernels.cuda.compact_state import (
        _select_ids,
        _stage_ids,
        select_compact_state_ids,
        stage_compact_state_ids,
    )
    device = torch.cuda.current_device()
    bank_width = width if alias else width + 1
    banks = torch.arange(1, 1 + 128 * bank_width, device='cuda').reshape(128, bank_width)
    inputs = torch.zeros(128, device='cuda', dtype=torch.int64)
    destination = SSMStateMetadata(banks.clone(), inputs.clone(), banks[:, :width].contiguous().clone())

    def run(batch):
        current, outputs = select_compact_state_ids(banks[:batch], inputs[:batch], width)
        source = SSMStateMetadata(banks[:batch], current, outputs)
        stage_compact_state_ids(source, destination)
        torch.testing.assert_close(current, inputs[:batch], rtol=0, atol=0)
        torch.testing.assert_close(outputs, banks[:batch, :width], rtol=0, atol=0)
        for src, dst in zip((source.state_ids, current, outputs),
                            (destination.state_ids, destination.input_state_ids, destination.output_state_ids)):
            torch.testing.assert_close(dst[:batch], src, rtol=0, atol=0)
            assert (dst[batch:] == -1).all()
        return source

    run(128)
    # Inspect the actual compiled variants, not just the decorator declaration.
    counts = [len(kernel.device_caches[device][0]) for kernel in (_select_ids, _stage_ids)]
    for batch in [0, *range(1, 65), 127, 128, 33, 1, 0]:
        run(batch)
    assert [len(kernel.device_caches[device][0]) for kernel in (_select_ids, _stage_ids)] == counts
    # Capture for one live B; successive staging launches clear shrinking padding.
    source = run(33)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        stage_compact_state_ids(source, destination)
    source.output_state_ids.add_(1000)
    graph.replay()
    torch.testing.assert_close(destination.output_state_ids[:33], source.output_state_ids, rtol=0, atol=0)
    assert (destination.output_state_ids[33:] == -1).all()
