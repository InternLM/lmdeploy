# Copyright (c) OpenMMLab. All rights reserved.
"""ID-only worker current boundary, independent of active batch membership."""
import pytest
import torch

from lmdeploy.pytorch.compact_state import SSMStateCursor, SSMStateMetadata


def _packet(batch, width, base=10):
    bank = torch.arange(base, base + batch * (width + 1), device='cuda').reshape(batch, width + 1)
    return SSMStateMetadata(bank, torch.zeros(batch, device='cuda', dtype=torch.long), bank[:, :1].contiguous())


@pytest.mark.parametrize('width', [1, 4, 6])
def test_cursor_retains_current_ids_across_batch_drop_and_recycled_slot_reset(monkeypatch, width):
    cursor = SSMStateCursor(8)
    slots = torch.tensor([1, 4, 6], device='cuda')
    cursor.current_ids[slots] = 999  # Stale previous occupants, ignored by cold bind.
    packet = _packet(3, width)
    history = torch.zeros(3, device='cuda', dtype=torch.long)
    current = cursor.bind(packet, slots, history)
    assert current.input_state_ids.tolist() == [0, 0, 0]
    assert cursor.current_ids[slots].tolist() == [999] * 3  # Bind is observational.
    lengths = torch.tensor([0, 72, 1], device='cuda')

    def forbidden(*args, **kwargs):
        pytest.fail('Cursor selection must not read acceptance or IDs back to the host.')

    with monkeypatch.context() as patch:
        for name in ('cpu', 'item', 'tolist', 'numpy'):
            patch.setattr(torch.Tensor, name, forbidden)
        advanced = cursor.advance(current, slots, (lengths > 0).long(), width=width)
    expected = torch.cat((torch.zeros(1, device='cuda', dtype=torch.long), current.output_state_ids[1:, 0]))
    torch.testing.assert_close(cursor.current_ids[slots], expected, rtol=0, atol=0)
    # Dropping all batch rows must not erase the stopped requests' current IDs.
    empty = advanced.index_select(torch.empty(0, device='cuda', dtype=torch.long))
    assert empty.input_state_ids.numel() == 0
    replacement = _packet(3, width, base=100)
    resumed = cursor.bind(replacement, slots, lengths)
    torch.testing.assert_close(resumed.input_state_ids, expected, rtol=0, atol=0)
    torch.testing.assert_close(resumed.output_state_ids, replacement.state_ids[:, :1], rtol=0, atol=0)
    # One slot is reused by a new request with history0, others resume.
    recycled = cursor.bind(replacement, slots, torch.tensor([0, 0, 1], device='cuda'))
    assert recycled.input_state_ids[1] == 0 and recycled.input_state_ids[2] == expected[2]
    assert cursor.current_ids[0] == -1


def test_cursor_explicit_source_override_and_dummy_slots_are_one_forward_only():
    cursor = SSMStateCursor(8)
    packet = _packet(5, 4)
    slots = torch.tensor([1, 2, -1, 0, 8], device='cuda')
    histories = torch.ones(5, device='cuda', dtype=torch.long)
    cursor.current_ids[1:3] = torch.tensor([31, 32], device='cuda')
    source = torch.tensor([500, -1, 500, 500, 500], device='cuda')
    before = cursor.current_ids.clone()
    bound = cursor.bind(packet, slots, histories, source)
    assert bound.input_state_ids.tolist() == [500, 32, -1, -1, -1]
    torch.testing.assert_close(cursor.current_ids, before, rtol=0, atol=0)
    cursor.advance(bound, slots, torch.ones(5, device='cuda', dtype=torch.long))
    resumed = cursor.bind(packet, slots, histories)
    assert resumed.input_state_ids[:2].tolist() == bound.output_state_ids[:2, 0].tolist()
    assert (resumed.input_state_ids[2:] == -1).all()
    assert cursor.current_ids[0] == -1 and torch.equal(cursor.current_ids[3:], before[3:])


def test_invalid_acceptance_poison_current_instead_of_retaining_stale_state():
    cursor = SSMStateCursor(2)
    packet = _packet(1, 4)
    slots = torch.ones(1, device='cuda', dtype=torch.long)
    cursor.current_ids[1] = 123
    cursor.advance(packet, slots, torch.tensor([2], device='cuda'))  # Only one prefill final exists.
    assert cursor.current_ids[1] == -1
    resumed = cursor.bind(packet, slots, torch.ones_like(slots))
    assert resumed.input_state_ids[0] == -1 and (resumed.output_state_ids == -1).all()


@pytest.mark.parametrize('slots', [0, 1])
def test_stateless_cursor_is_never_allocated(slots):
    with pytest.raises(ValueError, match='stateless'):
        SSMStateCursor(slots, device='cpu')


def test_cursor_selection_graph_replays_live_ids_without_host_readback():
    cursor = SSMStateCursor(4)
    packet = _packet(2, 4)
    slots = torch.tensor([1, 3], device='cuda')
    histories = torch.zeros(2, device='cuda', dtype=torch.long)
    counts = torch.ones(2, device='cuda', dtype=torch.long)
    # Compile all variants before capture.
    cursor.advance(cursor.bind(packet, slots, histories), slots, counts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        bound = cursor.bind(packet, slots, histories)
        next_ids = cursor.advance(bound, slots, counts)
    for history in (0, 72, 73):
        histories.fill_(history)
        before = cursor.current_ids[slots].clone()
        graph.replay()
        expected_source = torch.zeros_like(before) if history == 0 else before
        torch.testing.assert_close(bound.input_state_ids, expected_source, rtol=0, atol=0)
        torch.testing.assert_close(cursor.current_ids[slots], bound.output_state_ids[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(next_ids.input_state_ids, bound.output_state_ids[:, 0], rtol=0, atol=0)


@pytest.mark.parametrize('graph', [False, True])
def test_prefill_final_checkpoint_is_single_output_and_next_plan_is_private(graph):
    from lmdeploy.pytorch.compact_state import SSMPrefillMetadata

    cursor = SSMStateCursor(5)
    packet = _packet(3, 4)
    slots = torch.tensor([1, 2, 3], device='cuda')
    histories = torch.tensor([2048, 2048, 0], device='cuda')
    source = torch.tensor([500, 500, -1], device='cuda')
    prefill = SSMPrefillMetadata.from_lengths([64, 72, 56], [2048, 2048, 0],
                                               checkpoint_ids=[600, 601, 602],
                                               checkpoint_offsets=[64, 64, 56]).to_device('cuda')

    def bind():
        return cursor.bind(packet, slots, histories, source, prefill=prefill)

    bound = bind()
    if graph:
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture, stream=torch.cuda.Stream()):
            bound = bind()
        capture.replay()
    assert bound.input_state_ids.tolist() == [500, 500, 0]
    assert bound.output_state_ids[:, 0].tolist() == [600, 15, 602]
    before = packet.state_ids.clone()
    advanced = cursor.advance(bound, slots, torch.ones(3, device='cuda', dtype=torch.int64), width=4)
    assert advanced.input_state_ids.tolist() == [600, 15, 602]
    assert not torch.isin(advanced.output_state_ids, torch.tensor([500, 600, 601, 602], device='cuda')).any()
    torch.testing.assert_close(before, packet.state_ids, rtol=0, atol=0)
    retained = cursor.bind(packet, slots, histories + 64)
    assert retained.input_state_ids.tolist() == [600, 15, 602]
    assert retained.output_state_ids[:, 0].tolist() == [10, 16, 20]


@pytest.mark.parametrize('checkpoint', [10, 500])
def test_final_checkpoint_bank_or_source_alias_fails_closed(checkpoint):
    from lmdeploy.pytorch.compact_state import SSMPrefillMetadata

    cursor = SSMStateCursor(2)
    packet = _packet(1, 1)
    meta = SSMPrefillMetadata.from_lengths([64], [2048], checkpoint_ids=[checkpoint],
                                            checkpoint_offsets=[64]).to_device('cuda')
    bound = cursor.bind(packet, torch.tensor([1], device='cuda'), torch.tensor([2048], device='cuda'),
                         torch.tensor([500], device='cuda'), prefill=meta)
    assert bound.input_state_ids.tolist() == [-1] and bound.output_state_ids.tolist() == [[-1]]
