# Copyright (c) OpenMMLab. All rights reserved.
"""No-payload prefill IDs and deferred decode planning until admission."""
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.compact_state import SSMPrefillMetadata, SSMStateCursor, SSMStateMetadata
from lmdeploy.pytorch.kernels.cuda.compact_state import select_compact_state_ids
from tests.pytorch.compact_state_reference import plan_inputs


def ids(x):
    return torch.tensor(x, dtype=torch.int64, device='cuda')


def partial(bank):
    return SSMStateMetadata(ids(bank), ids([-1] * len(bank)), ids([[-1]] * len(bank)), partial_bank=True)


def test_partial_prefill_retains_final_until_full_decode_rebind():
    runtime = SSMStateCursor(4)
    request = ids([1, 2])
    state = partial([[1, -1, -1, -1, -1], [2, 3, -1, -1, -1]])
    state = runtime.bind(state, request, ids([0, 1024]), source_ids=ids([-1, 2]))
    assert state.input_state_ids.tolist() == [0, 2]
    assert state.output_state_ids.tolist() == [[1], [3]]
    next_state = runtime.advance(state, request, ids([1, 1]), width=4)
    assert next_state.partial_bank
    assert next_state.input_state_ids.tolist() == [1, 3]
    assert next_state.output_state_ids.tolist() == [[-1]*4, [-1]*4]
    assert runtime.current_ids.tolist() == [-1, 1, 3, -1]
    retained = runtime.current_ids.clone()
    admitted = runtime.rebind(next_state, ids([[1, 4, 5, 6, 7], [2, 3, 8, 9, 10]]))
    torch.testing.assert_close(runtime.current_ids, retained, rtol=0, atol=0)
    assert not admitted.partial_bank
    assert admitted.input_state_ids.tolist() == [1, 3]
    assert admitted.output_state_ids.tolist() == [[4, 5, 6, 7], [8, 9, 10, 2]]
    # The old submitted table/version remains unchanged.
    assert state.state_ids.tolist()[0] == [1, -1, -1, -1, -1]
    assert state.output_state_ids.tolist() == [[1], [3]]


@pytest.mark.parametrize('width', [1, 4])
def test_shared_source_needs_one_private_prefill_unit(width):
    bank = [[1] + [-1]*width, [2] + [-1]*width]
    state = partial(bank)
    runtime = SSMStateCursor(3)
    bound = runtime.bind(state, ids([1, 2]), ids([64, 64]), source_ids=ids([12, 12]))
    assert bound.input_state_ids.tolist() == [12, 12]
    assert bound.output_state_ids.tolist() == [[1], [2]]
    advanced = runtime.advance(bound, ids([1, 2]), ids([1, 1]), width=width)
    assert advanced.input_state_ids.tolist() == [1, 2]
    assert advanced.output_state_ids.tolist() == ([[1], [2]] if width == 1 else [[-1]*width, [-1]*width])


def test_final_checkpoint_can_produce_outside_single_private_bank():
    state = partial([[1, -1, -1, -1, -1]])
    prefill = SSMPrefillMetadata.from_lengths([8], [64], checkpoint_ids=[20], checkpoint_offsets=[8]).to_device('cuda')
    runtime = SSMStateCursor(2)
    bound = runtime.bind(state, ids([1]), ids([64]), source_ids=ids([1]), prefill=prefill)
    assert bound.input_state_ids.tolist() == [1] and bound.output_state_ids.tolist() == [[20]]
    advanced = runtime.advance(bound, ids([1]), ids([1]), width=4)
    assert advanced.input_state_ids.tolist() == [20]
    admitted = runtime.rebind(advanced, ids([[1, 2, 3, 4, 5]]))
    assert admitted.output_state_ids.tolist() == [[1, 2, 3, 4]]


@pytest.mark.parametrize('length', [0, 8])
def test_prefill_continuation_retains_no_progress_or_selects_final(length):
    previous = plan_inputs(ids([[1, -1, -1, -1, -1]]), ids([0]), 1, partial_bank=True)
    current = partial([[1, 2, -1, -1, -1]])
    runtime = SSMStateCursor(2)
    request = ids([1])
    runtime.advance(previous, request, ids([int(length > 0)]))
    following = runtime.bind(current, request, ids([length]))
    assert following.partial_bank
    assert following.input_state_ids.tolist() == [1 if length else 0]
    assert following.output_state_ids.tolist() == [[2 if length else 1]]


@pytest.mark.parametrize('bank', [[1, -1, 2, -1, -1], [0, -1, -1, -1, -1],
                                  [-1, -1, -1, -1, -1], [1, -2, -1, -1, -1]])
def test_invalid_partial_padding_fails_closed(bank):
    incoming, outgoing = select_compact_state_ids(ids([bank]), ids([0]), 1, allow_partial_bank=True)
    assert incoming.tolist() == [-1] and outgoing.tolist() == [[-1]]


def test_full_bank_contract_still_rejects_padding():
    incoming, outgoing = select_compact_state_ids(ids([[1, -1, -1, -1, -1]]), ids([0]), 1)
    assert incoming.tolist() == [-1] and outgoing.tolist() == [[-1]]


def test_graph_replay_changes_partial_bank_size_without_losing_state():
    runtime = SSMStateCursor(2)
    state = partial([[1, -1, -1, -1, -1]])
    request, history, source = ids([1]), ids([64]), ids([10])
    consumed = ids([1])
    warm = runtime.bind(state, request, history, source_ids=source)
    runtime.advance(warm, request, consumed, width=4)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        bound = runtime.bind(state, request, history, source_ids=source)
        advanced = runtime.advance(bound, request, consumed, width=4)
    for bank, input_id, output_id in [([1, -1, -1, -1, -1], 10, 1),
                                    ([1, 2, -1, -1, -1], 1, 2),
                                    ([1, 2, -1, -1, -1], 2, 1)]:
        state.state_ids.copy_(ids([bank]))
        source.copy_(ids([input_id]))
        graph.replay()
        assert bound.output_state_ids.tolist() == [[output_id]]
        assert advanced.input_state_ids.tolist() == [output_id]
        assert advanced.output_state_ids.tolist() == [[-1]*4]
        assert runtime.current_ids[1].item() == output_id


def test_partial_flag_survives_metadata_moves_and_reindex():
    state = partial([[1, -1, -1, -1, -1], [2, 3, -1, -1, -1]])
    moved = state.to_device('cpu')
    assert moved.partial_bank
    one = moved.index_select(torch.tensor([1]))
    assert one.partial_bank and one.state_ids.tolist() == [[2, 3, -1, -1, -1]]
    assert one.merge(one).partial_bank
    full = SSMStateMetadata(torch.tensor([[4, 5, 6, 7, 8]]), torch.tensor([4]), torch.tensor([[5]]))
    # A newly finished prefill joins an existing decode batch. The whole
    # merged packet remains guarded until the scheduler sends a full rebind.
    for merged in (one.merge(full), full.merge(one)):
        assert merged.partial_bank and merged.state_ids.shape == (2, 5)
    assert not full.merge(full).partial_bank


def test_decode_execution_rejects_unpromoted_metadata_before_model_access():
    from lmdeploy.pytorch.engine.model_agent.agent import model_forward
    state = SSMStateMetadata(torch.tensor([[1, -1]]), torch.tensor([1]), torch.tensor([[-1]]), partial_bank=True)
    with pytest.raises(ValueError, match='fully admitted bank rebind'):
        model_forward(None, SimpleNamespace(ssm_state_meta=state, is_decoding=True), None, None, None)
