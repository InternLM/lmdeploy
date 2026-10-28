# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch

from .reference import (
    build_draft_extension_key_offsets_reference,
    build_draft_refresh_inputs_reference,
    draft_argmax_and_store_token_reference,
    initialize_target_verification_reference,
    pack_stop_words,
    stop_criteria_reference,
)
from .speculative_sequence import (
    build_draft_extension_key_offsets,
    build_draft_refresh_inputs,
    draft_argmax_and_store_token,
    initialize_target_verification,
    speculative_stop_criteria,
    stop_criteria,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='CUDA is required')

_DEVICE = 'cuda'
_SPECULATIVE_K = 7
_OUTPUT_SENTINEL = -7777777
_TOKEN_SENTINEL = -6060606
_POISON_SEQUENCE_LENGTH = 1700000000
_POISON_ACCEPT_LEN = 900000000


def _make_widths(batch_size: int, pattern: str) -> torch.Tensor:
    if pattern == 'all_candidate':
        return torch.ones(batch_size, dtype=torch.int32, device=_DEVICE)
    if pattern == 'all_non_candidate':
        return torch.zeros(batch_size, dtype=torch.int32, device=_DEVICE)
    if pattern == 'mixed':
        rows = torch.arange(batch_size, dtype=torch.int32, device=_DEVICE)
        return ((rows % 3) != 1).to(torch.int32)
    raise ValueError(f'unknown width pattern: {pattern}')


def _make_offsets(widths: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        torch.zeros(1, dtype=torch.int32, device=widths.device),
        torch.cumsum(widths, dim=0, dtype=torch.int32),
    ])


def _make_case(batch_size: int, pattern: str):
    widths = _make_widths(batch_size, pattern)
    q_offsets = _make_offsets(widths)
    rows = torch.arange(batch_size, dtype=torch.int32, device=_DEVICE)
    entry_sequence_length = 9 + rows % 97
    accept_len = rows % 8

    zero_width = widths == 0
    entry_sequence_length = torch.where(
        zero_width,
        torch.full_like(entry_sequence_length, _POISON_SEQUENCE_LENGTH),
        entry_sequence_length,
    )
    accept_len = torch.where(
        zero_width,
        torch.full_like(accept_len, _POISON_ACCEPT_LEN),
        accept_len,
    )
    return q_offsets, entry_sequence_length, accept_len


def _guarded_output(batch_size: int):
    backing = torch.full((batch_size + 3, ),
                         _OUTPUT_SENTINEL,
                         dtype=torch.int32,
                         device=_DEVICE)
    return backing, backing[1:-1]


def _make_token_rows(row_values):
    backings = []
    rows = []
    for values in row_values:
        backing = torch.full((len(values) + 2, ),
                             _TOKEN_SENTINEL,
                             dtype=torch.int32,
                             device=_DEVICE)
        row = backing[1:-1]
        row.copy_(torch.tensor(values, dtype=torch.int32, device=_DEVICE))
        backings.append(backing)
        rows.append(row)
    return backings, rows


def _token_pointer_array(rows):
    return torch.tensor(
        [0 if row is None else row.data_ptr() for row in rows],
        dtype=torch.int64,
        device=_DEVICE,
    )


def _assert_token_guards(backings):
    for backing in backings:
        assert backing[0].item() == _TOKEN_SENTINEL
        assert backing[-1].item() == _TOKEN_SENTINEL


def _guarded_int_vector(values, sentinel=_OUTPUT_SENTINEL):
    backing = torch.full((len(values) + 2, ),
                         sentinel,
                         dtype=torch.int32,
                         device=_DEVICE)
    view = backing[1:-1]
    if values:
        view.copy_(torch.tensor(values, dtype=torch.int32, device=_DEVICE))
    return backing, view


def _guarded_bool_vector(values, sentinel=True):
    backing = torch.full((len(values) + 2, ),
                         sentinel,
                         dtype=torch.bool,
                         device=_DEVICE)
    view = backing[1:-1]
    if values:
        view.copy_(torch.tensor(values, dtype=torch.bool, device=_DEVICE))
    return backing, view


def _assert_vector_guards(backing, sentinel):
    assert backing[0].item() == sentinel
    assert backing[-1].item() == sentinel


def _stop_and_compare(
    token_rows,
    token_ids_ptrs,
    sequence_length,
    stop_words,
    sequence_length_limit,
    finished,
    accept_len=None,
):
    expected_accept_len, expected_finished = stop_criteria_reference(
        token_rows,
        sequence_length,
        stop_words,
        sequence_length_limit,
        finished,
        accept_len=accept_len,
    )
    pointer_before = token_ids_ptrs.clone()
    rows_before = [row.clone() for row in token_rows]
    sequence_length_before = sequence_length.clone()
    limit_before = sequence_length_limit.clone()
    stop_words_before = None if stop_words is None else stop_words.clone()

    if accept_len is None:
        result = stop_criteria(
            token_ids_ptrs,
            sequence_length,
            stop_words,
            sequence_length_limit,
            finished,
        )
    else:
        result = speculative_stop_criteria(
            token_ids_ptrs,
            sequence_length,
            accept_len,
            stop_words,
            sequence_length_limit,
            finished,
        )

    assert result is None
    assert torch.equal(finished, expected_finished)
    if accept_len is not None:
        assert torch.equal(accept_len, expected_accept_len)
    assert torch.equal(token_ids_ptrs, pointer_before)
    for actual, before in zip(token_rows, rows_before):
        assert torch.equal(actual, before)
    assert torch.equal(sequence_length, sequence_length_before)
    assert torch.equal(sequence_length_limit, limit_before)
    if stop_words is not None:
        assert torch.equal(stop_words, stop_words_before)


def _refresh_and_compare(
    token_rows,
    token_ids_ptrs,
    draft_input_ids,
    selected_token_pos,
    candidate_active,
    refresh_q_offsets,
    refresh_k_offsets,
    extension_q_offsets,
    accept_len,
    limit_to_accept_len,
    finished,
):
    expected = build_draft_refresh_inputs_reference(
        token_rows,
        refresh_q_offsets,
        refresh_k_offsets,
        extension_q_offsets,
        accept_len,
        limit_to_accept_len,
        finished,
    )
    token_rows_before = [
        None if row is None else row.clone() for row in token_rows
    ]
    inputs = [
        token_ids_ptrs,
        refresh_q_offsets,
        refresh_k_offsets,
        extension_q_offsets,
        accept_len,
        limit_to_accept_len,
        finished,
    ]
    inputs_before = [value.clone() for value in inputs]

    result = build_draft_refresh_inputs(
        draft_input_ids,
        selected_token_pos,
        candidate_active,
        token_ids_ptrs,
        refresh_q_offsets,
        refresh_k_offsets,
        extension_q_offsets,
        accept_len,
        limit_to_accept_len,
        finished,
    )

    expected_ids, expected_pos, expected_active = expected
    assert result is None
    assert torch.equal(draft_input_ids, expected_ids)
    assert torch.equal(selected_token_pos, expected_pos)
    assert torch.equal(candidate_active, expected_active)
    for actual, before in zip(inputs, inputs_before):
        assert torch.equal(actual, before)
    for actual, before in zip(token_rows, token_rows_before):
        if actual is not None:
            assert torch.equal(actual, before)


def _argmax_and_compare(
    logits,
    token_rows,
    proposal_ids,
    token_ids_ptrs,
    extension_q_offsets,
    candidate_active,
    entry_sequence_length,
    accept_len,
    proposal_index,
    vocab_size,
):
    expected_rows, expected_proposals = draft_argmax_and_store_token_reference(
        logits,
        token_rows,
        extension_q_offsets,
        candidate_active,
        entry_sequence_length,
        accept_len,
        proposal_index,
        vocab_size,
    )
    inputs = [
        logits,
        token_ids_ptrs,
        extension_q_offsets,
        candidate_active,
        entry_sequence_length,
        accept_len,
    ]
    inputs_before = [value.clone() for value in inputs]

    result = draft_argmax_and_store_token(
        logits,
        proposal_ids,
        token_ids_ptrs,
        extension_q_offsets,
        candidate_active,
        entry_sequence_length,
        accept_len,
        proposal_index,
        vocab_size,
    )

    assert result is None
    assert torch.equal(proposal_ids, expected_proposals)
    for actual, expected in zip(token_rows, expected_rows):
        if actual is not None:
            assert torch.equal(actual, expected)
    for actual, before in zip(inputs, inputs_before):
        if actual.is_floating_point():
            torch.testing.assert_close(actual,
                                       before,
                                       rtol=0,
                                       atol=0,
                                       equal_nan=True)
        else:
            assert torch.equal(actual, before)


@pytest.mark.parametrize('draft_count', range(1, 8))
@pytest.mark.parametrize('metrics_enabled', [False, True])
def test_initialize_target_verification_mixed_rows_and_poison_pointers(
    draft_count,
    metrics_enabled,
):
    batch_size = 7
    generation_count = 5

    entry_sequence_length = torch.tensor(
        [5, 6, 7, 8, 9, 4, 3],
        dtype=torch.int32,
        device=_DEVICE,
    )
    finished_on_entry = torch.tensor(
        [False, False, False, True, False, False, False],
        dtype=torch.bool,
        device=_DEVICE,
    )
    speculative_row = torch.tensor(
        [True, False, False, True, False, True, False],
        dtype=torch.bool,
        device=_DEVICE,
    )
    request_to_generation_row_offsets = torch.tensor(
        [0, 1, 1, 2, 3, 4, 5, 5],
        dtype=torch.int32,
        device=_DEVICE,
    )

    # Allocate row 5 before row 0 so request order differs from token-row
    # allocation order. All ordinary and inherited-finished pointers are null.
    row5_backing = torch.full(
        (4 + draft_count + 3, ),
        _TOKEN_SENTINEL,
        dtype=torch.int32,
        device=_DEVICE,
    )
    row5 = row5_backing[1:-1]
    row5.copy_(
        torch.arange(
            500,
            500 + row5.numel(),
            dtype=torch.int32,
            device=_DEVICE,
        ))
    row0_backing = torch.full(
        (5 + draft_count + 3, ),
        _TOKEN_SENTINEL,
        dtype=torch.int32,
        device=_DEVICE,
    )
    row0 = row0_backing[1:-1]
    row0.copy_(
        torch.arange(
            100,
            100 + row0.numel(),
            dtype=torch.int32,
            device=_DEVICE,
        ))
    token_rows = [row0, None, None, None, None, row5, None]
    request_token_ids_ptrs = _token_pointer_array(token_rows)

    block_logits_active = torch.ones((draft_count + 1, generation_count),
                                     dtype=torch.bool,
                                     device=_DEVICE)
    effective_history = torch.full(
        (draft_count + 1, generation_count),
        _OUTPUT_SENTINEL,
        dtype=torch.int32,
        device=_DEVICE,
    )
    verification_draft_ids = torch.full(
        (draft_count, generation_count),
        _OUTPUT_SENTINEL,
        dtype=torch.int32,
        device=_DEVICE,
    )

    accepted_backing = None
    accepted_draft_count = None
    if metrics_enabled:
        accepted_backing, accepted_draft_count = _guarded_int_vector(
            [_OUTPUT_SENTINEL] * batch_size)

    expected = initialize_target_verification_reference(
        token_rows,
        entry_sequence_length,
        finished_on_entry,
        speculative_row,
        request_to_generation_row_offsets,
        draft_count,
        metrics_enabled,
    )

    pointer_before = request_token_ids_ptrs.clone()
    row0_before = row0.clone()
    row5_before = row5.clone()

    initialize_target_verification(
        block_logits_active,
        effective_history,
        verification_draft_ids,
        request_token_ids_ptrs,
        entry_sequence_length,
        finished_on_entry,
        speculative_row,
        accepted_draft_count,
        request_to_generation_row_offsets,
    )

    (
        expected_block,
        expected_history,
        expected_drafts,
        expected_accepted_count,
    ) = expected

    assert torch.equal(block_logits_active, expected_block)
    assert torch.equal(effective_history, expected_history)
    assert torch.equal(verification_draft_ids, expected_drafts)
    if metrics_enabled:
        assert torch.equal(accepted_draft_count, expected_accepted_count)

    if metrics_enabled:
        _assert_vector_guards(accepted_backing, _OUTPUT_SENTINEL)

    assert torch.equal(request_token_ids_ptrs, pointer_before)
    assert torch.equal(row0, row0_before)
    assert torch.equal(row5, row5_before)
    _assert_token_guards([row0_backing, row5_backing])


@pytest.mark.parametrize('extension_index', [0, 2, _SPECULATIVE_K - 2])
@pytest.mark.parametrize('pattern',
                         ['all_candidate', 'all_non_candidate', 'mixed'])
@pytest.mark.parametrize('batch_size', [0, 1, 255, 256, 257, 513])
def test_build_draft_extension_key_offsets_matrix(batch_size, pattern,
                                                  extension_index):
    q_offsets, entry_sequence_length, accept_len = _make_case(
        batch_size, pattern)
    expected = build_draft_extension_key_offsets_reference(
        q_offsets,
        entry_sequence_length,
        accept_len,
        extension_index,
    )

    q_offsets_before = q_offsets.clone()
    entry_sequence_length_before = entry_sequence_length.clone()
    accept_len_before = accept_len.clone()
    backing, k_offsets = _guarded_output(batch_size)

    result = build_draft_extension_key_offsets(
        k_offsets,
        q_offsets,
        entry_sequence_length,
        accept_len,
        extension_index,
    )

    assert result is None
    assert torch.equal(k_offsets, expected)
    assert backing[0].item() == _OUTPUT_SENTINEL
    assert backing[-1].item() == _OUTPUT_SENTINEL
    assert torch.equal(q_offsets, q_offsets_before)
    assert torch.equal(entry_sequence_length, entry_sequence_length_before)
    assert torch.equal(accept_len, accept_len_before)


@pytest.mark.parametrize(
    ('case_name', 'entry_sequence_length', 'accept_len', 'extension_index',
     'expected_key_len'),
    [
        ('active_final_prompt', 31, 1, 0, 32),
        ('active_fallback', 17, 1, 0, 18),
        ('speculative_accept_one', 20, 1, 3, 24),
        ('speculative_accept_k', 20, _SPECULATIVE_K, 3,
         20 + _SPECULATIVE_K + 3),
        ('inherited_finished', 20, 0, 0, 20),
    ],
)
def test_build_draft_extension_key_offsets_semantics(
    case_name,
    entry_sequence_length,
    accept_len,
    extension_index,
    expected_key_len,
):
    del case_name
    q_offsets = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    entry_sequence_length = torch.tensor([entry_sequence_length],
                                         dtype=torch.int32,
                                         device=_DEVICE)
    accept_len = torch.tensor([accept_len], dtype=torch.int32, device=_DEVICE)
    k_offsets = torch.full((2, ),
                           _OUTPUT_SENTINEL,
                           dtype=torch.int32,
                           device=_DEVICE)

    build_draft_extension_key_offsets(
        k_offsets,
        q_offsets,
        entry_sequence_length,
        accept_len,
        extension_index,
    )

    assert torch.equal(
        k_offsets,
        torch.tensor([0, expected_key_len], dtype=torch.int32, device=_DEVICE))


def test_zero_width_rows_ignore_poisoned_values():
    q_offsets = torch.tensor([0, 1, 1, 2, 2],
                             dtype=torch.int32,
                             device=_DEVICE)
    entry_sequence_length = torch.tensor(
        [10, _POISON_SEQUENCE_LENGTH, 30, _POISON_SEQUENCE_LENGTH],
        dtype=torch.int32,
        device=_DEVICE,
    )
    accept_len = torch.tensor(
        [2, _POISON_ACCEPT_LEN, 7, _POISON_ACCEPT_LEN],
        dtype=torch.int32,
        device=_DEVICE,
    )
    extension_index = 4
    k_offsets = torch.empty(5, dtype=torch.int32, device=_DEVICE)

    build_draft_extension_key_offsets(
        k_offsets,
        q_offsets,
        entry_sequence_length,
        accept_len,
        extension_index,
    )

    expected = torch.tensor([0, 16, 16, 57, 57],
                            dtype=torch.int32,
                            device=_DEVICE)
    assert torch.equal(k_offsets, expected)


def test_build_draft_extension_key_offsets_nondefault_stream():
    batch_size = 257
    extension_index = 2
    q_offsets, entry_sequence_length, accept_len = _make_case(
        batch_size, 'mixed')
    expected = build_draft_extension_key_offsets_reference(
        q_offsets,
        entry_sequence_length,
        accept_len,
        extension_index,
    )
    backing, k_offsets = _guarded_output(batch_size)

    current_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current_stream)

    with torch.cuda.stream(stream):
        build_draft_extension_key_offsets(
            k_offsets,
            q_offsets,
            entry_sequence_length,
            accept_len,
            extension_index,
        )

    current_stream.wait_stream(stream)
    assert torch.equal(k_offsets, expected)
    assert backing[0].item() == _OUTPUT_SENTINEL
    assert backing[-1].item() == _OUTPUT_SENTINEL


def test_successive_launches_reuse_output_and_preserve_stream_order():
    batch_size = 513
    q_offsets, entry_sequence_length, accept_len = _make_case(
        batch_size, 'mixed')
    expected_first = build_draft_extension_key_offsets_reference(
        q_offsets,
        entry_sequence_length,
        accept_len,
        extension_index=0,
    )
    expected_second = build_draft_extension_key_offsets_reference(
        q_offsets,
        entry_sequence_length,
        accept_len,
        extension_index=_SPECULATIVE_K - 2,
    )
    backing, k_offsets = _guarded_output(batch_size)

    current_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current_stream)

    with torch.cuda.stream(stream):
        build_draft_extension_key_offsets(
            k_offsets,
            q_offsets,
            entry_sequence_length,
            accept_len,
            extension_index=0,
        )
        first_snapshot = k_offsets.clone()

        build_draft_extension_key_offsets(
            k_offsets,
            q_offsets,
            entry_sequence_length,
            accept_len,
            extension_index=_SPECULATIVE_K - 2,
        )
        second_snapshot = k_offsets.clone()

    current_stream.wait_stream(stream)
    assert torch.equal(first_snapshot, expected_first)
    assert torch.equal(second_snapshot, expected_second)
    assert torch.equal(k_offsets, expected_second)
    assert backing[0].item() == _OUTPUT_SENTINEL
    assert backing[-1].item() == _OUTPUT_SENTINEL


def test_ordinary_stop_criteria_length_and_packed_phrases():
    backings, allocations = _make_token_rows([
        [1, 2, 3, 0],
        [5, 6, 7, 0],
        [8, 9, 0, 0],
        [11, 12, 13, 0],
        [14, 15, 16, 0],
        [0, 0, 0, 0],
    ])
    token_rows = [
        allocations[2], allocations[0], allocations[4], allocations[1],
        allocations[5], allocations[3]
    ]
    token_ids_ptrs = _token_pointer_array(token_rows)
    sequence_length = torch.tensor([2, 3, 3, 3, 0, 3],
                                   dtype=torch.int32,
                                   device=_DEVICE)
    limits = torch.tensor([20, 3, 20, 20, 20, 20],
                          dtype=torch.int32,
                          device=_DEVICE)
    phrases = [
        [[9]],
        [],
        [[15, 16]],
        [[6, 7]],
        [],
        [[999]],
    ]
    stop_words = pack_stop_words(phrases, width=4, device=_DEVICE)
    finished_backing, finished = _guarded_bool_vector(
        [False, False, False, False, False, True])

    _stop_and_compare(token_rows, token_ids_ptrs, sequence_length, stop_words,
                      limits, finished)

    assert finished.tolist() == [True, True, True, True, False, True]
    _assert_vector_guards(finished_backing, True)
    _assert_token_guards(backings)


@pytest.mark.parametrize('span_len', range(1, 9))
def test_speculative_stop_criteria_no_boundary_preserves_span(span_len):
    entry_len = 2
    _, token_rows = _make_token_rows([[1, 2] + list(range(20, 20 + span_len)) +
                                      [0]])
    token_ids_ptrs = _token_pointer_array(token_rows)
    entry_sequence_length = torch.tensor([entry_len],
                                         dtype=torch.int32,
                                         device=_DEVICE)
    accept_len = torch.tensor([span_len], dtype=torch.int32, device=_DEVICE)
    limit = torch.tensor([100], dtype=torch.int32, device=_DEVICE)
    finished = torch.zeros(1, dtype=torch.bool, device=_DEVICE)

    _stop_and_compare(
        token_rows,
        token_ids_ptrs,
        entry_sequence_length,
        None,
        limit,
        finished,
        accept_len=accept_len,
    )

    assert accept_len.item() == span_len
    assert not finished.item()


@pytest.mark.parametrize('terminal_position', range(8))
def test_speculative_length_boundary_at_every_position(terminal_position):
    entry_len = 3
    _, token_rows = _make_token_rows([[1, 2, 3] + list(range(30, 38)) + [0]])
    token_ids_ptrs = _token_pointer_array(token_rows)
    entry_sequence_length = torch.tensor([entry_len],
                                         dtype=torch.int32,
                                         device=_DEVICE)
    accept_len = torch.tensor([8], dtype=torch.int32, device=_DEVICE)
    limit = torch.tensor([entry_len + terminal_position + 1],
                         dtype=torch.int32,
                         device=_DEVICE)
    finished = torch.zeros(1, dtype=torch.bool, device=_DEVICE)

    _stop_and_compare(
        token_rows,
        token_ids_ptrs,
        entry_sequence_length,
        None,
        limit,
        finished,
        accept_len=accept_len,
    )

    assert accept_len.item() == terminal_position + 1
    assert finished.item()


def test_speculative_single_token_stop_at_every_position():
    entry_len = 2
    row_values = [[1, 2] + [100 * b + j for j in range(8)] + [0]
                  for b in range(8)]
    _, token_rows = _make_token_rows(row_values)
    token_ids_ptrs = _token_pointer_array(token_rows)
    phrases = [[[7000 + b], [100 * b + b], [8000 + b]] for b in range(8)]
    stop_words = pack_stop_words(phrases, width=4, device=_DEVICE)
    entry_sequence_length = torch.full((8, ),
                                       entry_len,
                                       dtype=torch.int32,
                                       device=_DEVICE)
    accept_len = torch.full((8, ), 8, dtype=torch.int32, device=_DEVICE)
    limits = torch.full((8, ), 100, dtype=torch.int32, device=_DEVICE)
    finished = torch.zeros(8, dtype=torch.bool, device=_DEVICE)

    _stop_and_compare(
        token_rows,
        token_ids_ptrs,
        entry_sequence_length,
        stop_words,
        limits,
        finished,
        accept_len=accept_len,
    )

    assert accept_len.tolist() == list(range(1, 9))
    assert finished.all()


def test_speculative_stop_phrase_ending_at_every_position_and_crossing_entry():
    entry_len = 2
    row_values = []
    phrases = []
    for b in range(8):
        selected = [100 * b + j for j in range(8)]
        row = [10 * b + 1, 10 * b + 2] + selected + [0]
        phrase = [row[entry_len + b - 1], row[entry_len + b]]
        row_values.append(row)
        phrases.append([[9999], phrase])

    _, token_rows = _make_token_rows(row_values)
    token_ids_ptrs = _token_pointer_array(token_rows)
    stop_words = pack_stop_words(phrases, width=4, device=_DEVICE)
    entry_sequence_length = torch.full((8, ),
                                       entry_len,
                                       dtype=torch.int32,
                                       device=_DEVICE)
    accept_len = torch.full((8, ), 8, dtype=torch.int32, device=_DEVICE)
    limits = torch.full((8, ), 100, dtype=torch.int32, device=_DEVICE)
    finished = torch.zeros(8, dtype=torch.bool, device=_DEVICE)

    _stop_and_compare(
        token_rows,
        token_ids_ptrs,
        entry_sequence_length,
        stop_words,
        limits,
        finished,
        accept_len=accept_len,
    )

    assert accept_len.tolist() == list(range(1, 9))
    assert finished.all()


def test_speculative_stop_edge_semantics_guards_and_effective_eos():
    row_values = [
        [4, 5, 40, 41, 42, 0],
        [1, 2, 10, 11, 12, 0],
        [1, 2, 20, 21, 22, 0],
        [1, 2, 2, 31, 32, 0],
        [1, 2, 2, 41, 42, 0],
        [1, 2, 50, 51, 52, 53],
        [1, 2, 60, 61, 62, 63],
        [1, 2, 2, 71, 72, 0],
    ]
    backings, allocations = _make_token_rows(row_values)
    token_rows = [allocations[i] for i in [3, 0, 6, 1, 7, 2, 5, 4]]
    token_ids_ptrs = _token_pointer_array(token_rows)
    phrases = [
        [],
        [[4, 5]],
        [],
        [],
        [[2]],
        [[22]],
        [[51]],
        [],
    ]
    packed = pack_stop_words(phrases, width=4, device=_DEVICE)
    stop_backing = torch.full((packed.numel() + 2, ),
                              _OUTPUT_SENTINEL,
                              dtype=torch.int32,
                              device=_DEVICE)
    stop_words = stop_backing[1:-1].view_as(packed)
    stop_words.copy_(packed)
    entry_sequence_length = torch.full((8, ),
                                       2,
                                       dtype=torch.int32,
                                       device=_DEVICE)
    accept_backing, accept_len = _guarded_int_vector([3, 3, 0, 3, 3, 4, 4, 3])
    limit = torch.tensor([100, 100, 100, 100, 100, 4, 7, 3],
                         dtype=torch.int32,
                         device=_DEVICE)
    finished_backing, finished = _guarded_bool_vector(
        [False, False, False, False, False, False, False, False])

    _stop_and_compare(
        token_rows,
        token_ids_ptrs,
        entry_sequence_length,
        stop_words,
        limit,
        finished,
        accept_len=accept_len,
    )

    assert accept_len.tolist() == [3, 3, 0, 3, 1, 2, 2, 1]
    assert finished.tolist() == [
        False, False, False, False, True, True, True, True
    ]
    assert stop_backing[0].item() == _OUTPUT_SENTINEL
    assert stop_backing[-1].item() == _OUTPUT_SENTINEL
    _assert_vector_guards(accept_backing, _OUTPUT_SENTINEL)
    _assert_vector_guards(finished_backing, True)
    _assert_token_guards(backings)

    finished[1] = True
    accept_len[1] = 3
    _stop_and_compare(
        token_rows,
        token_ids_ptrs,
        entry_sequence_length,
        stop_words,
        limit,
        finished,
        accept_len=accept_len,
    )
    assert accept_len[1].item() == 0


def test_stop_criteria_zero_batch():
    token_ids_ptrs = torch.empty(0, dtype=torch.int64, device=_DEVICE)
    lengths = torch.empty(0, dtype=torch.int32, device=_DEVICE)
    limits = torch.empty(0, dtype=torch.int32, device=_DEVICE)
    finished = torch.empty(0, dtype=torch.bool, device=_DEVICE)
    accept_len = torch.empty(0, dtype=torch.int32, device=_DEVICE)

    stop_criteria(token_ids_ptrs, lengths, None, limits, finished)
    speculative_stop_criteria(token_ids_ptrs, lengths, accept_len, None,
                              limits, finished)


def test_speculative_stop_nondefault_stream_and_successive_snapshots():
    entry_len = 2
    _, token_rows = _make_token_rows([[1, 2, 90, 91, 0]])
    token_ids_ptrs = _token_pointer_array(token_rows)
    entry_sequence_length = torch.tensor([entry_len],
                                         dtype=torch.int32,
                                         device=_DEVICE)
    accept_len = torch.tensor([2], dtype=torch.int32, device=_DEVICE)
    limits = torch.tensor([100], dtype=torch.int32, device=_DEVICE)
    finished = torch.zeros(1, dtype=torch.bool, device=_DEVICE)
    current_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current_stream)

    with torch.cuda.stream(stream):
        speculative_stop_criteria(
            token_ids_ptrs,
            entry_sequence_length,
            accept_len,
            None,
            limits,
            finished,
        )
        first_snapshot = accept_len.clone(), finished.clone()
        limits.fill_(entry_len + 1)
        speculative_stop_criteria(
            token_ids_ptrs,
            entry_sequence_length,
            accept_len,
            None,
            limits,
            finished,
        )
        second_snapshot = accept_len.clone(), finished.clone()

    current_stream.wait_stream(stream)
    assert first_snapshot[0].item() == 2
    assert not first_snapshot[1].item()
    assert second_snapshot[0].item() == 1
    assert second_snapshot[1].item()


def test_build_draft_refresh_inputs_mixed_row_modes_mapping_and_guards():
    backings, allocations = _make_token_rows([
        list(range(100, 120)),
        list(range(200, 220)),
        list(range(300, 320)),
        list(range(400, 420)),
        list(range(500, 520)),
    ])
    token_rows = [
        allocations[2],
        allocations[0],
        allocations[4],
        allocations[1],
        allocations[3],
        None,
    ]
    token_ids_ptrs = _token_pointer_array(token_rows)
    refresh_q_offsets = torch.tensor(
        [0, 3, 5, 7, 11, 12, 12],
        dtype=torch.int32,
        device=_DEVICE,
    )
    refresh_k_offsets = torch.tensor(
        [0, 3, 9, 14, 22, 29, 29],
        dtype=torch.int32,
        device=_DEVICE,
    )
    extension_q_offsets = torch.tensor(
        [0, 0, 0, 1, 2, 2, 2],
        dtype=torch.int32,
        device=_DEVICE,
    )
    accept_len = torch.tensor(
        [
            _POISON_ACCEPT_LEN,
            _POISON_ACCEPT_LEN,
            _POISON_ACCEPT_LEN,
            2,
            1,
            _POISON_ACCEPT_LEN,
        ],
        dtype=torch.int32,
        device=_DEVICE,
    )
    limiting = torch.tensor(
        [False, False, False, True, True, True],
        dtype=torch.bool,
        device=_DEVICE,
    )
    finished = torch.tensor(
        [False, False, False, False, False, True],
        dtype=torch.bool,
        device=_DEVICE,
    )
    draft_backing, draft_input_ids = _guarded_int_vector([_OUTPUT_SENTINEL] *
                                                         12)
    selected_backing, selected_token_pos = _guarded_int_vector(
        [_OUTPUT_SENTINEL] * 2)
    active_backing, candidate_active = _guarded_bool_vector([False, False])

    _refresh_and_compare(
        token_rows,
        token_ids_ptrs,
        draft_input_ids,
        selected_token_pos,
        candidate_active,
        refresh_q_offsets,
        refresh_k_offsets,
        extension_q_offsets,
        accept_len,
        limiting,
        finished,
    )

    assert draft_input_ids.tolist() == [
        301,
        302,
        303,
        105,
        106,
        504,
        505,
        205,
        206,
        0,
        0,
        407,
    ]
    assert selected_token_pos.tolist() == [6, 8]
    assert candidate_active.tolist() == [True, True]
    _assert_vector_guards(draft_backing, _OUTPUT_SENTINEL)
    _assert_vector_guards(selected_backing, _OUTPUT_SENTINEL)
    _assert_vector_guards(active_backing, True)
    _assert_token_guards(backings)


@pytest.mark.parametrize(
    ('width', 'committed'),
    [(width, committed) for width in range(2, 9)
     for committed in range(width + 1)],
)
def test_build_draft_refresh_inputs_every_speculative_prefix(width, committed):
    entry_len = 3
    values = [10, 11, 12] + list(range(100, 100 + width)) + [999]
    backings, token_rows = _make_token_rows([values])
    token_ids_ptrs = _token_pointer_array(token_rows)
    refresh_q_offsets = torch.tensor([0, width],
                                     dtype=torch.int32,
                                     device=_DEVICE)
    refresh_k_offsets = torch.tensor(
        [0, entry_len + width - 1],
        dtype=torch.int32,
        device=_DEVICE,
    )
    extension_q_offsets = torch.tensor([0, 1],
                                       dtype=torch.int32,
                                       device=_DEVICE)
    accept_len = torch.tensor([committed], dtype=torch.int32, device=_DEVICE)
    limiting = torch.ones(1, dtype=torch.bool, device=_DEVICE)
    finished = torch.zeros(1, dtype=torch.bool, device=_DEVICE)
    draft_input_ids = torch.full(
        (width, ),
        _OUTPUT_SENTINEL,
        dtype=torch.int32,
        device=_DEVICE,
    )
    selected_token_pos = torch.full(
        (1, ),
        _OUTPUT_SENTINEL,
        dtype=torch.int32,
        device=_DEVICE,
    )
    candidate_active = torch.ones(1, dtype=torch.bool, device=_DEVICE)

    _refresh_and_compare(
        token_rows,
        token_ids_ptrs,
        draft_input_ids,
        selected_token_pos,
        candidate_active,
        refresh_q_offsets,
        refresh_k_offsets,
        extension_q_offsets,
        accept_len,
        limiting,
        finished,
    )

    assert draft_input_ids[:committed].tolist() == list(
        range(100, 100 + committed))
    assert not draft_input_ids[committed:].any()
    assert candidate_active.item() == (committed > 0)
    assert selected_token_pos.item() == max(0, committed - 1)
    _assert_token_guards(backings)


def test_build_draft_refresh_inputs_terminal_and_inherited_finished_rows():
    backings, allocations = _make_token_rows([
        [1, 2, 3, 40, 41, 42, 43, 44],
    ])
    token_rows = [allocations[0], None]
    token_ids_ptrs = _token_pointer_array(token_rows)
    refresh_q_offsets = torch.tensor([0, 4, 8],
                                     dtype=torch.int32,
                                     device=_DEVICE)
    refresh_k_offsets = torch.tensor([0, 6, 12],
                                     dtype=torch.int32,
                                     device=_DEVICE)
    extension_q_offsets = torch.tensor([0, 1, 2],
                                       dtype=torch.int32,
                                       device=_DEVICE)
    accept_len = torch.tensor([3, 0], dtype=torch.int32, device=_DEVICE)
    limiting = torch.ones(2, dtype=torch.bool, device=_DEVICE)
    finished = torch.ones(2, dtype=torch.bool, device=_DEVICE)
    draft_input_ids = torch.full(
        (8, ),
        _OUTPUT_SENTINEL,
        dtype=torch.int32,
        device=_DEVICE,
    )
    selected_token_pos = torch.full(
        (2, ),
        _OUTPUT_SENTINEL,
        dtype=torch.int32,
        device=_DEVICE,
    )
    candidate_active = torch.ones(2, dtype=torch.bool, device=_DEVICE)

    _refresh_and_compare(
        token_rows,
        token_ids_ptrs,
        draft_input_ids,
        selected_token_pos,
        candidate_active,
        refresh_q_offsets,
        refresh_k_offsets,
        extension_q_offsets,
        accept_len,
        limiting,
        finished,
    )

    assert draft_input_ids.tolist() == [40, 41, 42, 0, 0, 0, 0, 0]
    assert selected_token_pos.tolist() == [0, 0]
    assert candidate_active.tolist() == [False, False]
    _assert_token_guards(backings)


@pytest.mark.parametrize('committed', [0, 1])
def test_build_draft_refresh_inputs_ordinary_fallback(committed):
    if committed:
        backings, token_rows = _make_token_rows([[1, 2, 3, 4, 5, 91]])
    else:
        backings, token_rows = [], [None]
    token_ids_ptrs = _token_pointer_array(token_rows)
    draft_backing, draft_input_ids = _guarded_int_vector([_OUTPUT_SENTINEL])
    selected_backing, selected_token_pos = _guarded_int_vector([])
    active_backing, candidate_active = _guarded_bool_vector([])

    _refresh_and_compare(
        token_rows,
        token_ids_ptrs,
        draft_input_ids,
        selected_token_pos,
        candidate_active,
        torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE),
        torch.tensor([0, 5], dtype=torch.int32, device=_DEVICE),
        torch.tensor([0, 0], dtype=torch.int32, device=_DEVICE),
        torch.tensor([committed], dtype=torch.int32, device=_DEVICE),
        torch.ones(1, dtype=torch.bool, device=_DEVICE),
        torch.tensor([not committed], dtype=torch.bool, device=_DEVICE),
    )

    assert draft_input_ids.item() == (91 if committed else 0)
    _assert_vector_guards(draft_backing, _OUTPUT_SENTINEL)
    _assert_vector_guards(selected_backing, _OUTPUT_SENTINEL)
    _assert_vector_guards(active_backing, True)
    _assert_token_guards(backings)


def test_build_draft_refresh_inputs_empty_batches_and_zero_packed_rows():
    empty_i32 = torch.empty(0, dtype=torch.int32, device=_DEVICE)
    empty_i64 = torch.empty(0, dtype=torch.int64, device=_DEVICE)
    empty_bool = torch.empty(0, dtype=torch.bool, device=_DEVICE)
    zero_offsets = torch.zeros(1, dtype=torch.int32, device=_DEVICE)

    build_draft_refresh_inputs(
        empty_i32,
        empty_i32,
        empty_bool,
        empty_i64,
        zero_offsets,
        zero_offsets,
        zero_offsets,
        empty_i32,
        empty_bool,
        empty_bool,
    )

    batch_size = 4
    pointer_array = torch.zeros(batch_size, dtype=torch.int64, device=_DEVICE)
    offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=_DEVICE)
    accept_len = torch.full(
        (batch_size, ),
        _POISON_ACCEPT_LEN,
        dtype=torch.int32,
        device=_DEVICE,
    )
    limiting = torch.ones(batch_size, dtype=torch.bool, device=_DEVICE)
    finished = torch.ones(batch_size, dtype=torch.bool, device=_DEVICE)
    build_draft_refresh_inputs(
        empty_i32,
        empty_i32,
        empty_bool,
        pointer_array,
        offsets,
        torch.full_like(offsets, _POISON_SEQUENCE_LENGTH),
        offsets,
        accept_len,
        limiting,
        finished,
    )


def test_build_draft_refresh_inputs_nondefault_stream_successive_snapshots():
    _, token_rows = _make_token_rows([[1, 2, 3, 31, 32, 33, 34]])
    token_ids_ptrs = _token_pointer_array(token_rows)
    q_offsets = torch.tensor([0, 3], dtype=torch.int32, device=_DEVICE)
    k_offsets = torch.tensor([0, 5], dtype=torch.int32, device=_DEVICE)
    extension_offsets = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    accept_len = torch.tensor([1], dtype=torch.int32, device=_DEVICE)
    limiting = torch.ones(1, dtype=torch.bool, device=_DEVICE)
    finished = torch.zeros(1, dtype=torch.bool, device=_DEVICE)
    draft_input_ids = torch.empty(3, dtype=torch.int32, device=_DEVICE)
    selected_token_pos = torch.empty(1, dtype=torch.int32, device=_DEVICE)
    candidate_active = torch.empty(1, dtype=torch.bool, device=_DEVICE)
    current_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current_stream)

    with torch.cuda.stream(stream):
        build_draft_refresh_inputs(
            draft_input_ids,
            selected_token_pos,
            candidate_active,
            token_ids_ptrs,
            q_offsets,
            k_offsets,
            extension_offsets,
            accept_len,
            limiting,
            finished,
        )
        first_snapshot = (
            draft_input_ids.clone(),
            selected_token_pos.clone(),
            candidate_active.clone(),
        )
        accept_len.fill_(3)
        build_draft_refresh_inputs(
            draft_input_ids,
            selected_token_pos,
            candidate_active,
            token_ids_ptrs,
            q_offsets,
            k_offsets,
            extension_offsets,
            accept_len,
            limiting,
            finished,
        )
        second_snapshot = (
            draft_input_ids.clone(),
            selected_token_pos.clone(),
            candidate_active.clone(),
        )

    current_stream.wait_stream(stream)
    assert first_snapshot[0].tolist() == [31, 0, 0]
    assert first_snapshot[1].item() == 0
    assert first_snapshot[2].item()
    assert second_snapshot[0].tolist() == [31, 32, 33]
    assert second_snapshot[1].item() == 2
    assert second_snapshot[2].item()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('proposal_index', range(_SPECULATIVE_K))
def test_draft_argmax_random_rows_every_proposal_index(dtype, proposal_index):
    backings, allocations = _make_token_rows([
        list(range(100, 132)),
        list(range(200, 232)),
        list(range(300, 332)),
    ])
    token_rows = [allocations[2], None, allocations[0], None, allocations[1]]
    token_ids_ptrs = _token_pointer_array(token_rows)
    extension_offsets = torch.tensor(
        [0, 1, 1, 2, 2, 3],
        dtype=torch.int32,
        device=_DEVICE,
    )
    candidate_active = torch.ones(3, dtype=torch.bool, device=_DEVICE)
    entry_lengths = torch.tensor(
        [4, _POISON_SEQUENCE_LENGTH, 7, _POISON_SEQUENCE_LENGTH, 9],
        dtype=torch.int32,
        device=_DEVICE,
    )
    accept_len = torch.tensor(
        [2, _POISON_ACCEPT_LEN, 3, _POISON_ACCEPT_LEN, 1],
        dtype=torch.int32,
        device=_DEVICE,
    )
    vocab_size = 37
    logits = torch.randn(3, vocab_size + 5, dtype=dtype, device=_DEVICE)
    logits[:, vocab_size:] = torch.inf
    proposal_backing, proposal_ids = _guarded_int_vector([_OUTPUT_SENTINEL] *
                                                         3)

    _argmax_and_compare(
        logits,
        token_rows,
        proposal_ids,
        token_ids_ptrs,
        extension_offsets,
        candidate_active,
        entry_lengths,
        accept_len,
        proposal_index,
        vocab_size,
    )

    _assert_vector_guards(proposal_backing, _OUTPUT_SENTINEL)
    _assert_token_guards(backings)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_draft_argmax_concrete_vocab_size(dtype):
    vocab_size = 151936
    backings, token_rows = _make_token_rows([list(range(32))])
    token_ids_ptrs = _token_pointer_array(token_rows)
    logits = torch.randn(
        1,
        vocab_size + 8,
        dtype=dtype,
        device=_DEVICE,
    )
    logits[:, vocab_size:] = torch.inf
    proposal_ids = torch.full(
        (1, ),
        _OUTPUT_SENTINEL,
        dtype=torch.int32,
        device=_DEVICE,
    )

    _argmax_and_compare(
        logits,
        token_rows,
        proposal_ids,
        token_ids_ptrs,
        torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE),
        torch.ones(1, dtype=torch.bool, device=_DEVICE),
        torch.tensor([5], dtype=torch.int32, device=_DEVICE),
        torch.tensor([3], dtype=torch.int32, device=_DEVICE),
        2,
        vocab_size,
    )

    _assert_token_guards(backings)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_draft_argmax_ties_infinities_signed_zero_and_nan(dtype):
    values = [
        [-9.0, 4.0, 1.0, 4.0, -2.0, -3.0],
        [0.0, torch.inf, 5.0, torch.inf, -1.0, -2.0],
        [-3.0, -0.0, 0.0, -2.0, -4.0, -5.0],
        [-torch.inf] * 6,
        [torch.nan] * 6,
        [torch.nan, -2.0, 7.0, torch.nan, 7.0, -torch.inf],
    ]
    expected_token_ids = [1, 1, 1, 0, 0, 2]
    logits = torch.tensor(values, dtype=dtype, device=_DEVICE)
    logits = torch.cat([
        logits,
        torch.full(
            (len(values), 3),
            torch.inf,
            dtype=dtype,
            device=_DEVICE,
        ),
    ],
                       dim=1)
    backings, token_rows = _make_token_rows([list(range(20)) for _ in values])
    token_ids_ptrs = _token_pointer_array(token_rows)
    batch_size = len(values)
    offsets = torch.arange(
        batch_size + 1,
        dtype=torch.int32,
        device=_DEVICE,
    )
    proposal_backing, proposal_ids = _guarded_int_vector([_OUTPUT_SENTINEL] *
                                                         batch_size)

    _argmax_and_compare(
        logits,
        token_rows,
        proposal_ids,
        token_ids_ptrs,
        offsets,
        torch.ones(batch_size, dtype=torch.bool, device=_DEVICE),
        torch.full((batch_size, ), 4, dtype=torch.int32, device=_DEVICE),
        torch.arange(batch_size, dtype=torch.int32, device=_DEVICE),
        1,
        6,
    )

    assert proposal_ids.tolist() == expected_token_ids
    _assert_vector_guards(proposal_backing, _OUTPUT_SENTINEL)
    _assert_token_guards(backings)


def test_draft_argmax_inactive_and_noncandidate_poison_rows():
    backings, allocations = _make_token_rows([
        list(range(40)),
        list(range(100, 140)),
    ])
    token_rows = [allocations[1], None, None, None, None, allocations[0]]
    token_ids_ptrs = _token_pointer_array(token_rows)
    extension_offsets = torch.tensor(
        [0, 1, 1, 2, 2, 2, 3],
        dtype=torch.int32,
        device=_DEVICE,
    )
    candidate_active = torch.tensor(
        [True, False, True],
        dtype=torch.bool,
        device=_DEVICE,
    )
    entry_lengths = torch.tensor(
        [
            5,
            _POISON_SEQUENCE_LENGTH,
            _POISON_SEQUENCE_LENGTH,
            _POISON_SEQUENCE_LENGTH,
            _POISON_SEQUENCE_LENGTH,
            8,
        ],
        dtype=torch.int32,
        device=_DEVICE,
    )
    accept_len = torch.tensor(
        [
            2,
            _POISON_ACCEPT_LEN,
            _POISON_ACCEPT_LEN,
            _POISON_ACCEPT_LEN,
            _POISON_ACCEPT_LEN,
            3,
        ],
        dtype=torch.int32,
        device=_DEVICE,
    )
    logits = torch.tensor(
        [
            [1.0, 9.0, 3.0, 2.0],
            [torch.nan, torch.nan, torch.nan, torch.nan],
            [8.0, 2.0, 11.0, 5.0],
        ],
        dtype=torch.float16,
        device=_DEVICE,
    )
    proposal_backing, proposal_ids = _guarded_int_vector([_OUTPUT_SENTINEL] *
                                                         3)

    _argmax_and_compare(
        logits,
        token_rows,
        proposal_ids,
        token_ids_ptrs,
        extension_offsets,
        candidate_active,
        entry_lengths,
        accept_len,
        2,
        4,
    )

    assert proposal_ids.tolist() == [1, 0, 2]
    _assert_vector_guards(proposal_backing, _OUTPUT_SENTINEL)
    _assert_token_guards(backings)


def test_draft_argmax_empty_batch_and_zero_candidates():
    empty_i32 = torch.empty(0, dtype=torch.int32, device=_DEVICE)
    empty_i64 = torch.empty(0, dtype=torch.int64, device=_DEVICE)
    empty_bool = torch.empty(0, dtype=torch.bool, device=_DEVICE)
    empty_logits = torch.empty(
        (0, 8),
        dtype=torch.float16,
        device=_DEVICE,
    )
    zero_offsets = torch.zeros(1, dtype=torch.int32, device=_DEVICE)

    draft_argmax_and_store_token(
        empty_logits,
        empty_i32,
        empty_i64,
        zero_offsets,
        empty_bool,
        empty_i32,
        empty_i32,
        0,
        8,
    )

    batch_size = 4
    draft_argmax_and_store_token(
        empty_logits,
        empty_i32,
        torch.zeros(batch_size, dtype=torch.int64, device=_DEVICE),
        torch.zeros(batch_size + 1, dtype=torch.int32, device=_DEVICE),
        empty_bool,
        torch.full(
            (batch_size, ),
            _POISON_SEQUENCE_LENGTH,
            dtype=torch.int32,
            device=_DEVICE,
        ),
        torch.full(
            (batch_size, ),
            _POISON_ACCEPT_LEN,
            dtype=torch.int32,
            device=_DEVICE,
        ),
        0,
        8,
    )


def test_draft_argmax_nondefault_stream_successive_proposal_indices():
    _, token_rows = _make_token_rows([[10] * 24])
    token_ids_ptrs = _token_pointer_array(token_rows)
    offsets = torch.tensor([0, 1], dtype=torch.int32, device=_DEVICE)
    active = torch.ones(1, dtype=torch.bool, device=_DEVICE)
    entry_lengths = torch.tensor([4], dtype=torch.int32, device=_DEVICE)
    accept_len = torch.tensor([2], dtype=torch.int32, device=_DEVICE)
    logits = torch.tensor(
        [[1.0, 2.0, 9.0, 3.0]],
        dtype=torch.bfloat16,
        device=_DEVICE,
    )
    proposal_ids = torch.empty(1, dtype=torch.int32, device=_DEVICE)
    current_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current_stream)

    with torch.cuda.stream(stream):
        draft_argmax_and_store_token(
            logits,
            proposal_ids,
            token_ids_ptrs,
            offsets,
            active,
            entry_lengths,
            accept_len,
            0,
            4,
        )
        first_snapshot = proposal_ids.clone(), token_rows[0].clone()
        logits.copy_(
            torch.tensor(
                [[8.0, 2.0, 1.0, 11.0]],
                dtype=torch.bfloat16,
                device=_DEVICE,
            ))
        draft_argmax_and_store_token(
            logits,
            proposal_ids,
            token_ids_ptrs,
            offsets,
            active,
            entry_lengths,
            accept_len,
            1,
            4,
        )
        second_snapshot = proposal_ids.clone(), token_rows[0].clone()

    current_stream.wait_stream(stream)
    assert first_snapshot[0].item() == 2
    assert first_snapshot[1][6].item() == 2
    assert second_snapshot[0].item() == 3
    assert second_snapshot[1][6].item() == 2
    assert second_snapshot[1][7].item() == 3
