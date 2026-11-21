from dataclasses import dataclass

import pytest
import torch

from .speculative_sampling import (
    allocate_random_states,
    append_one_token_and_advance_sequence,
    initialize_random_states,
    sample_processed_probabilities,
    verify_target_block,
)

_MAX_LOGPROB = 1024
_OUTPUT_GUARD = 16
_SELECTED_GUARD_VALUE = 0x5A5A5A5A


@dataclass
class GuardedOutput:
    storage: torch.Tensor
    value: torch.Tensor
    guard_value: int | float | bool

    def assert_guards(self) -> None:
        leading = self.storage[:_OUTPUT_GUARD]
        trailing = self.storage[-_OUTPUT_GUARD:]
        expected_leading = torch.full_like(leading, self.guard_value)
        expected_trailing = torch.full_like(trailing, self.guard_value)
        assert torch.equal(leading, expected_leading)
        assert torch.equal(trailing, expected_trailing)


def _guarded_output(batch_size: int, dtype: torch.dtype,
                    device: torch.device | str,
                    interior_value: int | float | bool) -> GuardedOutput:
    guard_value = True if dtype == torch.bool else _SELECTED_GUARD_VALUE
    storage = torch.full((batch_size + 2 * _OUTPUT_GUARD, ),
                         guard_value,
                         dtype=dtype,
                         device=device)
    value = storage[_OUTPUT_GUARD:_OUTPUT_GUARD + batch_size]
    value.fill_(interior_value)
    return GuardedOutput(storage=storage, value=value, guard_value=guard_value)


def _optional_sampling_outputs(
    batch_size: int,
    device: torch.device | str,
) -> tuple[GuardedOutput, GuardedOutput, GuardedOutput]:
    sampled_logprobs = _guarded_output(batch_size * _MAX_LOGPROB,
                                       torch.float32, device, -91.0)
    sampled_indexes = _guarded_output(batch_size * _MAX_LOGPROB, torch.int32,
                                      device, -193)
    sampled_nums = _guarded_output(batch_size, torch.int32, device, -307)
    return sampled_logprobs, sampled_indexes, sampled_nums


def _strided_rows(
        batch_size: int,
        width: int,
        padding: int,
        storage_offset: int,
        dtype: torch.dtype,
        device: torch.device | str,
        fill_value: float | int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    stride = width + padding
    storage_size = storage_offset + (batch_size - 1) * stride + width + 7
    storage = torch.full((storage_size, ),
                         fill_value,
                         dtype=dtype,
                         device=device)
    view = torch.as_strided(storage, (batch_size, width), (stride, 1),
                            storage_offset=storage_offset)
    return storage, view


def _fixed_seeds(count: int,
                 device: torch.device | str,
                 start: int = 1729) -> torch.Tensor:
    return torch.arange(start, start + count, dtype=torch.int64,
                        device=device).to(torch.uint64)


def _initialize_state_storage(random_state_count: int,
                              device: torch.device | str,
                              start_seed: int = 1729) -> torch.Tensor:
    random_states = allocate_random_states(random_state_count, device)
    initialize_random_states(
        random_states, _fixed_seeds(random_state_count, device, start_seed),
        torch.ones(random_state_count, dtype=torch.bool, device=device))
    return random_states


def test_processed_sampling_null_mask_matches_all_true_with_optional_outputs(
) -> None:
    device = torch.device('cuda')
    batch_size = 4
    width = 3
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        probability_storage, probabilities = _strided_rows(
            batch_size,
            width,
            padding=7,
            storage_offset=5,
            dtype=torch.float32,
            device=device,
            fill_value=-17.0,
        )
        index_storage, indices = _strided_rows(
            batch_size,
            width,
            padding=7,
            storage_offset=3,
            dtype=torch.int32,
            device=device,
            fill_value=-29,
        )
        probabilities.zero_()
        probabilities[:, 0] = 1.0
        indices.copy_(
            torch.arange(101,
                         101 + batch_size * width,
                         dtype=torch.int32,
                         device=device).view(batch_size, width))
        kept = torch.ones(batch_size, dtype=torch.int32, device=device)
        state_indices = torch.tensor([4, 1, 3, 0],
                                     dtype=torch.int32,
                                     device=device)

        first_states = allocate_random_states(5, device)
        second_states = allocate_random_states(5, device)
        seeds = _fixed_seeds(5, device, start=31001)
        initialize = torch.ones(5, dtype=torch.bool, device=device)
        initialize_random_states(first_states, seeds, initialize)
        initialize_random_states(second_states, seeds, initialize)

        first_selected = _guarded_output(batch_size, torch.int32, device, -777)
        second_selected = _guarded_output(batch_size, torch.int32, device,
                                          -777)
        first_logprobs, first_indexes, first_nums = (
            _optional_sampling_outputs(batch_size, device))
        second_logprobs, second_indexes, second_nums = (
            _optional_sampling_outputs(batch_size, device))
        all_true = torch.ones(batch_size, dtype=torch.bool, device=device)
        probability_snapshot = probability_storage.clone()
        index_snapshot = index_storage.clone()

        sample_processed_probabilities(
            probabilities,
            indices,
            kept,
            first_states,
            state_indices,
            None,
            first_selected.value,
            first_logprobs.value.view(batch_size, _MAX_LOGPROB),
            first_indexes.value.view(batch_size, _MAX_LOGPROB),
            first_nums.value,
        )
        sample_processed_probabilities(
            probabilities,
            indices,
            kept,
            second_states,
            state_indices,
            all_true,
            second_selected.value,
            second_logprobs.value.view(batch_size, _MAX_LOGPROB),
            second_indexes.value.view(batch_size, _MAX_LOGPROB),
            second_nums.value,
        )

    stream.synchronize()

    assert probabilities.stride(0) == indices.stride(0) == width + 7
    assert probabilities.storage_offset() == 5
    assert indices.storage_offset() == 3
    assert torch.equal(first_selected.value, indices[:, 0])
    assert torch.equal(first_selected.storage, second_selected.storage)
    assert torch.equal(first_logprobs.storage, second_logprobs.storage)
    assert torch.equal(first_indexes.storage, second_indexes.storage)
    assert torch.equal(first_nums.storage, second_nums.storage)
    assert torch.equal(first_states, second_states)
    assert torch.equal(first_nums.value, torch.ones_like(first_nums.value))
    assert torch.equal(
        first_indexes.value.view(batch_size, _MAX_LOGPROB)[:, 0], indices[:,
                                                                          0])
    assert torch.equal(
        first_logprobs.value.view(batch_size, _MAX_LOGPROB)[:, 0],
        torch.zeros(batch_size, dtype=torch.float32, device=device))
    assert torch.equal(probability_storage, probability_snapshot)
    assert torch.equal(index_storage, index_snapshot)
    for output in (first_selected, second_selected, first_logprobs,
                   second_logprobs, first_indexes, second_indexes, first_nums,
                   second_nums):
        output.assert_guards()


def test_processed_sampling_all_false_mask_ignores_poisoned_rows() -> None:
    device = torch.device('cuda')
    batch_size = 3
    probabilities = torch.full((batch_size, 4),
                               -12345.0,
                               dtype=torch.float32,
                               device=device)
    indices = torch.full((batch_size, 4),
                         -23456,
                         dtype=torch.int32,
                         device=device)
    kept = torch.tensor([-101, -202, -303], dtype=torch.int32, device=device)
    state_indices = torch.tensor([-401, -502, -603],
                                 dtype=torch.int32,
                                 device=device)
    sample_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    selected = _guarded_output(batch_size, torch.int32, device, -777)
    sampled_logprobs, sampled_indexes, sampled_nums = (
        _optional_sampling_outputs(batch_size, device))
    probability_snapshot = probabilities.clone()
    index_snapshot = indices.clone()

    sample_processed_probabilities(
        probabilities,
        indices,
        kept,
        torch.empty(0, dtype=torch.uint8, device=device),
        state_indices,
        sample_mask,
        selected.value,
        sampled_logprobs.value.view(batch_size, _MAX_LOGPROB),
        sampled_indexes.value.view(batch_size, _MAX_LOGPROB),
        sampled_nums.value,
    )
    torch.cuda.current_stream(device).synchronize()

    assert torch.equal(selected.value, torch.full_like(selected.value, -777))
    assert torch.equal(sampled_logprobs.value,
                       torch.full_like(sampled_logprobs.value, -91.0))
    assert torch.equal(sampled_indexes.value,
                       torch.full_like(sampled_indexes.value, -193))
    assert torch.equal(sampled_nums.value,
                       torch.full_like(sampled_nums.value, -307))
    assert torch.equal(probabilities, probability_snapshot)
    assert torch.equal(indices, index_snapshot)
    selected.assert_guards()
    sampled_logprobs.assert_guards()
    sampled_indexes.assert_guards()
    sampled_nums.assert_guards()


def test_processed_sampling_mixed_mask_rng_and_all_null_outputs() -> None:
    device = torch.device('cuda')
    batch_size = 5
    width = 4
    sample_mask = torch.tensor([True, False, True, False, True],
                               dtype=torch.bool,
                               device=device)
    probabilities = torch.full((batch_size, width),
                               -73.0,
                               dtype=torch.float32,
                               device=device)
    indices = torch.full((batch_size, width),
                         -89,
                         dtype=torch.int32,
                         device=device)
    kept = torch.tensor([1, -103, 1, -107, 1],
                        dtype=torch.int32,
                        device=device)
    selected_values = torch.tensor([501, -1, 701, -1, 901],
                                   dtype=torch.int32,
                                   device=device)
    probabilities[sample_mask, 0] = 1.0
    indices[:, 0] = selected_values
    state_indices = torch.tensor([5, 3, 1, -109, 4],
                                 dtype=torch.int32,
                                 device=device)

    mixed_states = allocate_random_states(6, device)
    active_only_states = allocate_random_states(6, device)
    null_output_states = allocate_random_states(6, device)
    seeds = _fixed_seeds(6, device, start=33001)
    initialize = torch.ones(6, dtype=torch.bool, device=device)
    initialize_random_states(mixed_states, seeds, initialize)
    initialize_random_states(active_only_states, seeds, initialize)
    initialize_random_states(null_output_states, seeds, initialize)
    initial_states = mixed_states.clone()

    mixed_selected = _guarded_output(batch_size, torch.int32, device, -777)
    null_output_selected = _guarded_output(batch_size, torch.int32, device,
                                           -777)
    active_only_selected = _guarded_output(3, torch.int32, device, -777)
    sampled_logprobs, sampled_indexes, sampled_nums = (
        _optional_sampling_outputs(batch_size, device))
    probability_snapshot = probabilities.clone()
    index_snapshot = indices.clone()
    active_probabilities = probabilities[sample_mask].contiguous()
    active_indices = indices[sample_mask].contiguous()
    active_kept = kept[sample_mask].contiguous()
    active_state_indices = state_indices[sample_mask].contiguous()

    sample_processed_probabilities(
        probabilities,
        indices,
        kept,
        mixed_states,
        state_indices,
        sample_mask,
        mixed_selected.value,
        sampled_logprobs.value.view(batch_size, _MAX_LOGPROB),
        sampled_indexes.value.view(batch_size, _MAX_LOGPROB),
        sampled_nums.value,
    )
    sample_processed_probabilities(
        probabilities,
        indices,
        kept,
        null_output_states,
        state_indices,
        sample_mask,
        null_output_selected.value,
        None,
        None,
        None,
    )
    sample_processed_probabilities(
        active_probabilities,
        active_indices,
        active_kept,
        active_only_states,
        active_state_indices,
        None,
        active_only_selected.value,
        None,
        None,
        None,
    )
    torch.cuda.current_stream(device).synchronize()

    expected_active = selected_values[sample_mask]
    assert torch.equal(mixed_selected.value[sample_mask], expected_active)
    assert torch.equal(null_output_selected.value[sample_mask],
                       expected_active)
    assert torch.equal(active_only_selected.value, expected_active)
    assert torch.equal(
        mixed_selected.value[~sample_mask],
        torch.full((2, ), -777, dtype=torch.int32, device=device))
    assert torch.equal(
        null_output_selected.value[~sample_mask],
        torch.full((2, ), -777, dtype=torch.int32, device=device))
    assert torch.equal(mixed_states, active_only_states)
    assert torch.equal(mixed_states, null_output_states)
    assert not torch.equal(mixed_states, initial_states)

    logprob_rows = sampled_logprobs.value.view(batch_size, _MAX_LOGPROB)
    index_rows = sampled_indexes.value.view(batch_size, _MAX_LOGPROB)
    assert torch.equal(logprob_rows[sample_mask, 0],
                       torch.zeros(3, dtype=torch.float32, device=device))
    assert torch.equal(index_rows[sample_mask, 0], expected_active)
    assert torch.equal(
        logprob_rows[~sample_mask],
        torch.full((2, _MAX_LOGPROB),
                   -91.0,
                   dtype=torch.float32,
                   device=device))
    assert torch.equal(
        index_rows[~sample_mask],
        torch.full((2, _MAX_LOGPROB), -193, dtype=torch.int32, device=device))
    assert torch.equal(sampled_nums.value[sample_mask],
                       torch.ones(3, dtype=torch.int32, device=device))
    assert torch.equal(
        sampled_nums.value[~sample_mask],
        torch.full((2, ), -307, dtype=torch.int32, device=device))
    assert torch.equal(probabilities, probability_snapshot)
    assert torch.equal(indices, index_snapshot)
    mixed_selected.assert_guards()
    null_output_selected.assert_guards()
    active_only_selected.assert_guards()
    sampled_logprobs.assert_guards()
    sampled_indexes.assert_guards()
    sampled_nums.assert_guards()


def test_processed_sampling_and_append_empty_views() -> None:
    device = torch.device('cuda')
    width = 7
    probability_storage = torch.empty(width + 11,
                                      dtype=torch.float32,
                                      device=device)
    index_storage = torch.empty(width + 11, dtype=torch.int32, device=device)
    probabilities = torch.as_strided(probability_storage, (0, width),
                                     (width + 3, 1),
                                     storage_offset=4)
    indices = torch.as_strided(index_storage, (0, width), (width + 3, 1),
                               storage_offset=2)
    empty_int = torch.empty(0, dtype=torch.int32, device=device)
    empty_bool = torch.empty(0, dtype=torch.bool, device=device)

    sample_processed_probabilities(
        probabilities,
        indices,
        empty_int,
        torch.empty(0, dtype=torch.uint8, device=device),
        empty_int.clone(),
        empty_bool,
        empty_int.clone(),
        torch.empty((0, _MAX_LOGPROB), dtype=torch.float32, device=device),
        torch.empty((0, _MAX_LOGPROB), dtype=torch.int32, device=device),
        empty_int.clone(),
    )
    append_one_token_and_advance_sequence(
        torch.empty(0, dtype=torch.int64, device=device),
        empty_int.clone(),
        empty_int.clone(),
    )
    torch.cuda.current_stream(device).synchronize()


def test_append_one_token_uses_pointer_order_and_distinct_lengths() -> None:
    device = torch.device('cuda')
    batch_size = 4
    session_length = 10
    row_guard = 6
    row_guard_value = -123456789
    row_initial_value = -41
    allocations = [
        torch.full((session_length + 2 * row_guard, ),
                   row_guard_value,
                   dtype=torch.int32,
                   device=device) for _ in range(batch_size)
    ]
    rows = []
    for storage in allocations:
        row = storage[row_guard:row_guard + session_length]
        row.fill_(row_initial_value)
        rows.append(row)

    pointer_order = [2, 0, 3, 1]
    token_ids_ptrs = torch.tensor(
        [rows[index].data_ptr() for index in pointer_order],
        dtype=torch.int64,
        device=device)
    selected_values = [1101, 1102, 1103, 1104]
    selected_tokens = torch.tensor(selected_values,
                                   dtype=torch.int32,
                                   device=device)
    starting_lengths = [0, 3, 1, 5]
    sequence_length = torch.tensor(starting_lengths,
                                   dtype=torch.int32,
                                   device=device)
    allocation_snapshots = [storage.clone() for storage in allocations]

    append_one_token_and_advance_sequence(token_ids_ptrs, selected_tokens,
                                          sequence_length)
    torch.cuda.current_stream(device).synchronize()

    assert torch.equal(
        sequence_length,
        torch.tensor([value + 1 for value in starting_lengths],
                     dtype=torch.int32,
                     device=device))
    for logical_row, physical_row in enumerate(pointer_order):
        expected = allocation_snapshots[physical_row].clone()
        expected[row_guard +
                 starting_lengths[logical_row]] = selected_values[logical_row]
        assert torch.equal(allocations[physical_row], expected)
        assert torch.equal(
            allocations[physical_row][:row_guard],
            torch.full((row_guard, ),
                       row_guard_value,
                       dtype=torch.int32,
                       device=device))
        assert torch.equal(
            allocations[physical_row][-row_guard:],
            torch.full((row_guard, ),
                       row_guard_value,
                       dtype=torch.int32,
                       device=device))


def test_append_successive_launches_preserve_nondefault_stream_order() -> None:
    device = torch.device('cuda')
    batch_size = 2
    session_length = 8
    row_guard = 4
    row_guard_value = -987654321
    row_initial_value = -53
    allocations = [
        torch.full((session_length + 2 * row_guard, ),
                   row_guard_value,
                   dtype=torch.int32,
                   device=device) for _ in range(batch_size)
    ]
    rows = []
    for storage in allocations:
        row = storage[row_guard:row_guard + session_length]
        row.fill_(row_initial_value)
        rows.append(row)

    pointer_order = [1, 0]
    logical_rows = [rows[index] for index in pointer_order]
    token_ids_ptrs = torch.tensor([row.data_ptr() for row in logical_rows],
                                  dtype=torch.int64,
                                  device=device)
    starting_lengths = [1, 3]
    sequence_length = torch.tensor(starting_lengths,
                                   dtype=torch.int32,
                                   device=device)
    first_values = [2101, 2102]
    second_values = [2201, 2202]
    stream = torch.cuda.Stream(device=device)
    torch.cuda.current_stream(device).synchronize()

    with torch.cuda.stream(stream):
        first_selected_tokens = torch.tensor(first_values,
                                             dtype=torch.int32,
                                             device=device)
        second_selected_tokens = torch.tensor(second_values,
                                              dtype=torch.int32,
                                              device=device)
        append_one_token_and_advance_sequence(
            token_ids_ptrs,
            first_selected_tokens,
            sequence_length,
        )
        first_length_snapshot = sequence_length.clone()
        first_row_snapshot = torch.stack(logical_rows)
        append_one_token_and_advance_sequence(
            token_ids_ptrs,
            second_selected_tokens,
            sequence_length,
        )
        final_length_snapshot = sequence_length.clone()
        final_row_snapshot = torch.stack(logical_rows)

    stream.synchronize()

    expected_first_rows = torch.full((batch_size, session_length),
                                     row_initial_value,
                                     dtype=torch.int32,
                                     device=device)
    expected_final_rows = expected_first_rows.clone()
    for row in range(batch_size):
        expected_first_rows[row, starting_lengths[row]] = first_values[row]
        expected_final_rows[row, starting_lengths[row]] = first_values[row]
        expected_final_rows[row,
                            starting_lengths[row] + 1] = second_values[row]
    assert first_length_snapshot.device.type == 'cuda'
    assert torch.equal(
        first_length_snapshot,
        torch.tensor([value + 1 for value in starting_lengths],
                     dtype=torch.int32,
                     device=device))
    assert torch.equal(first_row_snapshot, expected_first_rows)
    assert torch.equal(
        final_length_snapshot,
        torch.tensor([value + 2 for value in starting_lengths],
                     dtype=torch.int32,
                     device=device))
    assert torch.equal(final_row_snapshot, expected_final_rows)
    assert torch.equal(sequence_length, final_length_snapshot)
    for storage in allocations:
        assert torch.equal(
            storage[:row_guard],
            torch.full((row_guard, ),
                       row_guard_value,
                       dtype=torch.int32,
                       device=device))
        assert torch.equal(
            storage[-row_guard:],
            torch.full((row_guard, ),
                       row_guard_value,
                       dtype=torch.int32,
                       device=device))


def _block_token_rows(
    request_count: int,
    device: torch.device,
    row_width: int = 24,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    guard = 5
    guard_value = -123456789
    allocations = [
        torch.full((row_width + 2 * guard, ),
                   guard_value,
                   dtype=torch.int32,
                   device=device) for _ in range(request_count)
    ]
    rows = [storage[guard:guard + row_width] for storage in allocations]
    for row in rows:
        row.fill_(-41)
    pointers = torch.tensor([row.data_ptr() for row in rows],
                            dtype=torch.int64,
                            device=device)
    return allocations, rows, pointers


def _assert_block_row_guards(allocations: list[torch.Tensor]) -> None:
    for storage in allocations:
        assert torch.equal(storage[:5],
                           torch.full_like(storage[:5], -123456789))
        assert torch.equal(storage[-5:],
                           torch.full_like(storage[-5:], -123456789))


def test_verify_target_block_mixed_rows_padded_position_major_inputs() -> None:
    device = torch.device('cuda')
    request_count = 5
    generation_count = 4
    position_count = 4
    rows_count = generation_count * position_count
    width = 3

    _, probabilities = _strided_rows(rows_count,
                                     width,
                                     padding=7,
                                     storage_offset=3,
                                     dtype=torch.float32,
                                     device=device)
    _, token_ids = _strided_rows(rows_count,
                                 width,
                                 padding=11,
                                 storage_offset=5,
                                 dtype=torch.int32,
                                 device=device)
    probabilities.zero_()
    token_ids.fill_(-1)

    def point(position: int, generation: int, token: int) -> None:
        flat = position * generation_count + generation
        probabilities[flat, 0] = 1.0
        token_ids[flat, 0] = token

    point(0, 0, 101)
    for position, token in enumerate((201, 202, 203, 204)):
        point(position, 1, token)
    probabilities[2, :2] = torch.tensor([0.0, 1.0], device=device)
    token_ids[2, :2] = torch.tensor([301, 399],
                                    dtype=torch.int32,
                                    device=device)
    for position, token in enumerate((401, 498, 403, 404)):
        point(position, 3, token)

    kept = torch.ones(rows_count, dtype=torch.int32, device=device)
    kept[2] = 2
    draft_storage = torch.full(
        ((position_count - 1) * (generation_count + 2), ),
        -17,
        dtype=torch.int32,
        device=device)
    draft_ids = torch.as_strided(draft_storage,
                                 (position_count - 1, generation_count),
                                 (generation_count + 2, 1))
    draft_ids.copy_(
        torch.tensor(
            [[0, 201, 301, 401], [0, 202, 302, 402], [0, 203, 303, 403]],
            dtype=torch.int32,
            device=device))
    greedy = torch.ones(rows_count, dtype=torch.bool, device=device)
    greedy[2] = False
    logits_active = torch.ones(rows_count, dtype=torch.bool, device=device)
    random_states = _initialize_state_storage(generation_count, device, 41001)
    state_indices = torch.arange(generation_count,
                                 dtype=torch.int32,
                                 device=device)
    allocations, token_rows, token_ptrs = _block_token_rows(
        request_count, device)
    entry_lengths = torch.tensor([2, 3, 5, 1, 4],
                                 dtype=torch.int32,
                                 device=device)
    offsets = torch.tensor([0, 1, 2, 2, 3, 4],
                           dtype=torch.int32,
                           device=device)
    speculative = torch.tensor([False, True, False, True, True],
                               dtype=torch.bool,
                               device=device)
    selected_storage = torch.full((request_count * (position_count + 2), ),
                                  -777,
                                  dtype=torch.int32,
                                  device=device)
    selected = torch.as_strided(selected_storage,
                                (request_count, position_count),
                                (position_count + 2, 1))
    accept_len = torch.full((request_count, ),
                            -9,
                            dtype=torch.int32,
                            device=device)
    accepted_count = torch.full((request_count, ),
                                -7,
                                dtype=torch.int32,
                                device=device)

    verify_target_block(probabilities, token_ids, kept, draft_ids, greedy,
                        logits_active, random_states, state_indices,
                        token_ptrs, entry_lengths, offsets, speculative,
                        selected, accept_len, accepted_count, position_count)
    torch.cuda.current_stream(device).synchronize()

    assert accept_len.tolist() == [1, 4, -9, 1, 2]
    assert selected[0, 0].item() == 101
    assert selected[1].tolist() == [201, 202, 203, 204]
    assert selected[3, :2].tolist() == [399, -777]
    assert selected[4, :2].tolist() == [401, 498]
    assert accepted_count.tolist() == [-7, 3, -7, 0, 1]
    assert token_rows[0][2].item() == 101
    assert token_rows[1][3:7].tolist() == [201, 202, 203, 204]
    assert token_rows[2].eq(-41).all()
    assert token_rows[3][1].item() == 399
    assert token_rows[4][4:6].tolist() == [401, 498]
    assert selected_storage.view(request_count, position_count +
                                 2)[:, position_count:].eq(-777).all()
    _assert_block_row_guards(allocations)


@pytest.mark.parametrize('rejection_position', [0, 1, 2])
def test_verify_target_block_rejection_position_and_rng_advance(
        rejection_position: int) -> None:
    device = torch.device('cuda')
    position_count = 4
    probabilities = torch.zeros((position_count, 2),
                                dtype=torch.float32,
                                device=device)
    token_ids = torch.empty((position_count, 2),
                            dtype=torch.int32,
                            device=device)
    drafts = torch.tensor([[501], [502], [503]],
                          dtype=torch.int32,
                          device=device)
    for position in range(position_count - 1):
        token_ids[position] = torch.tensor([501 + position, 601 + position],
                                           dtype=torch.int32,
                                           device=device)
        if position == rejection_position:
            probabilities[position] = torch.tensor([0.0, 1.0], device=device)
        else:
            probabilities[position, 0] = 1.0
    probabilities[-1, 0] = 1.0
    token_ids[-1] = torch.tensor([700, 701], dtype=torch.int32, device=device)

    block_states = _initialize_state_storage(1, device, 42001)
    control_states = _initialize_state_storage(1, device, 42001)
    _, _, token_ptrs = _block_token_rows(1, device)
    selected = torch.full((1, position_count),
                          -777,
                          dtype=torch.int32,
                          device=device)
    accept_len = torch.full((1, ), -1, dtype=torch.int32, device=device)
    accepted_count = torch.full((1, ), -1, dtype=torch.int32, device=device)
    offsets = torch.tensor([0, 1], dtype=torch.int32, device=device)
    state_indices = torch.zeros(1, dtype=torch.int32, device=device)

    verify_target_block(
        probabilities,
        token_ids,
        torch.tensor([2] * position_count, dtype=torch.int32, device=device),
        drafts,
        torch.zeros(position_count, dtype=torch.bool, device=device),
        torch.ones(position_count, dtype=torch.bool, device=device),
        block_states,
        state_indices,
        token_ptrs,
        torch.zeros(1, dtype=torch.int32, device=device),
        offsets,
        torch.ones(1, dtype=torch.bool, device=device),
        selected,
        accept_len,
        accepted_count,
        position_count,
    )
    block_accept_len = accept_len.clone()

    ordinary_probabilities = torch.tensor([[1.0]],
                                          dtype=torch.float32,
                                          device=device)
    ordinary_ids = torch.tensor([[900]], dtype=torch.int32, device=device)
    empty_drafts = torch.empty((0, 1), dtype=torch.int32, device=device)
    for _ in range(rejection_position + 2):
        verify_target_block(
            ordinary_probabilities,
            ordinary_ids,
            torch.ones(1, dtype=torch.int32, device=device),
            empty_drafts,
            torch.zeros(1, dtype=torch.bool, device=device),
            torch.ones(1, dtype=torch.bool, device=device),
            control_states,
            state_indices,
            token_ptrs,
            torch.zeros(1, dtype=torch.int32, device=device),
            offsets,
            torch.zeros(1, dtype=torch.bool, device=device),
            selected[:, :1],
            accept_len,
            None,
            1,
        )
    torch.cuda.current_stream(device).synchronize()

    assert block_accept_len.item() == rejection_position + 1
    assert accepted_count.item() == rejection_position
    assert torch.equal(block_states, control_states)


def test_verify_target_block_full_acceptance_rng_and_null_metrics() -> None:
    device = torch.device('cuda')
    position_count = 4
    probabilities = torch.zeros((position_count, 2),
                                dtype=torch.float32,
                                device=device)
    probabilities[:, 0] = 1.0
    token_ids = torch.tensor([[801, 901], [802, 902], [803, 903], [804, 904]],
                             dtype=torch.int32,
                             device=device)
    drafts = torch.tensor([[801], [802], [803]],
                          dtype=torch.int32,
                          device=device)
    block_states = _initialize_state_storage(1, device, 43001)
    control_states = _initialize_state_storage(1, device, 43001)
    allocations, token_rows, token_ptrs = _block_token_rows(1, device)
    selected = torch.full((1, position_count),
                          -777,
                          dtype=torch.int32,
                          device=device)
    accept_len = torch.zeros(1, dtype=torch.int32, device=device)
    offsets = torch.tensor([0, 1], dtype=torch.int32, device=device)
    state_indices = torch.zeros(1, dtype=torch.int32, device=device)

    common = dict(
        probabilities=probabilities,
        probability_token_ids=token_ids,
        kept_count=torch.ones(position_count, dtype=torch.int32,
                              device=device),
        verification_draft_ids=drafts,
        greedy=torch.zeros(position_count, dtype=torch.bool, device=device),
        logits_active=torch.ones(position_count,
                                 dtype=torch.bool,
                                 device=device),
        random_states=block_states,
        random_state_indices=state_indices,
        request_token_ids_ptrs=token_ptrs,
        entry_sequence_length=torch.tensor([2],
                                           dtype=torch.int32,
                                           device=device),
        request_to_generation_offsets=offsets,
        speculative_row=torch.ones(1, dtype=torch.bool, device=device),
        selected_span_ids=selected,
        accept_len=accept_len,
        accepted_draft_count=None,
        position_count=position_count,
    )
    verify_target_block(**common)

    ordinary_probabilities = torch.tensor([[1.0]],
                                          dtype=torch.float32,
                                          device=device)
    ordinary_ids = torch.tensor([[999]], dtype=torch.int32, device=device)
    empty_drafts = torch.empty((0, 1), dtype=torch.int32, device=device)
    for _ in range(position_count):
        verify_target_block(ordinary_probabilities, ordinary_ids,
                            torch.ones(1, dtype=torch.int32,
                                       device=device), empty_drafts,
                            torch.zeros(1, dtype=torch.bool, device=device),
                            torch.ones(1, dtype=torch.bool, device=device),
                            control_states, state_indices, token_ptrs,
                            torch.zeros(1, dtype=torch.int32,
                                        device=device), offsets,
                            torch.zeros(1, dtype=torch.bool, device=device),
                            selected[:, :1], accept_len, None, 1)
    torch.cuda.current_stream(device).synchronize()

    assert torch.equal(block_states, control_states)
    assert token_rows[0][2:6].tolist() == [801, 802, 803, 804]
    _assert_block_row_guards(allocations)
