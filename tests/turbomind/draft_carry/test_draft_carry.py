from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from .draft_carry import (
    is_available,
    select_draft_carry,
)
from .reference import (
    OwnedTokenRows,
    compute_token_ownership,
    select_draft_carry_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA is required for draft-carry tests',
)

_HIDDEN_SIZE = 4096
_GUARD_BYTES = 64
_GUARD_VALUE = 0xD7
_OUTPUT_POISON = 0xA5
_INVALID_ROW = 1_000_000_000

_WIDTH_DTYPES = {
    8: (torch.uint8, torch.int8),
    16: (torch.int16, torch.float16),
    32: (torch.int32, torch.float32),
}

_TOPOLOGIES = [
    pytest.param(1, (4, 3, 0), id='model_tp_1'),
    pytest.param(2, (4, 3, 1), id='model_tp_2'),
    pytest.param(8, (16, 5, 0), id='model_tp_8'),
]


def _require_bridge() -> None:
    if not is_available():
        pytest.skip('TurboMind draft-carry bridge is unavailable')


def _make_packed_diagnostic(
    local_token_nums: Sequence[int],
    row_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_rows = sum(local_token_nums)
    payload_bytes = total_rows * row_bytes
    storage = torch.full(
        (2 * _GUARD_BYTES + payload_bytes,),
        _GUARD_VALUE,
        dtype=torch.uint8,
        device='cuda',
    )
    payload = storage[
        _GUARD_BYTES:_GUARD_BYTES + payload_bytes
    ].reshape(total_rows, row_bytes)
    if payload_bytes:
        values = (
            torch.arange(
                payload_bytes,
                dtype=torch.int64,
                device='cuda',
            )
            .mul_(37)
            .add_(17)
            .remainder_(251)
            .add_(1)
            .to(torch.uint8)
        )
        payload.copy_(values.reshape_as(payload))
    return storage, payload


def _guarded_carry(
    candidate_count: int,
    row_bytes: int,
    carry_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload_bytes = candidate_count * row_bytes
    storage = torch.full(
        (2 * _GUARD_BYTES + payload_bytes,),
        _OUTPUT_POISON,
        dtype=torch.uint8,
        device='cuda',
    )
    payload = storage[
        _GUARD_BYTES:_GUARD_BYTES + payload_bytes
    ].reshape(candidate_count, row_bytes)
    return storage, payload.view(carry_dtype), payload


def _assert_guards(
    storage: torch.Tensor,
    guard_value: int,
) -> None:
    assert torch.all(storage[:_GUARD_BYTES] == guard_value).item()
    assert torch.all(storage[-_GUARD_BYTES:] == guard_value).item()


def _segment_ownership(
    dp_index: int,
    model_tp: int,
    tp0: int,
    tp1: int,
    local_token_nums: Sequence[int],
) -> list[OwnedTokenRows]:
    ownership = [
        compute_token_ownership(
            dp_index * model_tp + tp_index,
            tp0,
            tp1,
            local_token_nums,
        )
        for tp_index in range(model_tp)
    ]

    num = local_token_nums[dp_index]
    offset = sum(local_token_nums[:dp_index])
    cursor = 0
    for owned in ownership:
        assert owned.offset == offset
        assert owned.first == cursor
        assert owned.first <= owned.last <= num
        cursor = owned.last
    assert cursor == num
    return ownership


def _candidate_rows(
    ownership: Sequence[OwnedTokenRows],
    local_token_num: int,
) -> tuple[list[int], list[bool]]:
    boundary_rows: list[int] = []
    for owned in ownership:
        if owned.first < owned.last:
            boundary_rows.extend((owned.first, owned.last - 1))
            if owned.first > 0:
                boundary_rows.append(owned.first - 1)
            if owned.last < local_token_num:
                boundary_rows.append(owned.last)

    selected_rows = list(dict.fromkeys(boundary_rows))
    active = [True] * len(selected_rows)

    inactive_valid = selected_rows[0] if selected_rows else 0
    selected_rows.extend((inactive_valid, _INVALID_ROW, -_INVALID_ROW))
    active.extend((False, False, True))
    return selected_rows, active


@pytest.mark.parametrize('element_bits', [8, 16, 32])
@pytest.mark.parametrize('global_communicator_first', [True, False])
@pytest.mark.parametrize('model_tp,local_token_nums', _TOPOLOGIES)
def test_model_tp_owner_or_zero_matrix(
    model_tp,
    local_token_nums,
    global_communicator_first,
    element_bits,
):
    _require_bridge()
    global_rank_count = model_tp * len(local_token_nums)
    if global_communicator_first:
        tp0, tp1 = global_rank_count, model_tp
    else:
        tp0, tp1 = model_tp, global_rank_count

    source_dtype, carry_dtype = _WIDTH_DTYPES[element_bits]
    element_bytes = element_bits // 8
    row_bytes = _HIDDEN_SIZE * element_bytes
    diagnostic_storage, diagnostic_bytes = _make_packed_diagnostic(
        local_token_nums,
        row_bytes,
    )
    diagnostic_before = diagnostic_storage.clone()

    for dp_index, local_token_num in enumerate(local_token_nums):
        ownership = _segment_ownership(
            dp_index,
            model_tp,
            tp0,
            tp1,
            local_token_nums,
        )
        selected_rows, active = _candidate_rows(
            ownership,
            local_token_num,
        )
        selected_local_rows = torch.tensor(
            selected_rows,
            dtype=torch.int32,
            device='cuda',
        )
        candidate_active = torch.tensor(
            active,
            dtype=torch.bool,
            device='cuda',
        )

        offset = ownership[0].offset
        local_bytes = diagnostic_bytes[
            offset:offset + local_token_num
        ]
        local_residual = local_bytes.view(source_dtype)
        assert local_residual.shape == (
            local_token_num,
            _HIDDEN_SIZE,
        )

        rank_results = []
        for owned in ownership:
            carry_storage, carry, carry_bytes = _guarded_carry(
                len(selected_rows),
                row_bytes,
                carry_dtype,
            )
            expected = select_draft_carry_reference(
                local_bytes,
                selected_rows,
                active,
                owned.first,
                owned.last,
            )

            result = select_draft_carry(
                local_residual,
                selected_local_rows,
                candidate_active,
                carry,
                owned.first,
                owned.last,
            )

            assert result is None
            assert torch.equal(carry_bytes, expected)
            _assert_guards(carry_storage, _OUTPUT_POISON)
            rank_results.append(carry_bytes.clone())

        owner_sum = torch.stack(
            rank_results,
            dim=0,
        ).to(torch.int64).sum(dim=0)
        unsharded = select_draft_carry_reference(
            local_bytes,
            selected_rows,
            active,
            0,
            local_token_num,
        )
        assert torch.equal(
            owner_sum,
            unsharded.to(torch.int64),
        )

    assert torch.equal(diagnostic_storage, diagnostic_before)
    _assert_guards(diagnostic_storage, _GUARD_VALUE)


@pytest.mark.parametrize('element_bits', [8, 16, 32])
def test_identity_selection_supports_exact_payload_alias(element_bits):
    _require_bridge()
    source_dtype, carry_dtype = _WIDTH_DTYPES[element_bits]
    element_bytes = element_bits // 8
    row_bytes = _HIDDEN_SIZE * element_bytes
    candidate_count = 6
    payload_bytes = candidate_count * row_bytes

    storage = torch.full(
        (2 * _GUARD_BYTES + payload_bytes,),
        _GUARD_VALUE,
        dtype=torch.uint8,
        device='cuda',
    )
    payload = storage[
        _GUARD_BYTES:_GUARD_BYTES + payload_bytes
    ].reshape(candidate_count, row_bytes)
    values = (
        torch.arange(
            payload_bytes,
            dtype=torch.int64,
            device='cuda',
        )
        .mul_(29)
        .add_(11)
        .remainder_(251)
        .add_(1)
        .to(torch.uint8)
    )
    payload.copy_(values.reshape_as(payload))

    selected_rows = list(range(candidate_count))
    active = [True, True, False, True, True, True]
    expected = select_draft_carry_reference(
        payload.clone(),
        selected_rows,
        active,
        first=1,
        last=5,
    )
    selected_local_rows = torch.arange(
        candidate_count,
        dtype=torch.int32,
        device='cuda',
    )
    candidate_active = torch.tensor(
        active,
        dtype=torch.bool,
        device='cuda',
    )

    result = select_draft_carry(
        payload.view(source_dtype),
        selected_local_rows,
        candidate_active,
        payload.view(carry_dtype),
        first=1,
        last=5,
    )

    assert result is None
    assert torch.equal(payload, expected)
    _assert_guards(storage, _GUARD_VALUE)
