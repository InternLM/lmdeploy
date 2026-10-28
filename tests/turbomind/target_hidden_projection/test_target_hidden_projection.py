from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from .reference import capture_target_hidden_rows as reference_capture
from .target_hidden_projection import (
    capture_target_hidden_rows as native_capture,
)
from .target_hidden_projection import is_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA is required for target-hidden capture tests',
)

_CAPTURE_POISON_BITS = 0x3555
_SOURCE_POISON_BITS = 0x5A5A


@dataclass(frozen=True)
class _PitchedTensor:
    storage: torch.Tensor
    view: torch.Tensor


def _require_bridge() -> None:
    if not is_available():
        pytest.skip('TurboMind target-hidden capture bridge is unavailable')


def _pitched_tensor(
    rows: int,
    width: int,
    *,
    row_padding: int,
    prefix: int,
    suffix: int,
    dtype: torch.dtype,
    poison_bits: int,
) -> _PitchedTensor:
    leading_dimension = width + row_padding
    storage = torch.empty(
        prefix + rows * leading_dimension + suffix,
        dtype=dtype,
        device='cuda',
    )
    storage.view(torch.int16).fill_(poison_bits)
    view = torch.as_strided(
        storage,
        (rows, width),
        (leading_dimension, 1),
        storage_offset=prefix,
    )
    return _PitchedTensor(storage, view)


def _row_bits(hidden_units: int, seed: int) -> torch.Tensor:
    values = (
        torch.arange(
            hidden_units,
            dtype=torch.int32,
            device='cuda',
        ) * 251 + seed
    )
    return values.remainder(65536).sub(32768).to(torch.int16)


def _make_source(
    rows: int,
    hidden_units: int,
    owned_begin: int,
    owned_row_count: int,
    dtype: torch.dtype,
    *,
    seed: int,
) -> _PitchedTensor:
    source = _pitched_tensor(
        rows,
        hidden_units,
        row_padding=7,
        prefix=5,
        suffix=11,
        dtype=dtype,
        poison_bits=_SOURCE_POISON_BITS,
    )
    for row in range(owned_begin, owned_begin + owned_row_count):
        source.view[row].view(torch.int16).copy_(
            _row_bits(hidden_units, seed + row * 977))
    return source


def _make_capture(
    capacity: int,
    hidden_units: int,
    tap_count: int,
    dtype: torch.dtype,
) -> _PitchedTensor:
    return _pitched_tensor(
        capacity,
        tap_count * hidden_units,
        row_padding=13,
        prefix=9,
        suffix=17,
        dtype=dtype,
        poison_bits=_CAPTURE_POISON_BITS,
    )


def _assert_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.view(torch.int16),
        expected.view(torch.int16),
    )


def _run_capture_case(
    *,
    dtype: torch.dtype,
    source_rows: int,
    owned_begin: int,
    owned_row_count: int,
    capacity: int,
    hidden_units: int,
    tap_count: int,
    tap_ordinal: int,
    seed: int = 101,
) -> None:
    _require_bridge()
    source = _make_source(
        source_rows,
        hidden_units,
        owned_begin,
        owned_row_count,
        dtype,
        seed=seed,
    )
    source_before = source.storage.clone()
    captured = _make_capture(
        capacity,
        hidden_units,
        tap_count,
        dtype,
    )
    expected_storage = captured.storage.clone()
    expected = torch.as_strided(
        expected_storage,
        captured.view.shape,
        captured.view.stride(),
        storage_offset=captured.view.storage_offset(),
    )

    reference_capture(
        source.view,
        expected,
        owned_begin,
        owned_row_count,
        tap_ordinal,
    )
    native_capture(
        source.view,
        captured.view,
        owned_begin,
        owned_row_count,
        tap_ordinal,
    )

    _assert_bits_equal(captured.storage, expected_storage)
    _assert_bits_equal(source.storage, source_before)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    'source_rows,owned_begin,owned_row_count,capacity,hidden_units,tap_ordinal',
    [
        (0, 0, 0, 4, 13, 2),
        (6, 3, 1, 3, 17, 0),
        (11, 2, 5, 7, 31, 3),
        (7, 0, 7, 7, 16, 4),
    ],
    ids=['zero', 'one', 'uneven', 'full'],
)
def test_capture_interval_matrix_is_bit_exact(
    dtype,
    source_rows,
    owned_begin,
    owned_row_count,
    capacity,
    hidden_units,
    tap_ordinal,
):
    _run_capture_case(
        dtype=dtype,
        source_rows=source_rows,
        owned_begin=owned_begin,
        owned_row_count=owned_row_count,
        capacity=capacity,
        hidden_units=hidden_units,
        tap_count=5,
        tap_ordinal=tap_ordinal,
    )


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('tap_ordinal', range(5))
def test_each_tap_isolated(dtype, tap_ordinal):
    _run_capture_case(
        dtype=dtype,
        source_rows=9,
        owned_begin=3,
        owned_row_count=4,
        capacity=6,
        hidden_units=23,
        tap_count=5,
        tap_ordinal=tap_ordinal,
        seed=1000 + tap_ordinal,
    )


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_all_five_taps_populated(dtype):
    _require_bridge()
    hidden_units = 29
    owned_begin = 2
    owned_row_count = 5
    captured = _make_capture(7, hidden_units, 5, dtype)
    expected_storage = captured.storage.clone()
    expected = torch.as_strided(
        expected_storage,
        captured.view.shape,
        captured.view.stride(),
        storage_offset=captured.view.storage_offset(),
    )

    sources = []
    source_snapshots = []
    for tap_ordinal in range(5):
        source = _make_source(
            10,
            hidden_units,
            owned_begin,
            owned_row_count,
            dtype,
            seed=2000 + 100 * tap_ordinal,
        )
        sources.append(source)
        source_snapshots.append(source.storage.clone())
        reference_capture(
            source.view,
            expected,
            owned_begin,
            owned_row_count,
            tap_ordinal,
        )
        native_capture(
            source.view,
            captured.view,
            owned_begin,
            owned_row_count,
            tap_ordinal,
        )

    _assert_bits_equal(captured.storage, expected_storage)
    for source, snapshot in zip(sources, source_snapshots):
        _assert_bits_equal(source.storage, snapshot)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_single_tap_diagnostic(dtype):
    _run_capture_case(
        dtype=dtype,
        source_rows=8,
        owned_begin=1,
        owned_row_count=6,
        capacity=6,
        hidden_units=37,
        tap_count=1,
        tap_ordinal=0,
    )


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_hidden_size_4096(dtype):
    _run_capture_case(
        dtype=dtype,
        source_rows=6,
        owned_begin=2,
        owned_row_count=3,
        capacity=4,
        hidden_units=4096,
        tap_count=5,
        tap_ordinal=4,
    )


def test_nondefault_stream_and_asynchronous_return():
    _require_bridge()
    source = _make_source(
        8,
        41,
        2,
        4,
        torch.float16,
        seed=31415,
    )
    captured = _make_capture(6, 41, 5, torch.float16)
    expected_storage = captured.storage.clone()
    expected = torch.as_strided(
        expected_storage,
        captured.view.shape,
        captured.view.stride(),
        storage_offset=captured.view.storage_offset(),
    )
    reference_capture(source.view, expected, 2, 4, 3)

    torch.cuda.synchronize()
    stream = torch.cuda.Stream(device=source.view.device)
    with torch.cuda.stream(stream):
        if hasattr(torch.cuda, '_sleep'):
            torch.cuda._sleep(1_000_000_000)
        native_capture(source.view, captured.view, 2, 4, 3)
        if hasattr(torch.cuda, '_sleep'):
            assert not stream.query()
    stream.synchronize()

    _assert_bits_equal(captured.storage, expected_storage)
