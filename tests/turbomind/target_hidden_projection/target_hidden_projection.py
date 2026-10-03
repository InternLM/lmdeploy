from __future__ import annotations

import torch

_NATIVE_SYMBOL = 'capture_target_hidden_rows'


def _load_native_bridge():
    try:
        import _turbomind as tm
    except ImportError:
        return None
    return tm if hasattr(tm, _NATIVE_SYMBOL) else None


def _require_native_bridge():
    tm = _load_native_bridge()
    if tm is None:
        raise ImportError(
            'TurboMind target-hidden capture bridge is unavailable; '
            f'required symbol: {_NATIVE_SYMBOL}')
    return tm


def is_available() -> bool:
    return _load_native_bridge() is not None


def capture_target_hidden_rows(
    packed_residual: torch.Tensor,
    captured: torch.Tensor,
    owned_begin: int,
    owned_row_count: int,
    tap_ordinal: int,
) -> None:
    stream_ptr = int(
        torch.cuda.current_stream(
            packed_residual.device).cuda_stream)
    _require_native_bridge().capture_target_hidden_rows(
        packed_residual,
        captured,
        int(owned_begin),
        int(owned_row_count),
        int(tap_ordinal),
        stream_ptr,
    )
