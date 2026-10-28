from __future__ import annotations

import torch

_NATIVE_SYMBOL = 'select_draft_carry'


def _load_native_bridge():
    try:
        from lmdeploy.turbomind import _turbomind as tm
    except ImportError:
        return None
    return tm if hasattr(tm, _NATIVE_SYMBOL) else None


def _require_native_bridge():
    tm = _load_native_bridge()
    if tm is None:
        raise ImportError(
            'TurboMind draft-carry bridge is unavailable; '
            f'required symbol: {_NATIVE_SYMBOL}')
    return tm


def is_available() -> bool:
    return _load_native_bridge() is not None


def select_draft_carry(
    local_residual: torch.Tensor,
    selected_local_rows: torch.Tensor,
    candidate_active: torch.Tensor,
    carry: torch.Tensor,
    first: int,
    last: int,
) -> None:
    stream_ptr = int(
        torch.cuda.current_stream(
            local_residual.device).cuda_stream)
    _require_native_bridge().select_draft_carry(
        local_residual,
        selected_local_rows,
        candidate_active,
        carry,
        int(first),
        int(last),
        stream_ptr,
    )
