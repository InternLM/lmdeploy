from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OwnedTokenRows:
    offset: int
    first: int
    last: int


def compute_token_ownership(
    global_rank: int,
    tp0: int,
    tp1: int,
    local_token_nums: Sequence[int],
) -> OwnedTokenRows:
    """Compute one rank's packed offset and DP-local model-TP interval."""
    inner_tp = min(tp0, tp1)
    dp_index = global_rank // inner_tp
    tp_index = global_rank % inner_tp
    num = int(local_token_nums[dp_index])

    slice_size = (num + inner_tp - 1) // inner_tp
    first = min(num, tp_index * slice_size)
    last = min(num, first + slice_size)
    offset = sum(int(value) for value in local_token_nums[:dp_index])
    return OwnedTokenRows(offset=offset, first=first, last=last)


def select_draft_carry_reference(
    local_residual_bytes: torch.Tensor,
    selected_local_rows: Sequence[int],
    candidate_active: Sequence[bool],
    first: int,
    last: int,
) -> torch.Tensor:
    """Apply owner-or-zero selection to an opaque two-dimensional byte view."""
    candidate_count = len(selected_local_rows)
    row_bytes = local_residual_bytes.shape[1]
    carry = torch.zeros(
        (candidate_count, row_bytes),
        dtype=torch.uint8,
        device=local_residual_bytes.device,
    )

    local_token_num = local_residual_bytes.shape[0]
    for candidate, (row, active) in enumerate(
        zip(selected_local_rows, candidate_active)
    ):
        if (
            active
            and 0 <= row < local_token_num
            and first <= row < last
        ):
            carry[candidate].copy_(local_residual_bytes[row])
    return carry
