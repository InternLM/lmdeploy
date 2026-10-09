from __future__ import annotations

import torch


def capture_target_hidden_rows(
    packed_residual: torch.Tensor,
    captured: torch.Tensor,
    owned_begin: int,
    owned_row_count: int,
    tap_ordinal: int,
) -> None:
    """Apply the target-row capture contract using direct Torch assignment."""
    hidden_units = packed_residual.shape[1]
    tap_begin = tap_ordinal * hidden_units
    captured[:owned_row_count, tap_begin:tap_begin + hidden_units] = (
        packed_residual[owned_begin:owned_begin + owned_row_count, :])
