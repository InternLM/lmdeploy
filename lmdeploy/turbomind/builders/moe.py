# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from ._base import Builder, SplitSide

# ---------------------------------------------------------------------------
# MoeBuilder -- gate, non-expert params
# ---------------------------------------------------------------------------


class MoeBuilder(Builder):
    """MoE weight loading builder."""

    def add_gate(self, name, linear):
        """Commit a gate linear (broadcast, no split)."""
        self._add_linear(name, linear, split_side=None)

    def add_param(self, name, tensor, split_side=None):
        """Commit a non-expert MoE parameter."""
        if split_side is not None and not isinstance(split_side, SplitSide):
            split_side = None  # specs may pass None for broadcast
        self._add_tensor(name, tensor, split_side)
