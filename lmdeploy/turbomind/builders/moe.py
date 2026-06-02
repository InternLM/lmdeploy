# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from ._base import Builder, ParallelGroup, SplitSide
from .module_list import ModuleListBuilder, ModuleListConfig

# ---------------------------------------------------------------------------
# MoeBuilder -- gate, non-expert params
# ---------------------------------------------------------------------------


class MoeBuilder(Builder):
    """MoE weight loading builder."""

    def __init__(self, config, ctx, ep: ParallelGroup | None = None):
        super().__init__(config, ctx)
        self.ep = ep or ParallelGroup(1, None)
        if self.ep.size > 1 and config.expert_num % self.ep.size != 0:
            raise ValueError(
                f'num_experts={config.expert_num} must be divisible by '
                f'ep={self.ep.size}')
        self.config.ep_size = self.ep.size

    def add_gate(self, name, linear):
        """Commit a gate linear (broadcast, no split)."""
        self._add_linear(name, linear, split_side=None)

    def add_param(self, name, tensor, split_side=None):
        """Commit a non-expert MoE parameter."""
        if split_side is not None and not isinstance(split_side, SplitSide):
            split_side = None  # specs may pass None for broadcast
        self._add_tensor(name, tensor, split_side)

    def add_experts(self, build_expert, name='experts'):
        """Build and attach expert modules with EP ownership applied."""
        experts = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for expert_idx in range(self.config.expert_num):
            active_mask = self._expert_active_mask(expert_idx)
            if active_mask is None:
                experts[expert_idx] = build_expert(expert_idx)
                continue
            with self._ctx.active_mask_scope(active_mask):
                experts[expert_idx] = build_expert(expert_idx)
        setattr(self, name, experts.build())

    def _expert_active_mask(self, expert_idx: int):
        ep_size = self.ep.size
        if ep_size <= 1:
            return None
        ranks = self.ep.ranks
        assert ranks is not None
        local = self.config.expert_num // ep_size
        return [rank * local <= expert_idx < (rank + 1) * local
                for rank in ranks]
