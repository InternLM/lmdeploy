# Copyright (c) OpenMMLab. All rights reserved.
"""FFN weight loading builder and w1+w3 fusion helpers.

Provides ``FfnBuilder`` for committing FFN weights (w1/w2/w3 with optional
w1+w3 fusion) and helper functions for determining whether SiLU fusion
(interleave vs chunk) should be used and whether w1+w3 fusion is safe for
the given TP configuration.
"""
from __future__ import annotations

import math

import torch

from .. import _tm
from ..linear import Linear, pad_input_groups, pad_output_groups, transform_output_dim
from ._base import Builder, ParallelGroup, SplitSide

__all__ = [
    'FfnBuilder',
]

# ---------------------------------------------------------------------------
# @transform_output_dim / @transform_input_dim helpers
# ---------------------------------------------------------------------------


@transform_output_dim
def _block_pack_w1w3(w1: torch.Tensor, w3: torch.Tensor, *,
                     groups: int) -> torch.Tensor:
    """Interleave gate/up by logical output groups.

    The elements per group self-adapt for each Linear tensor kind. For FP8, one logical 128-wide weight group
    corresponds to one scale element.
    """
    assert groups > 0, f'groups must be positive, got {groups}'
    assert w1.shape[-1] % groups == 0, (
        f'output dim {w1.shape[-1]} not divisible by groups {groups}')
    assert w3.shape[-1] == w1.shape[-1], (
        f'w1/w3 output dims differ: {w1.shape[-1]} != {w3.shape[-1]}')
    w1g = w1.unflatten(-1, (groups, -1))
    w3g = w3.unflatten(-1, (groups, -1))
    return torch.stack([w1g, w3g], dim=-2).flatten(-3, -1).contiguous()

# ---------------------------------------------------------------------------
# TP padding
# ---------------------------------------------------------------------------


def _pad_ffn_for_tp(w1: Linear, w2: Linear, w3: Linear, tp: int, plan) -> tuple[Linear, Linear, Linear]:
    """Pad the intermediate axis to the plan's TP-local shape contract."""
    ((min_k, min_n), (align_k, align_n)) = plan.shape_constraints

    fmt = w1.weight_format
    block_in = fmt.block_in or 1
    block_out = fmt.block_out or 1
    unit = math.lcm(block_in, block_out)

    projection_min_n = (min_n + 1) // 2
    projection_align_n = math.lcm(block_out, align_n // math.gcd(align_n, 2))
    minimum = max(min_k, projection_min_n)
    granularity = math.lcm(align_k, projection_align_n)

    raw = int(w1.tensors['weight'].size(-1))
    divisor = granularity * tp
    target = max(raw, minimum * tp)
    target = (target + divisor - 1) // divisor * divisor
    groups = raw // unit
    target_groups = target // unit
    w1 = pad_output_groups(w1, src_groups=groups,
                           dst_groups=target_groups)
    w3 = pad_output_groups(w3, src_groups=groups,
                           dst_groups=target_groups)
    w2 = pad_input_groups(w2, src_groups=groups,
                          dst_groups=target_groups)
    return w1, w2, w3


# ---------------------------------------------------------------------------
# FfnBuilder -- w1+w3 fusion, w2 commit
# ---------------------------------------------------------------------------


class FfnBuilder(Builder):
    """FFN weight loading builder with w1+w3 fusion."""

    def __init__(self, config, ctx, tp: ParallelGroup):
        super().__init__(config, ctx)
        self.tp = tp
        self.config.tp_size = tp.size

    def add_ffn(self, w1, w2, w3):
        """Plan, pad, combine gate/up, then commit gate/up and down."""
        act_type = getattr(self.config, 'act_type', 0)
        if isinstance(act_type, int):
            act_type = {0: 'silu', 1: 'gpt-oss'}.get(act_type, 'silu')
        act_type = (_tm.ActivationType.kSiluGptOss
                    if act_type == 'gpt-oss'
                    else _tm.ActivationType.kSilu)

        plan = self._query_gemm(self._make_gemm_query(
            w1, grouped=self.config.is_expert))
        w1, w2, w3 = _pad_ffn_for_tp(w1, w2, w3, self.tp.size, plan)
        proj = w1.tensors['weight'].size(-1)
        gate_up_block = plan.gate_up(act_type, proj // self.tp.size)

        self.config.inter_size = proj
        self.config.fuse_silu = gate_up_block != 0
        groups = (proj // gate_up_block
                  if gate_up_block else self.tp.size)
        w1w3 = _block_pack_w1w3(w1, w3, groups=groups)
        self._add_linear('w1w3', w1w3, SplitSide.OUTPUT, plan)

        plan = self._query_gemm(self._make_gemm_query(
            w2, grouped=self.config.is_expert))
        self._add_linear('w2', w2, SplitSide.INPUT, plan)
