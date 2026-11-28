# Copyright (c) OpenMMLab. All rights reserved.
"""Builder for the DFlash2 method weight tree (ModelWeight.spec)."""

import _turbomind as _tm

from ._base import Builder

Dflash2WeightConfig = _tm.Dflash2WeightConfig


class Dflash2WeightBuilder(Builder):
    """fc + hidden_norm/norm + per-layer convs + candidate selector.

    ``hidden_norm``, ``norm``, ``attention_convs``, ``mlp_convs`` and
    ``selector`` are attached by assigning their ``build()`` output to the
    matching attribute. ``fc`` is committed here.
    """

    def add_fc(self, linear):
        # 5 * hidden -> hidden context projection; replicate for now.
        self._add_linear('fc', linear, split_side=None)
