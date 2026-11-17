# Copyright (c) OpenMMLab. All rights reserved.
"""Builder for the DFlash2 candidate-selector weight module."""

import _turbomind as _tm

from ._base import Builder

CandidateSelectorWeightConfig = _tm.CandidateSelectorWeightConfig


class CandidateSelectorWeightBuilder(Builder):
    """Low-rank hidden projection + predecessor/successor codebooks."""

    def add_hidden_projection(self, linear):
        self._add_linear('hidden_projection', linear, split_side=None)

    def add_predecessor_codebook(self, tensor):
        self._add_tensor('predecessor_codebook', tensor, split_side=None)

    def add_successor_codebook(self, tensor):
        self._add_tensor('successor_codebook', tensor, split_side=None)
