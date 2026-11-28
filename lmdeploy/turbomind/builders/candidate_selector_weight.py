# Copyright (c) OpenMMLab. All rights reserved.
"""Builder for the DFlash2 candidate-selector weight module."""

import _turbomind as _tm

from ._base import _CPP_TO_TORCH, Builder

CandidateSelectorWeightConfig = _tm.CandidateSelectorWeightConfig


class CandidateSelectorWeightBuilder(Builder):
    """Low-rank hidden projection + predecessor/successor codebooks."""

    def add_hidden_projection(self, linear):
        self._add_linear('hidden_projection', linear, split_side=None)

    def _add_book(self, name, tensor):
        # The selector kernel reads the codebooks at the model compute dtype;
        # the checkpoint ships them bf16, so cast to avoid bit reinterpretation.
        dtype = _CPP_TO_TORCH.get(self.config.data_type)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        self._add_tensor(name, tensor, split_side=None)

    def add_predecessor_codebook(self, tensor):
        self._add_book('predecessor_codebook', tensor)

    def add_successor_codebook(self, tensor):
        self._add_book('successor_codebook', tensor)
