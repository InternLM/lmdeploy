# Copyright (c) OpenMMLab. All rights reserved.
"""Builder for affine LayerNorm weights (gamma + beta)."""
import _turbomind as _tm
import torch

from ._base import Builder


def make_layer_norm_config(*, dim, data_type, norm_eps):
    cfg = _tm.LayerNormConfig()
    cfg.dim = dim
    cfg.data_type = data_type
    cfg.norm_eps = norm_eps
    return cfg


class LayerNormBuilder(Builder):
    """Builder for a single LayerNorm weight module (weight + bias)."""

    def set_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        """Commit the LayerNorm gamma (and optional beta) tensors."""
        self._add_tensor('weight', weight)
        if bias is not None:
            self._add_tensor('bias', bias)
