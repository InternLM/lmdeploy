# Copyright (c) OpenMMLab. All rights reserved.
import _turbomind as _tm
import torch

from ._base import Builder


def make_norm_config(*, dim, norm_eps):
    cfg = _tm.NormConfig()
    cfg.dim = dim
    cfg.norm_eps = norm_eps
    return cfg


class NormBuilder(Builder):
    """Builder for a single norm weight module."""

    def set_weight(self, tensor: torch.Tensor):
        """Commit the norm weight tensor to all GPU handles."""
        self._add_tensor('weight', tensor)
