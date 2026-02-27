# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache

import torch

from ..causal_conv1d import CausalConv1dBuilder, CausalConv1dImpl


class CausalConv1dTilelangImpl(CausalConv1dImpl):
    """CausalConv1d update implementation."""

    def __init__(self):
        from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update

    def conv1d_fn(self,
                  x: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor | None = None,
                  seq_idx: torch.Tensor | None = None,
                  return_final_states: bool = False,
                  activation: str | None = None):
        return self.causal_conv1d_fn(x,
                                     weight,
                                     bias=bias,
                                     seq_idx=seq_idx,
                                     return_final_states=return_final_states,
                                     activation=activation)

    def update_fn(self,
                  x: torch.Tensor,
                  conv_state: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor | None = None,
                  activation: str | None = None,
                  conv_state_indices: torch.Tensor | None = None):
        """Update conv state."""
        return self.causal_conv1d_update(x,
                                         conv_state,
                                         weight,
                                         bias=bias,
                                         activation=activation,
                                         conv_state_indices=conv_state_indices)


class CausalConv1dDaoImpl(CausalConv1dTilelangImpl):

    def __init__(self):
        try:
            import causal_conv1d
            self.causal_conv1d_fn = causal_conv1d.causal_conv1d_fn
            self.causal_conv1d_update = causal_conv1d.causal_conv1d_update
        except Exception:
            raise RuntimeError(
                'causal_conv1d is not installed, please refer to https://github.com/Dao-AILab/causal-conv1d')


@lru_cache
def has_tilelang():
    try:
        import tilelang  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache
def has_dao():
    try:
        import causal_conv1d  # noqa: F401
        causal_conv1d_fn = causal_conv1d.causal_conv1d_fn  # noqa: F841
        causal_conv1d_update = causal_conv1d.causal_conv1d_update  # noqa: F841
        return True
    except Exception:
        return False


class CausalConv1dCudaBuilder(CausalConv1dBuilder):
    """CausalConv1d update implementation builder."""

    @staticmethod
    def build() -> CausalConv1dImpl:
        """build."""
        if has_tilelang():
            return CausalConv1dTilelangImpl()
        elif has_dao():
            return CausalConv1dDaoImpl()
        else:
            raise RuntimeError('No available implementation for CausalConv1d, '
                               'please install https://tilelang.com/ or https://github.com/Dao-AILab/causal-conv1d')
