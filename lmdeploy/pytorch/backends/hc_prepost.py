# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .base import BuildSpec


class HCPrePostImpl(ABC):
    """Backend interface for DeepSeek-V4 hyper-connection reductions."""

    @abstractmethod
    def pre(
        self,
        x: torch.Tensor,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run sinkhorn and reduce HC states from ``[..., hc, dim]`` to ``[...,
        dim]``."""
        raise NotImplementedError

    @abstractmethod
    def pre_reduce(self, x: torch.Tensor, pre: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """Reduce HC hidden states from ``[..., hc, dim]`` to ``[...,
        dim]``."""
        raise NotImplementedError

    @abstractmethod
    def post_expand(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                    comb: torch.Tensor) -> torch.Tensor:
        """Expand one hidden state back to ``[..., hc, dim]``."""
        raise NotImplementedError


@dataclass(frozen=True)
class HCPrePostBuildSpec(BuildSpec[HCPrePostImpl]):
    """Immutable requirements for DeepSeek-V4 hyper-connection kernels."""

    hc_mult: int
    sinkhorn_iters: int
    eps: float
