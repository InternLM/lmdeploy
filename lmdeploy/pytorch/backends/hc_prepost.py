# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch


class BaseHcPrePost(ABC):
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
        """Run sinkhorn and reduce HC states from ``[..., hc, dim]`` to
        ``[..., dim]``.
        """
        raise NotImplementedError

    @abstractmethod
    def pre_reduce(self, x: torch.Tensor, pre: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """Reduce HC hidden states from ``[..., hc, dim]`` to ``[..., dim]``."""
        raise NotImplementedError

    @abstractmethod
    def post_expand(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                    comb: torch.Tensor) -> torch.Tensor:
        """Expand one hidden state back to ``[..., hc, dim]``."""
        raise NotImplementedError


class BaseHcPrePostBuilder:

    @staticmethod
    @abstractmethod
    def build(hc_mult: int, sinkhorn_iters: int, eps: float) -> BaseHcPrePost:
        raise NotImplementedError
