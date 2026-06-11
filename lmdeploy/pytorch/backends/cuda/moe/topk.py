# Copyright (c) OpenMMLab. All rights reserved.
"""CUDA softmax top-k implementations."""

import torch

from lmdeploy.pytorch.backends.default.moe import DefaultSoftmaxTopKImpl
from lmdeploy.pytorch.backends.moe import SoftmaxTopKBuilder

from ..batch_invariant import is_batch_invariant_policy_enabled


def _load_flashinfer():
    try:
        import flashinfer
    except ImportError as exc:
        raise RuntimeError(
            'enable_batch_invariant for CUDA SoftmaxTopK requires flashinfer with stable top_k support. '
            'Please install flashinfer or disable enable_batch_invariant.') from exc

    if not hasattr(flashinfer, 'top_k'):
        raise RuntimeError(
            'enable_batch_invariant for CUDA SoftmaxTopK requires flashinfer.top_k, '
            'but the installed flashinfer package does not provide it.')
    return flashinfer


def _canonicalize_topk(topk_weights: torch.Tensor, topk_ids: torch.Tensor):
    """Sort selected experts by expert id for canonical combine order."""
    id_order = torch.argsort(topk_ids, dim=-1, stable=True)
    topk_weights = topk_weights.gather(-1, id_order)
    topk_ids = topk_ids.gather(-1, id_order)
    return topk_weights, topk_ids


class CudaSoftmaxTopKImpl(DefaultSoftmaxTopKImpl):
    """CUDA softmax top-k implementation."""

    def __init__(self, top_k: int, dim: int = -1, n_groups: int = -1):
        super().__init__(top_k, dim, n_groups=n_groups)
        self.batch_invariant = is_batch_invariant_policy_enabled()
        self.flashinfer = _load_flashinfer() if self.batch_invariant else None

    def _flashinfer_topk(self, x: torch.Tensor, k: int):
        topk_weights, topk_ids = self.flashinfer.top_k(
            x,
            k,
            sorted=True,
            deterministic=True,
            tie_break=1,
        )
        return _canonicalize_topk(topk_weights, topk_ids)

    def forward(self, x: torch.Tensor):
        """forward."""
        if not self.batch_invariant:
            return super().forward(x)

        if self.dim not in (-1, x.dim() - 1):
            raise RuntimeError('Batch-invariant CUDA SoftmaxTopK currently supports only the last dimension.')

        routing_weights = torch.softmax(x, dim=-1, dtype=torch.float32)
        if self.n_groups > 0:
            assert routing_weights.shape[-1] % self.n_groups == 0, (
                f'{routing_weights.shape[-1]} cannot be divided by {self.n_groups}')
            per_group_top_k = self.top_k // self.n_groups
            group_size = routing_weights.shape[-1] // self.n_groups
            group_offsets = self.get_group_offsets(self.n_groups, group_size, routing_weights.device)
            grouped_weights = routing_weights.unflatten(-1, (self.n_groups, group_size))
            original_shape = grouped_weights.shape
            flat_weights = grouped_weights.reshape(-1, group_size)
            topk_weights, topk_ids = self._flashinfer_topk(flat_weights, per_group_top_k)
            topk_weights = topk_weights.reshape(*original_shape[:-1], per_group_top_k)
            topk_ids = topk_ids.reshape(*original_shape[:-1], per_group_top_k)
            topk_ids = (topk_ids + group_offsets).flatten(-2, -1)
            topk_weights = topk_weights.flatten(-2, -1)
            return _canonicalize_topk(topk_weights, topk_ids)

        return self._flashinfer_topk(routing_weights, self.top_k)


class CudaSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """CUDA softmax top-k implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1, n_groups: int = -1):
        """build."""
        return CudaSoftmaxTopKImpl(top_k, dim, n_groups=n_groups)
