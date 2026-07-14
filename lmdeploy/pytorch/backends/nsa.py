# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass
class NSAIndexMeta:
    """Meta info of NSAIndex layer."""
    cu_seqlen_q: Tensor
    q_seqlens: Tensor
    k_seqlens: Tensor
    block_offset: Tensor
    max_q_seqlen: int = None
    max_kv_seqlen: int = None


class BaseNSAIndexFP8(ABC):

    @abstractmethod
    def forward(self, q: Tensor, k: Tensor, weights: Tensor, k_cache: Tensor, k_s_cache: Tensor,
                meta: NSAIndexMeta) -> Tensor:
        """forward."""
        raise NotImplementedError('Not implemented.')


class BaseNSAIndexFP8Builder:

    @staticmethod
    @abstractmethod
    def build(topk: int, softmax_scale: float, block_size: int = 128, fill: int = -1) -> BaseNSAIndexFP8:
        """Build layer implementation."""
        raise NotImplementedError('Not implemented.')
