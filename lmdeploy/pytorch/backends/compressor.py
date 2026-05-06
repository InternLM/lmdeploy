# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class V4CompressorMetadata:
    """DeepSeek V4 compressor metadata."""

    cu_q_seqlens: torch.Tensor
    kv_seqlens: torch.Tensor
    block_offsets: torch.Tensor
    block_size: int
    max_kv_seqlen: int


class BaseV4Compressor(ABC):

    @abstractmethod
    def score_and_fill_state(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        state_ids: torch.Tensor,
        meta: V4CompressorMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def write_compressed_kv(
        self,
        compressed_kv: torch.Tensor,
        kv_cache: torch.Tensor | None,
        meta: V4CompressorMetadata,
        fp8_cache: torch.Tensor | None = None,
        kv_scale_cache: torch.Tensor | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaseV4CompressorBuilder:

    @staticmethod
    @abstractmethod
    def build(compress_ratio: int, overlap: bool, head_dim: int) -> BaseV4Compressor:
        raise NotImplementedError
