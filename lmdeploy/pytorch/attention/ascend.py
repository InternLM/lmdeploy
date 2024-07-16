# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Type

import torch

from .base import AttentionBackend, AttentionImpl, AttentionMetadata


class AscendAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        raise 'ascend'

    @staticmethod
    def get_impl_cls() -> Type['AttentionImpl']:
        return AscendAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type['AttentionMetadata']:
        return AscendAttentionMetadata

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            block_size,
            num_heads,
            head_size,
        )

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            block_size,
            num_heads,
            head_size,
        )


class AscendAttentionMetadata(AttentionMetadata):
    pass


class AscendAttentionImpl(AttentionImpl[AscendAttentionMetadata]):
    pass
